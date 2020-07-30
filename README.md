# Churn in telecom

With the rapid development of telecommunication industry, the service providers are inclined more towards expansion of the subscriber base. To meet the need of surviving in the competitive environment, the retention of existing customers has become a huge challenge. In the survey done in the Telecom industry, it is stated that the cost of acquiring a new customer is far more that retaining the existing one. Therefore, by collecting knowledge from the telecom industries can help in predicting the association of the customers as whether or not they will leave the company. The required action needs to be undertaken by the telecom industries in order to initiate the acquisition of their associated customers for making their market value stagnant.The goal of this article is to apply analytical techniques, using Elastic Stack, to predict a customer churn and analyse the churning and non churning customers.
This article has been inspired from [Elastic Blog](https://www.elastic.co/blog/using-elastic-supervised-machine-learning-for-binary-classification)


# Why supervised learning ? 

Supervised machine learning trains a model from labeled data (means a dataset with a list of customers with a label to show if the customer has churned or not). This is in contrast to unsupervised learning where the model learns from unlabeled data. In [data frame analytics](https://www.elastic.co/guide/en/machine-learning/master/ml-dfa-overview.html), Elastic Stack give you the toolset to create supervised models without knowledge of the underlying algorithms. You can then use the trained models to make predictions on unobserved data (via inference). The model applies what it learned through training to provide a prediction.

Why use this instead of [anomaly detection](https://www.elastic.co/guide/en/machine-learning/master/xpack-ml.html) ? Anomaly detection, an example of unsupervised learning, is excellent at learning patterns and predicting metrics based on time series data. This is powerful for many use cases, but what if we wanted to identify bad actors by detecting suspicious domains ? What about automatically detecting language and applying the appropriate analyzers for better search ? 

These use cases need richer feature sets to provide accurate predictions. They also have known examples from which to learn. Meaning, there exists a training dataset where the data is labeled with the correct prediction. Transforms supplies the tools needed to build complex feature sets. Data frame analytics can train supervised machine learning models. We can use both to deploy the feature generation and supervised models for production use on streaming data.

# Supervised learning example : binary classification

Let’s build and use a binary classification model from scratch. Don’t be intimidated. No data science degree is necessary. We will use the building blocks within the Elastic Stack to create our feature set, train our model, and deploy it in an ingest pipeline to perform inference. Using an ingest processor allows our data to be enriched with the model’s predictions before being indexed.

In this example, we will predict whether a telecom customer is likely to churn based on features of their customer metadata and call logs. We’ll use examples of churned and not churned customers as the training dataset. 

# Setting up your environment

To follow along, start a free trial of [Elastic Cloud](https://www.elastic.co/elasticsearch/service), spin up a [new deployment](https://www.elastic.co/guide/en/cloud/current/ec-create-deployment.html) or simply download Elasticsearch, Kibana and Logstash from [Elastic](https://www.elastic.co) web site, download the [calls.csv](./call.7z) and [customers.csv](./customers.7z) files, and upload them via the [CSV upload feature](https://www.elastic.co/blog/importing-csv-and-log-data-into-elasticsearch-with-file-data-visualizer) (or use Logstash). Data is derived from a dataset referenced in various sources : [openml](https://www.openml.org/d/40701), [kaggle](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset), [Larose 2014](http://dataminingconsultant.com/DKD2e_data_sets.zip). This dataset was disaggregated into line items, and random features were added. Phone numbers and other data are fabricated and any resemblance to real data is coincidental. 

Once uploaded you should have data that looks like this :

In the customers index :

```json
{
  "record_id": 332,
  "@timestamp": "2020-09-08T10:16:15.019Z",
  "phone_number": "2253803437",
  "account_age": 86,
  "number_vmail_messages": 0,
  "customer_service_calls": 7,
  "voice_mail_plan": "no",
  "state": "AB",
  "international_plan": "no",
  "churn": 1,
}
```
In the calls index :

```json
{
  "record_id": 1,
  "@timestamp": "2020-07-07T07:21:40.102Z",
  "phone_number": "2253803437",
  "dialled_number": "2253283266",
  "call_charges": 0.34497474848639376,
  "call_duration": 3.3484235812459193
}
```

# Building the feature set

First, we want to build out our feature set. We have two sources of data: customer metadata and call logs. This requires the use of transforms and the enrich processor. This combination allows us to merge the metadata along with the call information in the other index.

The enrichment policy and pipeline. (Execute the commands in the Kibana dev console)

````json
# Enrichment policy for customer metadata

PUT /_enrich/policy/customer_metadata
{
  "match": {
    "indices": "customers",
    "match_field": "phone_number",
    "enrich_fields": [
      "account_age",
      "churn",
      "customer_service_calls",
      "international_plan",
      "number_vmail_messages",
      "state",
      "voice_mail_plan"
    ]
  }
}

# Execute the policy so we can populate with the metadata

POST /_enrich/policy/customer_metadata/_execute

# Our enrichment pipeline for generating features

PUT _ingest/pipeline/customer_metadata
{
  "description": "Adds metadata about customers by lookup on phone_number",
  "processors": [
    {
      "enrich": {
        "policy_name": "customer_metadata",
        "field": "phone_number",
        "target_field": "customer",
        "max_matches": 1
      }
    }
  ]
}
```

We can now build a transform that utilizes our new pipeline. Here is the transform definition:

```json
# transform for enriching the data for training

PUT _transform/customer_churn_transform
{
  "source": {
    "index": [
      "calls"
    ]
  },
  "dest": {
    "index": "churn_transform_index",
    "pipeline": "customer_metadata"
  },
  "pivot": {
    "group_by": {
      "phone_number": {
        "terms": {
          "field": "phone_number"
        }
      }
    },
    "aggregations": {
      "call_charges": {
        "sum": {
          "field": "call_charges"
        }
      },
      "call_duration": {
        "sum": {
          "field": "call_duration"
        }
      },
      "call_count": {
        "value_count": {
          "field": "dialled_number"
        }
      }
    }
  }
}
```
Now that the transform is created, we can start it and see its progress on the Transforms page under Stack Management.

<img src="./screens/tranform.png" align="middle">


Once tranfor executed  you should have data that looks like this :

In the customers churn_transform_index :

```json
{
  "phone_number": "2253271058",
  "call_count": 253,
  "call_charges": 45.72,
  "call_duration": 488.5,
  "customer": {
    "voice_mail_plan": "no",
    "number_vmail_messages": 0,
    "churn": 0,
    "account_age": 112,
    "phone_number": "2253271058",
    "state": "AB",
    "international_plan": "no",
    "customer_service_calls": 0
  }
}
```

### Using New Platform config in a new plugin

After setting the config schema for your plugin, you might want to reach the configuration in the plugin.
It is provided as part of the [PluginInitializerContext](../../docs/development/core/server/kibana-plugin-core-server.plugininitializercontext.md)
in the *constructor* of the plugin:

```ts
// myPlugin/(public|server)/index.ts

import { PluginInitializerContext } from 'kibana/server';
import { MyPlugin } from './plugin';

export function plugin(initializerContext: PluginInitializerContext) {
  return new MyPlugin(initializerContext);
}
```

```ts
// myPlugin/(public|server)/plugin.ts

import { Observable } from 'rxjs';
import { first } from 'rxjs/operators';
import { CoreSetup, Logger, Plugin, PluginInitializerContext, PluginName } from 'kibana/server';
import { MyPlugin } from './plugin';

export class MyPlugin implements Plugin {
  private readonly config$: Observable<MyPluginConfig>;
  private readonly log: Logger;

  constructor(private readonly initializerContext: PluginInitializerContext) {
    this.log = initializerContext.logger.get();
    this.config$ = initializerContext.config.create();
  }

  public async setup(core: CoreSetup, deps: Record<PluginName, unknown>) {
    const isEnabled = await this.config$.pipe(first()).toPromise();
    ...
  }
  ...
}
}
```

Additionally, some plugins need to read other plugins' config to act accordingly (like timing out a request, matching ElasticSearch's timeout). For those use cases, the plugin can rely on the *globalConfig* and *env* properties in the context:

```ts
export class MyPlugin implements Plugin {
...
  public async setup(core: CoreSetup, deps: Record<PluginName, unknown>) {
    const { mode: { dev }, packageInfo: { version } } = this.initializerContext.env;
    const { elasticsearch: { shardTimeout }, path: { data } } = await this.initializerContext.config.legacy.globalConfig$
      .pipe(first()).toPromise();
    ...
  }
```

### Using New Platform config from a Legacy plugin

During the migration process, you'll want to migrate your schema to the new
format. However, legacy plugins cannot directly get access to New Platform's
config service due to the way that config is tied to the `kibana.json` file
(which does not exist for legacy plugins).

There is a workaround though:

- Create a New Platform plugin that contains your plugin's config schema in the new format
- Expose the config from the New Platform plugin in its setup contract
- Read the config from the setup contract in your legacy plugin

#### Create a New Platform plugin

For example, if wanted to move the legacy `timelion` plugin's configuration to
the New Platform, we could create a NP plugin with the same name in
`src/plugins/timelion` with the following files:

```json5
// src/plugins/timelion/kibana.json
{
  "id": "timelion",
  "server": true
}
```

```ts
// src/plugins/timelion/server/index.ts
import { schema, TypeOf } from '@kbn/config-schema';
import { PluginInitializerContext } from 'src/core/server';
import { TimelionPlugin } from './plugin';

export const config = {
  schema: schema.object({
    enabled: schema.boolean({ defaultValue: true }),
  });
}

export const plugin = (initContext: PluginInitializerContext) => new TimelionPlugin(initContext);

export type TimelionConfig = TypeOf<typeof config.schema>;
export { TimelionSetup } from './plugin';
```

```ts
// src/plugins/timelion/server/plugin.ts
import { PluginInitializerContext, Plugin, CoreSetup } from '../../core/server';
import { TimelionConfig } from '.';

export class TimelionPlugin implements Plugin<TimelionSetup> {
  constructor(private readonly initContext: PluginInitializerContext) {}

  public setup(core: CoreSetup) {
    return {
      __legacy: {
        config$: this.initContext.config.create<TimelionConfig>(),
      },
    };
  }

  public start() {}
  public stop() {}
}

export interface TimelionSetup {
  /** @deprecated */
  __legacy: {
    config$: Observable<TimelionConfig>;
  };
}
```

With the New Platform plugin in place, you can then read this `config$`
Observable from your legacy plugin:

```ts
import { take } from 'rxjs/operators';

new kibana.Plugin({
  async init(server) {
    const { config$ } = server.newPlatform.setup.plugins.timelion;
    const currentConfig = await config$.pipe(take(1)).toPromise();
  }
});
```

## HTTP Routes

In the legacy platform, plugins have direct access to the Hapi `server` object
which gives full access to all of Hapi's API. In the New Platform, plugins have
access to the
[HttpServiceSetup](/docs/development/core/server/kibana-plugin-core-server.httpservicesetup.md)
interface, which is exposed via the
[CoreSetup](/docs/development/core/server/kibana-plugin-core-server.coresetup.md)
object injected into the `setup` method of server-side plugins.

This interface has a different API with slightly different behaviors.

- All input (body, query parameters, and URL parameters) must be validated using
  the `@kbn/config-schema` package. If no validation schema is provided, these
  values will be empty objects.
- All exceptions thrown by handlers result in 500 errors. If you need a specific
  HTTP error code, catch any exceptions in your handler and construct the
  appropriate response using the provided response factory. While you can
  continue using the `boom` module internally in your plugin, the framework does
  not have native support for converting Boom exceptions into HTTP responses.

Because of the incompatibility between the legacy and New Platform HTTP Route
API's it might be helpful to break up your migration work into several stages.

### 1. Legacy route registration

```ts
// legacy/plugins/myplugin/index.ts
import Joi from 'joi';

new kibana.Plugin({
  init(server) {
    server.route({
      path: '/api/demoplugin/search',
      method: 'POST',
      options: {
        validate: {
          payload: Joi.object({
            field1: Joi.string().required(),
          }),
        }
      },
      handler(req, h) {
        return { message: `Received field1: ${req.payload.field1}` };
      }
    });
  }
});
```

### 2. New Platform shim using legacy router

Create a New Platform shim and inject the legacy `server.route` into your
plugin's setup function.

```ts
// legacy/plugins/demoplugin/index.ts
import { Plugin, LegacySetup } from './server/plugin';
export default (kibana) => {
  return new kibana.Plugin({
    id: 'demo_plugin',

    init(server) {
      // core shim
      const coreSetup: server.newPlatform.setup.core;
      const pluginSetup = {};
      const legacySetup: LegacySetup = {
        route: server.route
      };

      new Plugin().setup(coreSetup, pluginSetup, legacySetup);
    }
  }
}
```

```ts
// legacy/plugins/demoplugin/server/plugin.ts
import { CoreSetup } from 'src/core/server';
import { Legacy } from 'kibana';

export interface LegacySetup {
  route: Legacy.Server['route'];
};

export interface DemoPluginsSetup {};

export class Plugin {
  public setup(core: CoreSetup, plugins: DemoPluginsSetup, __LEGACY: LegacySetup) {
    __LEGACY.route({
      path: '/api/demoplugin/search',
      method: 'POST',
      options: {
        validate: {
          payload: Joi.object({
            field1: Joi.string().required(),
          }),
        }
      },
      async handler(req) {
        return { message: `Received field1: ${req.payload.field1}` };
      },
    });
  }
}
```

### 3. New Platform shim using New Platform router

We now switch the shim to use the real New Platform HTTP API's in `coreSetup`
instead of relying on the legacy `server.route`. Since our plugin is now using
the New Platform API's we are guaranteed that our HTTP route handling is 100%
compatible with the New Platform. As a result, we will also have to adapt our
route registration accordingly.

```ts
// legacy/plugins/demoplugin/index.ts
import { Plugin } from './server/plugin';
export default (kibana) => {
  return new kibana.Plugin({
    id: 'demo_plugin',

    init(server) {
      // core shim
      const coreSetup = server.newPlatform.setup.core;
      const pluginSetup = {};

      new Plugin().setup(coreSetup, pluginSetup);
    }
  }
}
```

```ts
// legacy/plugins/demoplugin/server/plugin.ts
import { schema } from '@kbn/config-schema';
import { CoreSetup } from 'src/core/server';

export interface DemoPluginsSetup {};

class Plugin {
  public setup(core: CoreSetup, pluginSetup: DemoPluginSetup) {
    const router = core.http.createRouter();
    router.post(
      {
        path: '/api/demoplugin/search',
        validate: {
          body: schema.object({
            field1: schema.string(),
          }),
        }
      },
      (context, req, res) => {
        return res.ok({
          body: {
            message: `Received field1: ${req.body.field1}`
          }
        });
      }
    )
  }
}
```

If your plugin still relies on throwing Boom errors from routes, you can use the `router.handleLegacyErrors`
as a temporary solution until error migration is complete:

```ts
// legacy/plugins/demoplugin/server/plugin.ts
import { schema } from '@kbn/config-schema';
import { CoreSetup } from 'src/core/server';

export interface DemoPluginsSetup {};

class Plugin {
  public setup(core: CoreSetup, pluginSetup: DemoPluginSetup) {
    const router = core.http.createRouter();
    router.post(
      {
        path: '/api/demoplugin/search',
        validate: {
          body: schema.object({
            field1: schema.string(),
          }),
        }
      },
      router.handleLegacyErrors((context, req, res) => {
        throw Boom.notFound('not there'); // will be converted into proper New Platform error
      })
    )
  }
}
```

#### 4. New Platform plugin

As the final step we delete the shim and move all our code into a New Platform
plugin. Since we were already consuming the New Platform API's no code changes
are necessary inside `plugin.ts`.

```ts
// Move legacy/plugins/demoplugin/server/plugin.ts -> plugins/demoplugin/server/plugin.ts
```

### Accessing Services

Services in the Legacy Platform were typically available via methods on either
`server.plugins.*`, `server.*`, or `req.*`. In the New Platform, all services
are available via the `context` argument to the route handler. The type of this
argument is the
[RequestHandlerContext](/docs/development/core/server/kibana-plugin-core-server.requesthandlercontext.md).
The APIs available here will include all Core services and any services
registered by plugins this plugin depends on.

```ts
new kibana.Plugin({
  init(server) {
    const { callWithRequest } = server.plugins.elasticsearch.getCluster('data');

    server.route({
      path: '/api/my-plugin/my-route',
      method: 'POST',
      async handler(req, h) {
        const results = await callWithRequest(req, 'search', query);
        return { results };
      }
    });
  }
});

class Plugin {
  public setup(core) {
    const router = core.http.createRouter();
    router.post(
      {
        path: '/api/my-plugin/my-route',
      },
      async (context, req, res) => {
        const results = await context.elasticsearch.dataClient.callAsCurrentUser('search', query);
        return res.ok({
          body: { results }
        });
      }
    )
  }
}
```

### Migrating Hapi "pre" handlers

In the Legacy Platform, routes could provide a "pre" option in their config to
register a function that should be run prior to the route handler. These
"pre" handlers allow routes to share some business logic that may do some
pre-work or validation. In Kibana, these are often used for license checks.

The Kibana Platform's HTTP interface does not provide this functionality,
however it is simple enough to port over using a higher-order function that can
wrap the route handler.

#### Simple example

In this simple example, a pre-handler is used to either abort the request with
an error or continue as normal. This is a simple "gate-keeping" pattern.

```ts
// Legacy pre-handler
const licensePreRouting = (request) => {
  const licenseInfo = getMyPluginLicenseInfo(request.server.plugins.xpack_main);
  if (!licenseInfo.isOneOf(['gold', 'platinum', 'trial'])) {
    throw Boom.forbidden(`You don't have the right license for MyPlugin!`);
  }
}

server.route({
  method: 'GET',
  path: '/api/my-plugin/do-something',
  config: {
    pre: [{ method: licensePreRouting }]
  },
  handler: (req) => {
    return doSomethingInteresting();
  }
})
```

In the Kibana Platform, the same functionality can be acheived by creating a
function that takes a route handler (or factory for a route handler) as an
argument and either invokes it in the successful case or returns an error
response in the failure case.

We'll call this a "high-order handler" similar to the "high-order component"
pattern common in the React ecosystem.

```ts
// New Platform high-order handler
const checkLicense = <P, Q, B>(
  handler: RequestHandler<P, Q, B, RouteMethod>
): RequestHandler<P, Q, B, RouteMethod> => {
  return (context, req, res) => {
    const licenseInfo = getMyPluginLicenseInfo(context.licensing.license);

    if (licenseInfo.hasAtLeast('gold')) {
      return handler(context, req, res);
    } else {
      return res.forbidden({ body: `You don't have the right license for MyPlugin!` });
    }
  }
}

router.get(
  { path: '/api/my-plugin/do-something', validate: false },
  checkLicense(async (context, req, res) => {
    const results = doSomethingInteresting();
    return res.ok({ body: results });
  }),
)
```

#### Full Example

In some cases, the route handler may need access to data that the pre-handler
retrieves. In this case, you can utilize a handler _factory_ rather than a raw
handler.

```ts
// Legacy pre-handler
const licensePreRouting = (request) => {
  const licenseInfo = getMyPluginLicenseInfo(request.server.plugins.xpack_main);
  if (licenseInfo.isOneOf(['gold', 'platinum', 'trial'])) {
    // In this case, the return value of the pre-handler is made available on
    // whatever the 'assign' option is in the route config.
    return licenseInfo;
  } else {
    // In this case, the route handler is never called and the user gets this
    // error message
    throw Boom.forbidden(`You don't have the right license for MyPlugin!`);
  }
}

server.route({
  method: 'GET',
  path: '/api/my-plugin/do-something',
  config: {
    pre: [{ method: licensePreRouting, assign: 'licenseInfo' }]
  },
  handler: (req) => {
    const licenseInfo = req.pre.licenseInfo;
    return doSomethingInteresting(licenseInfo);
  }
})
```

In many cases, it may be simpler to duplicate the function call
to retrieve the data again in the main handler. In this other cases, you can
utilize a handler _factory_ rather than a raw handler as the argument to your
high-order handler. This way the high-order handler can pass arbitrary arguments
to the route handler.

```ts
// New Platform high-order handler
const checkLicense = <P, Q, B>(
  handlerFactory: (licenseInfo: MyPluginLicenseInfo) => RequestHandler<P, Q, B, RouteMethod>
): RequestHandler<P, Q, B, RouteMethod> => {
  return (context, req, res) => {
    const licenseInfo = getMyPluginLicenseInfo(context.licensing.license);

    if (licenseInfo.hasAtLeast('gold')) {
      const handler = handlerFactory(licenseInfo);
      return handler(context, req, res);
    } else {
      return res.forbidden({ body: `You don't have the right license for MyPlugin!` });
    }
  }
}

router.get(
  { path: '/api/my-plugin/do-something', validate: false },
  checkLicense(licenseInfo => async (context, req, res) => {
    const results = doSomethingInteresting(licenseInfo);
    return res.ok({ body: results });
  }),
)
```

## Chrome

In the Legacy Platform, the `ui/chrome` import contained APIs for a very wide
range of features. In the New Platform, some of these APIs have changed or moved
elsewhere.

| Legacy Platform                                       | New Platform                                                                                                                        | Notes                                                                                                                                                                            |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `chrome.addBasePath`                                  | [`core.http.basePath.prepend`](/docs/development/core/public/kibana-plugin-public.httpservicebase.basepath.md)                      |                                                                                                                                                                                  |
| `chrome.breadcrumbs.set`                              | [`core.chrome.setBreadcrumbs`](/docs/development/core/public/kibana-plugin-public.chromestart.setbreadcrumbs.md)                    |                                                                                                                                                                                  |
| `chrome.getUiSettingsClient`                          | [`core.uiSettings`](/docs/development/core/public/kibana-plugin-public.uisettingsclient.md)                                         |                                                                                                                                                                                  |
| `chrome.helpExtension.set`                            | [`core.chrome.setHelpExtension`](/docs/development/core/public/kibana-plugin-public.chromestart.sethelpextension.md)                |                                                                                                                                                                                  |
| `chrome.setVisible`                                   | [`core.chrome.setIsVisible`](/docs/development/core/public/kibana-plugin-public.chromestart.setisvisible.md)                        |                                                                                                                                                                                  |
| `chrome.getInjected`                                  | [`core.injectedMetadata.getInjected`](/docs/development/core/public/kibana-plugin-public.coresetup.injectedmetadata.md) (temporary) | A temporary API is available to read injected vars provided by legacy plugins. This will be removed after [#41990](https://github.com/elastic/kibana/issues/41990) is completed. |
| `chrome.setRootTemplate` / `chrome.setRootController` | --                                                                                                                                  | Use application mounting via `core.application.register` (not currently avaiable to legacy plugins).                                                                             |
| `chrome.navLinks.update`                              | [`core.appbase.updater`](/docs/development/core/public/kibana-plugin-public.appbase.updater_.md)                                    | Use the `updater$` property when registering your application via `core.application.register`                                                                                    |

In most cases, the most convenient way to access these APIs will be via the
[AppMountContext](/docs/development/core/public/kibana-plugin-public.appmountcontext.md)
object passed to your application when your app is mounted on the page.

### Updating an application navlink

In the legacy platform, the navlink could be updated using `chrome.navLinks.update`

```ts
uiModules.get('xpack/ml').run(() => {
  const showAppLink = xpackInfo.get('features.ml.showLinks', false);
  const isAvailable = xpackInfo.get('features.ml.isAvailable', false);

  const navLinkUpdates = {
    // hide by default, only show once the xpackInfo is initialized
    hidden: !showAppLink,
    disabled: !showAppLink || (showAppLink && !isAvailable),
  };

  npStart.core.chrome.navLinks.update('ml', navLinkUpdates);
});
```

In the new platform, navlinks should not be updated directly. Instead, it is now possible to add an `updater` when 
registering an application to change the application or the navlink state at runtime.

```ts
// my_plugin has a required dependencie to the `licensing` plugin
interface MyPluginSetupDeps {
  licensing: LicensingPluginSetup;
}

export class MyPlugin implements Plugin {
  setup({ application }, { licensing }: MyPluginSetupDeps) {
    const updater$ = licensing.license$.pipe(
      map(license => {
        const { hidden, disabled } = calcStatusFor(license);
        if (hidden) return { navLinkStatus: AppNavLinkStatus.hidden };
        if (disabled) return { navLinkStatus: AppNavLinkStatus.disabled };
        return { navLinkStatus: AppNavLinkStatus.default };
      })
    );

    application.register({
      id: 'my-app',
      title: 'My App',
      updater$,
      async mount(params) {
        const { renderApp } = await import('./application');
        return renderApp(params);
      },
    });
  }
```

## Chromeless Applications

In Kibana, a "chromeless" application is one where the primary Kibana UI components
such as header or navigation can be hidden. In the legacy platform these were referred to
as "hidden" applications, and were set via the `hidden` property in a Kibana plugin.
Chromeless applications are also not displayed in the left navbar.

To mark an application as chromeless, specify `chromeless: false` when registering your application
to hide the chrome UI when the application is mounted:

```ts
application.register({
  id: 'chromeless',
  chromeless: true,
  async mount(context, params) {
    /* ... */
  },
});
```

If you wish to render your application at a route that does not follow the `/app/${appId}` pattern,
this can be done via the `appRoute` property. Doing this currently requires you to register a server
route where you can return a bootstrapped HTML page for your application bundle. Instructions on
registering this server route is covered in the next section: [Render HTML Content](#render-html-content).

```ts
application.register({
  id: 'chromeless',
  appRoute: '/chromeless',
  chromeless: true,
  async mount(context, params) {
    /* ... */
  },
});
```

## Render HTML Content

You can return a blank HTML page bootstrapped with the core application bundle from an HTTP route handler
via the `httpResources` service. You may wish to do this if you are rendering a chromeless application with a
custom application route or have other custom rendering needs.

```typescript
httpResources.register(
  { path: '/chromeless', validate: false },
  (context, request, response) => {
    //... some logic
    return response.renderCoreApp();
  }
);
```

You can also specify to exclude user data from the bundle metadata. User data
comprises all UI Settings that are *user provided*, then injected into the page.
You may wish to exclude fetching this data if not authorized or to slim the page
size.

```typescript
httpResources.register(
  { path: '/', validate: false, options: { authRequired: false } },
  (context, request, response) => {
    //... some logic
    return response.renderAnonymousCoreApp();
  }
);
```

## Saved Objects types

In the legacy platform, saved object types were registered using static definitions in the `uiExports` part of
the plugin manifest.

In the new platform, all these registration are to be performed programmatically during your plugin's `setup` phase,
using the core `savedObjects`'s `registerType` setup API.

The most notable difference is that in the new platform, the type registration is performed in a single call to 
`registerType`, passing a new `SavedObjectsType` structure that is a superset of the legacy `schema`, `migrations` 
`mappings` and `savedObjectsManagement`.

### Concrete example

Let say we have the following in a legacy plugin:

```js
// src/legacy/core_plugins/my_plugin/index.js
import mappings from './mappings.json';
import { migrations } from './migrations';

new kibana.Plugin({
  init(server){
    // [...]
  },
  uiExports: {
    mappings,
    migrations,
    savedObjectSchemas: {
      'first-type': {
        isNamespaceAgnostic: true,
      },
      'second-type': {
        isHidden: true,
      },
    },
    savedObjectsManagement: {
      'first-type': {
        isImportableAndExportable: true,
        icon: 'myFirstIcon',
        defaultSearchField: 'title',
        getTitle(obj) {
          return obj.attributes.title;
        },
        getEditUrl(obj) {
          return `/some-url/${encodeURIComponent(obj.id)}`;
        },
      },
      'second-type': {
        isImportableAndExportable: false,
        icon: 'mySecondIcon',
        getTitle(obj) {
          return obj.attributes.myTitleField;
        },
        getInAppUrl(obj) {
          return {
            path: `/some-url/${encodeURIComponent(obj.id)}`,
            uiCapabilitiesPath: 'myPlugin.myType.show',
          };
        },
      },
    },
  },
})
```

```json
// src/legacy/core_plugins/my_plugin/mappings.json
{
  "first-type": {
    "properties": {
      "someField": {
        "type": "text"
      },
      "anotherField": {
        "type": "text"
      }
    }
  },
  "second-type": {
    "properties": {
      "textField": {
        "type": "text"
      },
      "boolField": {
        "type": "boolean"
      }
    }
  }
}
```

```js
// src/legacy/core_plugins/my_plugin/migrations.js
export const migrations = {
  'first-type': {
    '1.0.0': migrateFirstTypeToV1,
    '2.0.0': migrateFirstTypeToV2,
  },
  'second-type': {
    '1.5.0': migrateSecondTypeToV15,
  }
}
```

To migrate this, we will have to regroup the declaration per-type. That would become:

First type:
 
```typescript
// src/plugins/my_plugin/server/saved_objects/first_type.ts
import { SavedObjectsType } from 'src/core/server';

export const firstType: SavedObjectsType = {
  name: 'first-type',
  hidden: false,
  namespaceType: 'agnostic',
  mappings: {
    properties: {
      someField: {
        type: 'text',
      },
      anotherField: {
        type: 'text',
      },
    },
  },
  migrations: {
    '1.0.0': migrateFirstTypeToV1,
    '2.0.0': migrateFirstTypeToV2,
  },
  management: {
    importableAndExportable: true,
    icon: 'myFirstIcon',
    defaultSearchField: 'title',
    getTitle(obj) {
      return obj.attributes.title;
    },
    getEditUrl(obj) {
      return `/some-url/${encodeURIComponent(obj.id)}`;
    },
  },
};
```

Second type:

```typescript
// src/plugins/my_plugin/server/saved_objects/second_type.ts
import { SavedObjectsType } from 'src/core/server';

export const secondType: SavedObjectsType = {
  name: 'second-type',
  hidden: true,
  namespaceType: 'single',
  mappings: {
    properties: {
      textField: {
        type: 'text',
      },
      boolField: {
        type: 'boolean',
      },
    },
  },
  migrations: {
    '1.5.0': migrateSecondTypeToV15,
  },
  management: {
    importableAndExportable: false,
    icon: 'mySecondIcon',
    getTitle(obj) {
      return obj.attributes.myTitleField;
    },
    getInAppUrl(obj) {
      return {
        path: `/some-url/${encodeURIComponent(obj.id)}`,
        uiCapabilitiesPath: 'myPlugin.myType.show',
      };
    },
  },
};
```

Registration in the plugin's setup phase:

```typescript
// src/plugins/my_plugin/server/plugin.ts
import { firstType, secondType } from './saved_objects';

export class MyPlugin implements Plugin {
  setup({ savedObjects }) {
    savedObjects.registerType(firstType);
    savedObjects.registerType(secondType);
  }
}
```

### Changes in structure compared to legacy

The NP `registerType` expected input is very close to the legacy format. However, there are some minor changes:

- The `schema.isNamespaceAgnostic` property has been renamed: `SavedObjectsType.namespaceType`. It no longer accepts a boolean but instead an enum of 'single', 'multiple', or 'agnostic' (see [SavedObjectsNamespaceType](/docs/development/core/server/kibana-plugin-core-server.savedobjectsnamespacetype.md)).

- The `schema.indexPattern` was accepting either a `string` or a `(config: LegacyConfig) => string`. `SavedObjectsType.indexPattern` only accepts a string, as you can access the configuration during your plugin's setup phase.

- The `savedObjectsManagement.isImportableAndExportable` property has been renamed: `SavedObjectsType.management.importableAndExportable`

- The migration function signature has changed:
In legacy, it was `(doc: SavedObjectUnsanitizedDoc, log: SavedObjectsMigrationLogger) => SavedObjectUnsanitizedDoc;`
In new platform, it is now `(doc: SavedObjectUnsanitizedDoc, context: SavedObjectMigrationContext) => SavedObjectUnsanitizedDoc;`

With context being:

```typescript
export interface SavedObjectMigrationContext {
  log: SavedObjectsMigrationLogger;
}
```

The changes is very minor though. The legacy migration:

```js
const migration = (doc, log) => {...}
```

Would be converted to:

```typescript
const migration: SavedObjectMigrationFn<OldAttributes, MigratedAttributes> = (doc, { log }) => {...}
```

### Remarks

The `registerType` API will throw if called after the service has started, and therefor cannot be used from 
legacy plugin code. Legacy plugins should use the legacy savedObjects service and the legacy way to register
saved object types until migrated.

## UiSettings
UiSettings defaults registration performed during `setup` phase via `core.uiSettings.register` API.

```js
// Before:
uiExports: {
  uiSettingDefaults: {
    'my-plugin:my-setting': {
      name: 'just-work',
      value: true,
      description: 'make it work',
      category: ['my-category'],
    },
  }
}
```

```ts
// After:
// src/plugins/my-plugin/server/plugin.ts
setup(core: CoreSetup){
  core.uiSettings.register({
    'my-plugin:my-setting': {
      name: 'just-work',
      value: true,
      description: 'make it work',
      category: ['my-category'],
      schema: schema.boolean(),
    },
  })
}
```

## Elasticsearch client

The new elasticsearch client is a thin wrapper around `@elastic/elasticsearch`'s `Client` class. Even if the API
is quite close to the legacy client Kibana was previously using, there are some subtle changes to take into account 
during migration.

[Official documentation](https://www.elastic.co/guide/en/elasticsearch/client/javascript-api/current/index.html) 

### Client API Changes

The most significant changes for the consumers are the following:

- internal / current user client accessors has been renamed and are now properties instead of functions
  - `callAsInternalUser('ping')` -> `asInternalUser.ping()`
  - `callAsCurrentUser('ping')` -> `asCurrentUser.ping()`

- the API now reflects the `Client`'s instead of leveraging the string-based endpoint names the `LegacyAPICaller` was using

before:

```ts
const body = await client.callAsInternalUser('indices.get', { index: 'id' });
```

after:

```ts
const { body } = await client.asInternalUser.indices.get({ index: 'id' });
```

- calling any ES endpoint now returns the whole response object instead of only the body payload

before:

```ts
const body = await legacyClient.callAsInternalUser('get', { id: 'id' });
```

after:

```ts
const { body } = await client.asInternalUser.get({ id: 'id' });
```

Note that more information from the ES response is available:

```ts
const {
  body,        // response payload
  statusCode,  // http status code of the response
  headers,     // response headers
  warnings,    // warnings returned from ES
  meta         // meta information about the request, such as request parameters, number of attempts and so on
} = await client.asInternalUser.get({ id: 'id' });
```

- all API methods are now generic to allow specifying the response body type

before:

```ts
const body: GetResponse = await legacyClient.callAsInternalUser('get', { id: 'id' });
```

after:

```ts
// body is of type `GetResponse`
const { body } = await client.asInternalUser.get<GetResponse>({ id: 'id' });
// fallback to `Record<string, any>` if unspecified
const { body } = await client.asInternalUser.get({ id: 'id' });
```

- the returned error types changed 

There are no longer specific errors for every HTTP status code (such as `BadRequest` or `NotFound`). A generic
`ResponseError` with the specific `statusCode` is thrown instead.

before:

```ts
import { errors } from 'elasticsearch';
try {
  await legacyClient.callAsInternalUser('ping');
} catch(e) {
  if(e instanceof errors.NotFound) {
    // do something
  }
}
``` 

after:

```ts
import { errors } from '@elastic/elasticsearch';
try {
  await client.asInternalUser.ping();
} catch(e) {
  if(e instanceof errors.ResponseError && e.statusCode === 404) {
    // do something
  }
  // also possible, as all errors got a name property with the name of the class,
  // so this slightly better in term of performances
  if(e.name === 'ResponseError' && e.statusCode === 404) {
    // do something
  }
}
```

- the parameter property names changed from camelCase to snake_case

Even if technically, the javascript client accepts both formats, the typescript definitions are only defining the snake_case
properties.

before:

```ts
legacyClient.callAsCurrentUser('get', {
  id: 'id',
  storedFields: ['some', 'fields'],
})
```

after:

```ts
client.asCurrentUser.get({
  id: 'id',
  stored_fields: ['some', 'fields'],
})
```

- the request abortion API changed

All promises returned from the client API calls now have an `abort` method that can be used to cancel the request.

before:

```ts
const controller = new AbortController();
legacyClient.callAsCurrentUser('ping', {}, {
  signal: controller.signal,
})
// later
controller.abort();
```

after:

```ts
const request = client.asCurrentUser.ping();
// later
request.abort();
```

- it is now possible to override headers when performing specific API calls.

Note that doing so is strongly discouraged due to potential side effects with the ES service internal
behavior when scoping as the internal or as the current user.

```ts
const request = client.asCurrentUser.ping({}, { 
  headers: {
    authorization: 'foo',
    custom: 'bar',
  }
});
```

Please refer to the  [Breaking changes list](https://www.elastic.co/guide/en/elasticsearch/client/javascript-api/current/breaking-changes.html)
for more information about the changes between the legacy and new client.

### Accessing the client from a route handler

Apart from the API format change, accessing the client from within a route handler
did not change. As it was done for the legacy client, a preconfigured scoped client 
bound to the request is accessible using `core` context provider:

before:

```ts
router.get(
  {
    path: '/my-route',
  },
  async (context, req, res) => {
    const { client } = context.core.elasticsearch.legacy;
    // call as current user
    const res = await client.callAsCurrentUser('ping');
    // call as internal user
    const res2 = await client.callAsInternalUser('search', options);
    return res.ok({ body: 'ok' });
  }
);
```

after:

```ts
router.get(
  {
    path: '/my-route',
  },
  async (context, req, res) => {
    const { client } = context.core.elasticsearch;
    // call as current user
    const res = await client.asCurrentUser.ping();
    // call as internal user
    const res2 = await client.asInternalUser.search(options);
    return res.ok({ body: 'ok' });
  }
);
```

### Creating a custom client

Note that the `plugins` option is now longer available on the new client. As the API is now exhaustive, adding custom
endpoints using plugins should no longer be necessary.

The API to create custom clients did not change much:

before:

```ts
const customClient = coreStart.elasticsearch.legacy.createClient('my-custom-client', customConfig);
// do something with the client, such as
await customClient.callAsInternalUser('ping');
// custom client are closable
customClient.close();
```

after:

```ts
const customClient = coreStart.elasticsearch.createClient('my-custom-client', customConfig);
// do something with the client, such as
await customClient.asInternalUser.ping();
// custom client are closable
customClient.close();
```

If, for any reasons, one still needs to reach an endpoint not listed on the client API, using `request.transport` 
is still possible:

```ts
const { body } = await client.asCurrentUser.transport.request({
  method: 'get',
  path: '/my-custom-endpoint',
  body: { my: 'payload'},
  querystring: { param: 'foo' }
})
```

Remark: the new client creation API is now only available from the `start` contract of the elasticsearch service.