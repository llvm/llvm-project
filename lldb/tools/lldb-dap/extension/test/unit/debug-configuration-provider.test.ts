import * as assert from "assert";
import Module = require("module");
import { EventEmitter } from "events";

import {
  getEnvironmentKey,
  getEnvironmentValue,
  supportsCliFlag,
} from "../../src/compatibility-utils";

interface ProviderLoadOptions {
  platform: "linux" | "win32";
  serverModeEnabled: boolean;
  helpText: string;
  forceSupportedFlags?: string[];
  pythonProbeExitCode?: number;
  pythonProbeStdout?: string;
  pythonProbeStderr?: string;
  pythonProbeEmitError?: string;
}

interface ProviderLoadResult {
  provider: {
    resolveDebugConfigurationWithSubstitutedVariables(
      folder: unknown,
      debugConfiguration: Record<string, unknown>,
      token?: unknown,
    ): Promise<Record<string, unknown> | null | undefined>;
  };
  calls: {
    helpProbeCount: number;
    serverStartCount: number;
    spawnCount: number;
    warningCount: number;
    warningMessages: string[];
  };
  restore(): void;
}

function loadProviderForTest(options: ProviderLoadOptions): ProviderLoadResult {
  const calls = {
    helpProbeCount: 0,
    serverStartCount: 0,
    spawnCount: 0,
    warningCount: 0,
    warningMessages: [] as string[],
  };

  const moduleCtor = Module as unknown as {
    _load: (
      request: string,
      parent: NodeModule | null,
      isMain: boolean,
    ) => unknown;
  };
  const originalLoad = moduleCtor._load;

  moduleCtor._load = function (request, parent, isMain) {
    if (request === "vscode") {
      return {
        commands: {
          registerCommand: () => ({ dispose: () => undefined }),
        },
        workspace: {
          getConfiguration: () => ({
            get: <T>(key: string, defaultValue: T): T => {
              if (key === "serverMode") {
                return options.serverModeEnabled as T;
              }
              return defaultValue;
            },
          }),
        },
        window: {
          showErrorMessage: async () => undefined,
          showWarningMessage: async (message: string) => {
            calls.warningCount += 1;
            calls.warningMessages.push(message);
            return undefined;
          },
        },
      };
    }

    if (
      request === "./ui/show-error-message" ||
      request === "./show-error-message"
    ) {
      class ConfigureButton {
        async callback() {
          return undefined;
        }
      }

      return {
        ConfigureButton,
        showErrorMessage: async () => undefined,
      };
    }

    if (request === "child_process") {
      const util = originalLoad.call(this, "util", parent, isMain) as {
        promisify: { custom: symbol };
      };

      const execFile = (
        _file: string,
        args: readonly string[] | undefined,
        _opts: unknown,
        cb:
          | ((error: Error | null, stdout: string, stderr: string) => void)
          | undefined,
      ) => {
        const callback =
          typeof _opts === "function"
            ? (_opts as (error: Error | null, stdout: string, stderr: string) => void)
            : cb ?? (() => undefined);
        if (args?.includes("--help")) {
          calls.helpProbeCount += 1;
        }
        callback(null, options.helpText, "");
        return {};
      };
      (execFile as unknown as Record<symbol, unknown>)[util.promisify.custom] =
        (_file: string, args: readonly string[] | undefined) => {
          if (args?.includes("--help")) {
            calls.helpProbeCount += 1;
          }
          return Promise.resolve({ stdout: options.helpText, stderr: "" });
        };

      return {
        execFile,
        spawn: () => {
          calls.spawnCount += 1;
          const child = new EventEmitter() as EventEmitter & {
            stdout: EventEmitter;
            stderr: EventEmitter;
            kill(): void;
          };
          child.stdout = new EventEmitter();
          child.stderr = new EventEmitter();
          child.kill = () => undefined;

          const stdout = options.pythonProbeStdout ?? "";
          const stderr = options.pythonProbeStderr ?? "";
          const exitCode = options.pythonProbeExitCode ?? 0;

          process.nextTick(() => {
            if (options.pythonProbeEmitError) {
              child.emit("error", new Error(options.pythonProbeEmitError));
              return;
            }
            if (stdout.length > 0) {
              child.stdout.emit("data", stdout);
            }
            if (stderr.length > 0) {
              child.stderr.emit("data", stderr);
            }
            child.emit("close", exitCode);
          });

          return child;
        },
      };
    }

    if (request === "os") {
      return {
        platform: () => options.platform,
      };
    }

    if (request === "./compatibility-utils") {
      const module = originalLoad.call(this, request, parent, isMain) as {
        supportsCliFlag: (helpText: string, flag: string) => boolean;
      };

      return {
        ...module,
        supportsCliFlag: (helpText: string, flag: string) => {
          if (options.forceSupportedFlags?.includes(flag)) {
            return true;
          }
          return module.supportsCliFlag(helpText, flag);
        },
      };
    }

    if (request === "./debug-adapter-factory") {
      return {
        createDebugAdapterExecutable: async () => ({
          command: "lldb-dap",
          args: [],
          options: {},
        }),
      };
    }

    return originalLoad.call(this, request, parent, isMain);
  };

  let providerPath: string | undefined;
  let errorWithNotificationPath: string | undefined;
  let showErrorMessagePath: string | undefined;
  const restore = () => {
    if (providerPath) {
      delete require.cache[providerPath];
    }
    if (errorWithNotificationPath) {
      delete require.cache[errorWithNotificationPath];
    }
    if (showErrorMessagePath) {
      delete require.cache[showErrorMessagePath];
    }
    moduleCtor._load = originalLoad;
  };

  try {
    providerPath = require.resolve("../../src/debug-configuration-provider");
    errorWithNotificationPath = require.resolve(
      "../../src/ui/error-with-notification",
    );
    showErrorMessagePath = require.resolve("../../src/ui/show-error-message");
    delete require.cache[providerPath];
    delete require.cache[errorWithNotificationPath];
    delete require.cache[showErrorMessagePath];
    const providerModule = require("../../src/debug-configuration-provider") as {
      LLDBDapConfigurationProvider: new (
        server: { start: (...args: unknown[]) => Promise<unknown> },
        logger: {
          info: (...args: unknown[]) => void;
          debug: (...args: unknown[]) => void;
          warn: (...args: unknown[]) => void;
          error: (...args: unknown[]) => void;
        },
        logFilePath: unknown,
      ) => ProviderLoadResult["provider"];
    };

    const server = {
      start: async () => {
        calls.serverStartCount += 1;
        return { host: "127.0.0.1", port: 12345 };
      },
    };
    const logger = {
      info: () => undefined,
      debug: () => undefined,
      warn: () => undefined,
      error: () => undefined,
    };

    const provider = new providerModule.LLDBDapConfigurationProvider(
      server,
      logger,
      {},
    );

    return {
      provider,
      calls,
      restore,
    };
  } catch (error) {
    restore();
    throw error;
  }
}

suite("debug-configuration-provider helpers", function () {
  test("supportsCliFlag matches exact standalone flag", function () {
    const help = `
Usage: lldb-dap [options]
  --connection <connection>
  --connection-timeout <timeout>
`;

    assert.strictEqual(supportsCliFlag(help, "--connection"), true);
  });

  test("supportsCliFlag does not match similarly prefixed flag", function () {
    const help = `
Usage: lldb-dap [options]
  --connection-timeout <timeout>
`;

    assert.strictEqual(supportsCliFlag(help, "--connection"), false);
  });

  test("supportsCliFlag matches check-python exactly", function () {
    const help = `
Usage: lldb-dap [options]
  --check-python
`;

    assert.strictEqual(supportsCliFlag(help, "--check-python"), true);
    assert.strictEqual(supportsCliFlag(help, "--check-python-extra"), false);
  });

  test("getEnvironmentKey matches keys case-insensitively", function () {
    const env = { Path: "C:\\Windows", FOO: "bar" };

    assert.strictEqual(getEnvironmentKey(env, "PATH"), "Path");
    assert.strictEqual(getEnvironmentKey(env, "foo"), "FOO");
    assert.strictEqual(getEnvironmentKey(env, "MISSING"), undefined);
  });

  test("getEnvironmentValue returns values case-insensitively", function () {
    const env = { Path: "C:\\Windows" };

    assert.strictEqual(getEnvironmentValue(env, "PATH"), "C:\\Windows");
    assert.strictEqual(getEnvironmentValue(env, "MISSING"), undefined);
    assert.strictEqual(getEnvironmentValue(undefined, "PATH"), undefined);
  });
});

suite("debug-configuration-provider integration behavior", function () {
  test("does not probe server capabilities when server mode is disabled", async function () {
    const harness = loadProviderForTest({
      platform: "linux",
      serverModeEnabled: false,
      helpText: "Usage: lldb-dap [options]\n  --connection <connection>\n",
    });

    try {
      const config: Record<string, unknown> = {
        name: "test",
        request: "launch",
        type: "lldb-dap",
        console: "integratedConsole",
      };

      const resolved =
        await harness.provider.resolveDebugConfigurationWithSubstitutedVariables(
          undefined,
          config,
        );

      assert.strictEqual(harness.calls.helpProbeCount, 0);
      assert.strictEqual(harness.calls.serverStartCount, 0);
      assert.strictEqual(resolved, config);
    } finally {
      harness.restore();
    }
  });

  test("starts server mode when check-python is unsupported, without a visible warning", async function () {
    const harness = loadProviderForTest({
      platform: "win32",
      serverModeEnabled: true,
      helpText:
        "Usage: lldb-dap [options]\n  --connection <connection>\n  --connection-timeout <timeout>\n",
    });

    try {
      const config: Record<string, unknown> = {
        name: "test",
        request: "launch",
        type: "lldb-dap",
        console: "integratedConsole",
      };

      const resolved =
        await harness.provider.resolveDebugConfigurationWithSubstitutedVariables(
          undefined,
          config,
        );

      assert.strictEqual(config.console, "integratedConsole");
      assert.strictEqual((resolved as Record<string, unknown>).console, "integratedConsole");
      // Server mode only depends on --connection support, which this
      // adapter has, so it must still start even without --check-python.
      assert.strictEqual(harness.calls.serverStartCount, 1);
      assert.strictEqual(
        (resolved as Record<string, unknown>).debugAdapterHostname,
        "127.0.0.1",
      );
      assert.strictEqual(
        (resolved as Record<string, unknown>).debugAdapterPort,
        12345,
      );
      assert.ok(harness.calls.helpProbeCount >= 1);
      // Missing --check-python support is common on older, working
      // adapters, so it must not raise a visible warning notification.
      assert.strictEqual(harness.calls.warningCount, 0);
    } finally {
      harness.restore();
    }
  });

  test("shows the skipped-check warning only once per adapter when the probe errors", async function () {
    const harness = loadProviderForTest({
      platform: "win32",
      serverModeEnabled: false,
      helpText: "Usage: lldb-dap [options]\n  --check-python\n",
      pythonProbeEmitError: "spawn EACCES",
    });

    try {
      const config: Record<string, unknown> = {
        name: "test",
        request: "launch",
        type: "lldb-dap",
      };

      await harness.provider.resolveDebugConfigurationWithSubstitutedVariables(
        undefined,
        { ...config },
      );
      await harness.provider.resolveDebugConfigurationWithSubstitutedVariables(
        undefined,
        { ...config },
      );

      assert.strictEqual(harness.calls.warningCount, 1);
      assert.match(
        harness.calls.warningMessages[0],
        /Skipped the Python runtime check[\s\S]*--check-python/,
      );
    } finally {
      harness.restore();
    }
  });

  test("blocks launch when supported check-python probe fails", async function () {
    const harness = loadProviderForTest({
      platform: "win32",
      serverModeEnabled: true,
      helpText:
        "Usage: lldb-dap [options]\n  --check-python\n  --connection <connection>\n",
      pythonProbeExitCode: 1,
      pythonProbeStderr: "missing python runtime",
    });

    try {
      const config: Record<string, unknown> = {
        name: "test",
        request: "launch",
        type: "lldb-dap",
        console: "integratedConsole",
      };

      const resolved =
        await harness.provider.resolveDebugConfigurationWithSubstitutedVariables(
          undefined,
          config,
        );

      assert.strictEqual(resolved, undefined);
      assert.strictEqual(harness.calls.serverStartCount, 0);
      assert.strictEqual(harness.calls.spawnCount, 1);
      // A genuine --check-python failure blocks with a modal error rather
      // than the non-blocking skipped-check warning.
      assert.strictEqual(harness.calls.warningCount, 0);
    } finally {
      harness.restore();
    }
  });
});
