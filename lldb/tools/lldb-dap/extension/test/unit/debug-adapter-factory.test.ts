import * as assert from "assert";
import Module = require("module");

function loadResolveWindowsPythonRuntimeLibrary(): typeof import("../../src/debug-adapter-factory").resolveWindowsPythonRuntimeLibrary {
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
      return {};
    }
    return originalLoad.call(this, request, parent, isMain);
  };

  const modulePath = require.resolve("../../src/debug-adapter-factory");
  delete require.cache[modulePath];

  try {
    const debugAdapterFactoryModule = require("../../src/debug-adapter-factory") as typeof import("../../src/debug-adapter-factory");
    return debugAdapterFactoryModule.resolveWindowsPythonRuntimeLibrary;
  } finally {
    delete require.cache[modulePath];
    moduleCtor._load = originalLoad;
  }
}

const resolveWindowsPythonRuntimeLibrary =
  loadResolveWindowsPythonRuntimeLibrary();

suite("debug-adapter-factory Windows Python runtime selection", function () {
  test("prefers explicit LLDB_PYTHON_LIBRARY over discovered runtimes", function () {
    const explicitRuntime = "C:\\python\\python310.dll";

    const resolved = resolveWindowsPythonRuntimeLibrary(
      { LLDB_PYTHON_LIBRARY: explicitRuntime },
      "win32",
    );

    assert.strictEqual(resolved, explicitRuntime);
  });

  test("accepts case-insensitive LLDB_PYTHON_LIBRARY key names", function () {
    const explicitRuntime = "C:\\python\\python311.dll";

    const resolved = resolveWindowsPythonRuntimeLibrary(
      { lldb_python_library: explicitRuntime },
      "win32",
    );

    assert.strictEqual(resolved, explicitRuntime);
  });

  test("does not infer LLDB_PYTHON_LIBRARY from PYTHONHOME or PATH", function () {
    const resolved = resolveWindowsPythonRuntimeLibrary(
      {
        PYTHONHOME: "C:\\Python313",
        PATH: "C:\\Python313;C:\\Python313\\DLLs",
      },
      "win32",
    );

    assert.strictEqual(resolved, undefined);
  });
});
