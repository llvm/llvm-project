//===- tools/plugins-shlib/pypass.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm-c/Types.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

using namespace llvm;

static cl::opt<std::string>
    DylibPath("pypass-dylib", cl::desc("Path to the Python shared library"),
              cl::init(""));

static cl::opt<std::string>
    ScriptPath("pypass-script", cl::desc("Path to the Python script to run"),
               cl::init(""));

static std::string findPython() {
  if (!DylibPath.empty())
    return DylibPath;
  if (const char *Path = std::getenv("LLVM_PYPASS_DYLIB"))
    return std::string(Path);
  // TODO: Run Python from PATH and use a script to query the shared lib
  errs() << "Failed to detect Python dynamic library\n";
  return std::string{};
}

static std::string findScript() {
  if (!ScriptPath.empty())
    return ScriptPath;
  if (const char *Path = std::getenv("LLVM_PYPASS_SCRIPT"))
    return std::string(Path);
  return std::string{};
}

struct PythonAPI {
  using Py_InitializeEx_t = void(int);
  using Py_FinalizeEx_t = int();
  using Py_DecRef_t = void(void *);
  using Py_IncRef_t = void(void *);
  using PyDict_GetItemString_t = void *(void *, const char *);
  using PyDict_New_t = void *();
  using PyDict_SetItemString_t = int(void *, const char *, void *);
  using PyErr_Print_t = void();
  using PyGILStateEnsure_t = int();
  using PyGILStateRelease_t = void(int);
  using PyImport_AddModule_t = void *(const char *);
  using PyImport_ImportModule_t = void *(const char *);
  using PyLong_FromVoidPtr_t = void *(void *);
  using PyUnicode_FromString_t = void *(const char *);
  using PyModule_GetDict_t = void *(void *);
  using PyObject_CallObject_t = void *(void *, void *);
  using PyObject_GetAttrString_t = void *(void *, const char *);
  using PyObject_IsTrue_t = int(void *);
  using PyTuple_SetItem_t = int(void *, long, void *);
  using PyTuple_New_t = void *(long);
  using PyTypeObject_t = void *;

  // pylifecycle.h
  Py_InitializeEx_t *Py_InitializeEx;
  Py_FinalizeEx_t *Py_FinalizeEx;

  // pystate.h
  PyGILStateEnsure_t *PyGILState_Ensure;
  PyGILStateRelease_t *PyGILState_Release;

  // pythonrun.h
  PyErr_Print_t *PyErr_Print;

  // import.h
  PyImport_AddModule_t *PyImport_AddModule;
  PyImport_ImportModule_t *PyImport_ImportModule;

  // object.h
  PyObject_IsTrue_t *PyObject_IsTrue;
  PyObject_GetAttrString_t *PyObject_GetAttrString;
  Py_IncRef_t *Py_IncRef;
  Py_DecRef_t *Py_DecRef;

  // moduleobject.h
  PyModule_GetDict_t *PyModule_GetDict;

  // dictobject.h
  PyDict_GetItemString_t *PyDict_GetItemString;
  PyDict_SetItemString_t *PyDict_SetItemString;
  PyDict_New_t *PyDict_New;

  // abstract.h
  PyObject_CallObject_t *PyObject_CallObject;

  // longobject.h
  PyLong_FromVoidPtr_t *PyLong_FromVoidPtr;

  // unicodeobject.h
  PyUnicode_FromString_t *PyUnicode_FromString;

  // tupleobject.h
  PyTuple_SetItem_t *PyTuple_SetItem;
  PyTuple_New_t *PyTuple_New;

  void *PyGlobals;
  void *PyBuiltins;

private:
  PythonAPI() : Valid(false) {
    if (!loadDylib(findPython()))
      return;
    if (!resolveSymbols())
      return;
    Py_InitializeEx(0);
    PyBuiltins = PyImport_ImportModule("builtins");
    PyGlobals = PyModule_GetDict(PyImport_AddModule("__main__"));
    Valid = true;
  }

  ~PythonAPI() {
    if (std::atomic_exchange(&Valid, false)) {
      Py_DecRef(PyBuiltins);
      Py_DecRef(PyGlobals);
      Py_FinalizeEx();
    }
  }

  bool loadDylib(std::string Path) {
    // Empty paths dlopen host process
    if (Path.empty()) {
      errs() << "Failed to load Python shared library: ''\n";
      return false;
    }
    std::string Err;
    Dylib = sys::DynamicLibrary::getPermanentLibrary(Path.c_str(), &Err);
    if (!Dylib.isValid()) {
      errs() << "Failed to load Python shared library: '" << Path << "' ("
             << Err << ")\n";
      return false;
    }

    return true;
  }

  bool resolveSymbols() {
    bool Success = true;
    Success &= resolve("_Py_IncRef", &Py_IncRef);
    Success &= resolve("_Py_DecRef", &Py_DecRef);
    Success &= resolve("Py_InitializeEx", &Py_InitializeEx);
    Success &= resolve("Py_FinalizeEx", &Py_FinalizeEx);
    Success &= resolve("PyErr_Print", &PyErr_Print);
    Success &= resolve("PyGILState_Ensure", &PyGILState_Ensure);
    Success &= resolve("PyGILState_Release", &PyGILState_Release);
    Success &= resolve("PyImport_AddModule", &PyImport_AddModule);
    Success &= resolve("PyImport_ImportModule", &PyImport_ImportModule);
    Success &= resolve("PyModule_GetDict", &PyModule_GetDict);
    Success &= resolve("PyDict_GetItemString", &PyDict_GetItemString);
    Success &= resolve("PyDict_SetItemString", &PyDict_SetItemString);
    Success &= resolve("PyDict_New", &PyDict_New);
    Success &= resolve("PyObject_CallObject", &PyObject_CallObject);
    Success &= resolve("PyObject_GetAttrString", &PyObject_GetAttrString);
    Success &= resolve("PyObject_IsTrue", &PyObject_IsTrue);
    Success &= resolve("PyLong_FromVoidPtr", &PyLong_FromVoidPtr);
    Success &= resolve("PyUnicode_FromString", &PyUnicode_FromString);
    Success &= resolve("PyTuple_SetItem", &PyTuple_SetItem);
    Success &= resolve("PyTuple_New", &PyTuple_New);
    return Success;
  }

  bool importModule(const char *Name) const {
    void *Mod = PyImport_ImportModule(Name);
    if (!Mod) {
      PyErr_Print();
      return false;
    }
    PyDict_SetItemString(PyGlobals, Name, Mod);
    Py_DecRef(Mod);
    return true;
  }

  bool evaluate(std::string Code, void *Globals) const {
    void *Exec = PyObject_GetAttrString(PyBuiltins, "exec");
    void *Args = PyTuple_New(2);
    if (!Args)
      return false;
    if (PyTuple_SetItem(Args, 0, PyUnicode_FromString(Code.c_str())))
      return false;
    Py_IncRef(Globals);
    if (PyTuple_SetItem(Args, 1, Globals))
      return false;

    // Interpreter is not thread-safe
    auto GIL = make_scope_exit(
        [this, Lock = PyGILState_Ensure()]() { PyGILState_Release(Lock); });
    void *Result = PyObject_CallObject(Exec, Args);
    Py_DecRef(Args);

    if (Result == nullptr) {
      PyErr_Print();
      return false;
    }

    return true;
  }

public:
  static const PythonAPI &instance() {
    static const PythonAPI PyAPI;
    return PyAPI;
  }

  bool isValid() const { return Valid; }

  bool loadScript(const std::string &Path) const {
    std::filesystem::path Script(Path);
    if (!std::filesystem::exists(Script)) {
      errs() << "Failed to locate script file: '" << Script << "'\n";
      return false;
    }

    // Make relative paths resolve naturally in import statements
    if (!importModule("sys")) {
      // Python import error was dumped already
      return false;
    }
    std::string Dir = Script.parent_path().u8string();
    if (!evaluate("sys.path.append('" + Dir + "')", PyGlobals)) {
      errs() << "Failed to add import search path: '" << Dir << "'\n";
      return false;
    }

    // Evaluating the script runs top-level statements in the global context.
    // TODO: In order to support multiple instances of the plugin with different
    // scripts, we will need an isolated context for each.
    if (!importModule("runpy")) {
      // Python import error was dumped already
      return false;
    }
    std::string LoadCmd = "globals().update(runpy.run_path('" + Path + "'))";
    if (!evaluate(LoadCmd, PyGlobals)) {
      errs() << "Failed to load script for PyPass plugin: '" << Path << "'\n";
      return false;
    }

    // Validate script file
    if (!PyDict_GetItemString(PyGlobals, "run")) {
      errs() << "Script defines no run() function: " << Path << "\n";
      return false;
    }

    return true;
  }

  void *getFunction(std::string Name) const {
    return PyDict_GetItemString(PyGlobals, Name.c_str());
  }

  // Run Python function with boolean result
  bool invoke(void *Fn, void *Args = nullptr) const {
    // Interpreter is not thread-safe
    auto GIL = make_scope_exit(
        [this, Lock = PyGILState_Ensure()]() { PyGILState_Release(Lock); });
    // If we get no result, there was an error in Python
    void *Result = PyObject_CallObject(Fn, Args);
    if (Args)
      Py_DecRef(Args);
    if (!Result) {
      errs() << "PyPassContext error: invoke failed\n";
      return false;
    }
    // If the result is truthy, then it's a yes
    return PyObject_IsTrue(Result);
  }

private:
  sys::DynamicLibrary Dylib;
  std::atomic<bool> Valid;

  template <typename FnTy> bool resolve(const char *Name, FnTy **Var) {
    assert(Dylib.isValid() && "dlopen shared library first");
    assert(*Var == nullptr && "Resolve symbols once");
    if (void *FnPtr = Dylib.getAddressOfSymbol(Name)) {
      *Var = reinterpret_cast<FnTy *>(FnPtr);
      return true;
    }
    errs() << "Missing required CPython API symbol '" << Name
           << "' in: " << DylibPath << "\n";
    return false;
  };
};

class PyPassContext {
public:
  PyPassContext(const PythonAPI &PyAPI) : PyAPI(PyAPI) {}

  bool registerEP(std::string Name) {
    // Default is no, if the function is not defined
    if (void *Fn = PyAPI.getFunction("register" + Name))
      return PyAPI.invoke(Fn);
    return false;
  }

  bool run(void *Entity, void *Ctx, const char *Stage) {
    void *Args = PyAPI.PyTuple_New(3);
    if (!Args)
      return false;
    if (PyAPI.PyTuple_SetItem(Args, 0, PyAPI.PyLong_FromVoidPtr(Entity)) != 0)
      return false;
    if (PyAPI.PyTuple_SetItem(Args, 1, PyAPI.PyLong_FromVoidPtr(Ctx)) != 0)
      return false;
    if (PyAPI.PyTuple_SetItem(Args, 2, PyAPI.PyUnicode_FromString(Stage)) != 0)
      return false;

    // TODO: Should we expose PyPassContext and/or arguments from specific
    // entry-points like OptLevel, ThinOrFullLTOPhase or InnerPipeline?

    void *RunFn = PyAPI.getFunction("run");
    assert(RunFn && "Checked on load");
    return PyAPI.invoke(RunFn, Args);
  }

private:
  const PythonAPI &PyAPI;
};

struct PyPass : PassInfoMixin<PyPass> {
  PyPass(std::shared_ptr<PyPassContext> Context, StringRef EP)
      : Stage(EP.str()), Context(std::move(Context)) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    LLVMModuleRef Mod = wrap(&M);
    LLVMContextRef Ctx = wrap(&M.getContext());
    bool Changed = Context->run(Mod, Ctx, Stage.c_str());
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    LLVMValueRef Fn = wrap(&F);
    LLVMContextRef Ctx = wrap(&F.getContext());
    bool Changed = Context->run(Fn, Ctx, Stage.c_str());
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  std::string Stage;
  std::shared_ptr<PyPassContext> Context;
  std::optional<OptimizationLevel> OptLevel;
  std::optional<ThinOrFullLTOPhase> LTOPhase;
  std::optional<ArrayRef<llvm::PassBuilder::PipelineElement>> InnerPipeline;
};

static void registerCallbacks(PassBuilder &PB) {
  // Python API is initialized once and shared across all threads
  const PythonAPI &PyAPI = PythonAPI::instance();

  // Don't register callbacks if init fails, e.g. shared library not found
  if (!PyAPI.isValid())
    return;

  // Don't register callbacks if script fails to load, e.g. file not found
  std::string ScriptPath = findScript();
  if (!PyAPI.loadScript(ScriptPath))
    return;

  // Context is shared across all entry-points in this pipeline
  auto Context = std::make_shared<PyPassContext>(PythonAPI::instance());

  // Create one PyPass instance per entry-point
  if (Context->registerEP("PipelineStartEPCallback"))
    PB.registerPipelineStartEPCallback(
        [Context](ModulePassManager &MPM, OptimizationLevel Opt) {
          PyPass P(Context, "PipelineStartEPCallback");
          P.OptLevel = Opt;
          MPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("PipelineEarlySimplificationEPCallback"))
    PB.registerPipelineEarlySimplificationEPCallback(
        [Context](ModulePassManager &MPM, OptimizationLevel Opt,
                  ThinOrFullLTOPhase Phase) {
          PyPass P(Context, "PipelineEarlySimplificationEPCallback");
          P.OptLevel = Opt;
          P.LTOPhase = Phase;
          MPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("OptimizerEarlyEPCallback"))
    PB.registerOptimizerEarlyEPCallback([Context](ModulePassManager &MPM,
                                                  OptimizationLevel Opt,
                                                  ThinOrFullLTOPhase Phase) {
      PyPass P(Context, "OptimizerEarlyEPCallback");
      P.OptLevel = Opt;
      P.LTOPhase = Phase;
      MPM.addPass(std::move(P));
      return true;
    });

  if (Context->registerEP("OptimizerLastEPCallback"))
    PB.registerOptimizerLastEPCallback([Context](ModulePassManager &MPM,
                                                 OptimizationLevel Opt,
                                                 ThinOrFullLTOPhase Phase) {
      PyPass P(Context, "OptimizerLastEPCallback");
      P.OptLevel = Opt;
      P.LTOPhase = Phase;
      MPM.addPass(std::move(P));
      return true;
    });

  if (Context->registerEP("PeepholeEPCallback"))
    PB.registerPeepholeEPCallback(
        [Context](FunctionPassManager &FPM, OptimizationLevel Opt) {
          PyPass P(Context, "PeepholeEPCallback");
          P.OptLevel = Opt;
          FPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("ScalarOptimizerLateEPCallback"))
    PB.registerScalarOptimizerLateEPCallback(
        [Context](FunctionPassManager &FPM, OptimizationLevel Opt) {
          PyPass P(Context, "ScalarOptimizerLateEPCallback");
          P.OptLevel = Opt;
          FPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("VectorizerStartEPCallback"))
    PB.registerVectorizerStartEPCallback(
        [Context](FunctionPassManager &FPM, OptimizationLevel Opt) {
          PyPass P(Context, "VectorizerStartEPCallback");
          P.OptLevel = Opt;
          FPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("VectorizerEndEPCallback"))
    PB.registerVectorizerEndEPCallback(
        [Context](FunctionPassManager &FPM, OptimizationLevel Opt) {
          PyPass P(Context, "VectorizerEndEPCallback");
          P.OptLevel = Opt;
          FPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("FullLinkTimeOptimizationEarlyEPCallback"))
    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [Context](ModulePassManager &MPM, OptimizationLevel Opt) {
          PyPass P(Context, "FullLinkTimeOptimizationEarlyEPCallback");
          P.OptLevel = Opt;
          MPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("FullLinkTimeOptimizationLastEPCallback"))
    PB.registerFullLinkTimeOptimizationLastEPCallback(
        [Context](ModulePassManager &MPM, OptimizationLevel Opt) {
          PyPass P(Context, "FullLinkTimeOptimizationLastEPCallback");
          P.OptLevel = Opt;
          MPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("ModulePipelineParsingCallback"))
    PB.registerPipelineParsingCallback(
        [Context](StringRef Name, ModulePassManager &MPM,
                  ArrayRef<llvm::PassBuilder::PipelineElement> InnerPipeline) {
          if (Name.lower() != "pypass")
            return false;
          PyPass P(Context, "Module");
          P.InnerPipeline = InnerPipeline;
          MPM.addPass(std::move(P));
          return true;
        });

  if (Context->registerEP("FunctionPipelineParsingCallback"))
    PB.registerPipelineParsingCallback(
        [Context](StringRef Name, FunctionPassManager &FPM,
                  ArrayRef<llvm::PassBuilder::PipelineElement> InnerPipeline) {
          if (Name.lower() != "pypass")
            return false;
          PyPass P(Context, "Function");
          P.InnerPipeline = InnerPipeline;
          FPM.addPass(std::move(P));
          return true;
        });

  // TODO: This is invoked as fallback if neither module nor function parsing
  // yields a result. Does it make sense for plugins?
  if (Context->registerEP("TopLevelPipelineParsingCallback"))
    PB.registerParseTopLevelPipelineCallback(
        [Context](ModulePassManager &MPM,
                  ArrayRef<llvm::PassBuilder::PipelineElement> InnerPipeline) {
          PyPass P(Context, "Module");
          P.InnerPipeline = InnerPipeline;
          MPM.addPass(std::move(P));
          return true;
        });
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PyPass", LLVM_VERSION_STRING,
          registerCallbacks};
}
