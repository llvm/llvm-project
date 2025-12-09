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
  using Py_FinalizeEx_t = int(void);
  using PyDict_GetItemString_t = void *(void *, const char *);
  using PyGILStateEnsure_t = int();
  using PyGILStateRelease_t = void(int);
  using PyImport_AddModule_t = void *(const char *);
  using PyLong_FromVoidPtr_t = void *(void *);
  using PyUnicode_FromString_t = void *(const char *);
  using PyModule_GetDict_t = void *(void *);
  using PyObject_CallObject_t = void *(void *, void *);
  using PyObject_IsTrue_t = int(void *);
  using PyRun_SimpleString_t = int(const char *);
  using PyTuple_SetItem_t = int(void *, long, void *);
  using PyTuple_New_t = void *(long);
  using PyTypeObject_t = void *;

  // pylifecycle.h
  Py_InitializeEx_t *Py_InitializeEx;
  Py_FinalizeEx_t *Py_FinalizeEx;

  // pythonrun.h
  PyRun_SimpleString_t *PyRun_SimpleString;

  // pystate.h
  PyGILStateEnsure_t *PyGILState_Ensure;
  PyGILStateRelease_t *PyGILState_Release;

  // import.h
  PyImport_AddModule_t *PyImport_AddModule;

  // object.h
  PyObject_IsTrue_t *PyObject_IsTrue;

  // moduleobject.h
  PyModule_GetDict_t *PyModule_GetDict;

  // dictobject.h
  PyDict_GetItemString_t *PyDict_GetItemString;

  // abstract.h
  PyObject_CallObject_t *PyObject_CallObject;

  // longobject.h
  PyLong_FromVoidPtr_t *PyLong_FromVoidPtr;

  // unicodeobject.h
  PyUnicode_FromString_t *PyUnicode_FromString;

  // tupleobject.h
  PyTuple_SetItem_t *PyTuple_SetItem;
  PyTuple_New_t *PyTuple_New;

private:
  PythonAPI() : Valid(false) {
    if (!loadDylib(findPython()))
      return;
    if (!resolveSymbols())
      return;
    Py_InitializeEx(0);
    Valid = true;
  }

  ~PythonAPI() {
    if (std::atomic_exchange(&Valid, false)) {
      Py_FinalizeEx();
    }
  }

public:
  // Python interface is initialized on first access and it is shared across all
  // threads. It can be used like a state-less thread-safe object.
  static const PythonAPI &instance() {
    static const PythonAPI PyAPI;
    return PyAPI;
  }

  bool loadDylib(std::string Path) {
    std::string Err;
    Dylib = sys::DynamicLibrary::getPermanentLibrary(Path.c_str(), &Err);
    if (!Dylib.isValid()) {
      errs() << "dlopen for '" << Path << "' failed: " << Err << "\n";
      return false;
    }

    return true;
  }

  bool resolveSymbols() {
    bool Success = true;
    Success &= resolve("Py_InitializeEx", &Py_InitializeEx);
    Success &= resolve("Py_FinalizeEx", &Py_FinalizeEx);
    Success &= resolve("PyGILState_Ensure", &PyGILState_Ensure);
    Success &= resolve("PyGILState_Release", &PyGILState_Release);
    Success &= resolve("PyRun_SimpleString", &PyRun_SimpleString);
    Success &= resolve("PyImport_AddModule", &PyImport_AddModule);
    Success &= resolve("PyModule_GetDict", &PyModule_GetDict);
    Success &= resolve("PyDict_GetItemString", &PyDict_GetItemString);
    Success &= resolve("PyObject_CallObject", &PyObject_CallObject);
    Success &= resolve("PyObject_IsTrue", &PyObject_IsTrue);
    Success &= resolve("PyLong_FromVoidPtr", &PyLong_FromVoidPtr);
    Success &= resolve("PyUnicode_FromString", &PyUnicode_FromString);
    Success &= resolve("PyTuple_SetItem", &PyTuple_SetItem);
    Success &= resolve("PyTuple_New", &PyTuple_New);
    return Success;
  }

  bool isValid() const { return Valid; }

  bool loadScript(const std::string &ScriptPath) const {
    std::string LoadCmd;
    raw_string_ostream(LoadCmd)
        << "import runpy\n"
        << "globals().update(runpy.run_path('" << ScriptPath << "'))";

    if (PyRun_SimpleString(LoadCmd.c_str()) != 0) {
      errs() << "Failed to load script: " << ScriptPath << "\n";
      return false;
    }

    return true;
  }

  bool addImportSearchPath(std::string Path) const {
    std::string LoadCmd;
    raw_string_ostream(LoadCmd) << "import sys\n"
                                << "sys.path.append('" << Path << "')";
    // Interpreter is not thread-safe
    auto GIL = make_scope_exit(
        [this, Lock = PyGILState_Ensure()]() { PyGILState_Release(Lock); });
    if (PyRun_SimpleString(LoadCmd.c_str()) != 0) {
      errs() << "Failed to add import search path: " << Path << "\n";
      return false;
    }

    return true;
  }

  void *addModule(const char *Name) const {
    void *Mod = PyImport_AddModule(Name);
    return PyModule_GetDict(Mod);
  }

  // Very simple interface to execute a Python function
  bool invoke(void *Mod, const char *Name, void *Args = nullptr) const {
    // If the function doesn't exist, we assume no
    void *Fn = PyDict_GetItemString(Mod, Name);
    if (!Fn)
      return false;
    // Interpreter is not thread-safe
    auto GIL = make_scope_exit(
        [this, Lock = PyGILState_Ensure()]() { PyGILState_Release(Lock); });
    // If we get no result, there was an error in Python
    void *Result = PyObject_CallObject(Fn, Args);
    if (!Result) {
      errs() << "PyPassContext error: " << Name << "() failed\n";
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

  bool loadScript(const std::string &Path) {
    // Make relative paths resolve naturally in import statements
    std::string Dir = std::filesystem::path(Path).parent_path().u8string();
    if (!PyAPI.addImportSearchPath(Dir))
      return false;

    if (!PyAPI.loadScript(Path))
      return false;

    PyGlobals = PyAPI.addModule("__main__");
    PyBuiltins = PyAPI.addModule("builtins");
    if (!PyGlobals || !PyBuiltins)
      return false;

    return PyAPI.PyDict_GetItemString(PyGlobals, "run");
  }

  bool registerEP(std::string Name) {
    std::string EP = "register" + Name;
    return PyAPI.invoke(PyGlobals, EP.c_str());
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

    return PyAPI.invoke(PyGlobals, "run", Args);
  }

private:
  const PythonAPI &PyAPI;
  void *PyGlobals;
  void *PyBuiltins;
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
  if (!PyAPI.isValid())
    return;

  // Context is shared across all entry-points in this pipeline
  auto Context = std::make_shared<PyPassContext>(PyAPI);

  // All entry-point callbacks are forwarded to the same script
  // TODO: In order to support multiple instances of the plugin with different
  // scripts, we will need an isolated interpreter session for each.
  std::string ScriptPath = findScript();
  if (!Context->loadScript(ScriptPath)) {
    errs() << "Failed to load script for PyPass plugin: '" << ScriptPath
           << "\n";
    return;
  }

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
