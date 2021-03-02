//===-- ScriptInterpreterPython.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "PythonDataObjects.h"
#include "PythonReadline.h"
#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"
#include "ScriptedProcessPythonInterface.h"

#include "lldb/API/SBError.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Breakpoint/WatchpointOptions.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/ReproducerInstrumentation.h"
#include "lldb/Utility/Timer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"

#include <memory>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using llvm::Expected;

LLDB_PLUGIN_DEFINE(ScriptInterpreterPython)

// Defined in the SWIG source file
#if PY_MAJOR_VERSION >= 3
extern "C" PyObject *PyInit__lldb(void);

#define LLDBSwigPyInit PyInit__lldb

#else
extern "C" void init_lldb(void);

#define LLDBSwigPyInit init_lldb
#endif

// These prototypes are the Pythonic implementations of the required callbacks.
// Although these are scripting-language specific, their definition depends on
// the public API.

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

// Disable warning C4190: 'LLDBSwigPythonBreakpointCallbackFunction' has
// C-linkage specified, but returns UDT 'llvm::Expected<bool>' which is
// incompatible with C
#if _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4190)
#endif

extern "C" llvm::Expected<bool> LLDBSwigPythonBreakpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame,
    const lldb::BreakpointLocationSP &sb_bp_loc, StructuredDataImpl *args_impl);

#if _MSC_VER
#pragma warning (pop)
#endif

#pragma clang diagnostic pop

extern "C" bool LLDBSwigPythonWatchpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame, const lldb::WatchpointSP &sb_wp);

extern "C" bool LLDBSwigPythonCallTypeScript(
    const char *python_function_name, void *session_dictionary,
    const lldb::ValueObjectSP &valobj_sp, void **pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP &options_sp, std::string &retval);

extern "C" void *
LLDBSwigPythonCreateSyntheticProvider(const char *python_class_name,
                                      const char *session_dictionary_name,
                                      const lldb::ValueObjectSP &valobj_sp);

extern "C" void *
LLDBSwigPythonCreateCommandObject(const char *python_class_name,
                                  const char *session_dictionary_name,
                                  const lldb::DebuggerSP debugger_sp);

extern "C" void *LLDBSwigPythonCreateScriptedThreadPlan(
    const char *python_class_name, const char *session_dictionary_name,
    StructuredDataImpl *args_data,
    std::string &error_string,
    const lldb::ThreadPlanSP &thread_plan_sp);

extern "C" bool LLDBSWIGPythonCallThreadPlan(void *implementor,
                                             const char *method_name,
                                             Event *event_sp, bool &got_error);

extern "C" void *LLDBSwigPythonCreateScriptedBreakpointResolver(
    const char *python_class_name, const char *session_dictionary_name,
    lldb_private::StructuredDataImpl *args, lldb::BreakpointSP &bkpt_sp);

extern "C" unsigned int
LLDBSwigPythonCallBreakpointResolver(void *implementor, const char *method_name,
                                     lldb_private::SymbolContext *sym_ctx);

extern "C" void *LLDBSwigPythonCreateScriptedStopHook(
    TargetSP target_sp, const char *python_class_name,
    const char *session_dictionary_name, lldb_private::StructuredDataImpl *args,
    lldb_private::Status &error);

extern "C" bool
LLDBSwigPythonStopHookCallHandleStop(void *implementor,
                                     lldb::ExecutionContextRefSP exc_ctx,
                                     lldb::StreamSP stream);

extern "C" size_t LLDBSwigPython_CalculateNumChildren(void *implementor,
                                                      uint32_t max);

extern "C" void *LLDBSwigPython_GetChildAtIndex(void *implementor,
                                                uint32_t idx);

extern "C" int LLDBSwigPython_GetIndexOfChildWithName(void *implementor,
                                                      const char *child_name);

extern lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue(void *data);

extern "C" bool LLDBSwigPython_UpdateSynthProviderInstance(void *implementor);

extern "C" bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance(void *implementor);

extern "C" void *
LLDBSwigPython_GetValueSynthProviderInstance(void *implementor);

extern "C" bool
LLDBSwigPythonCallCommand(const char *python_function_name,
                          const char *session_dictionary_name,
                          lldb::DebuggerSP &debugger, const char *args,
                          lldb_private::CommandReturnObject &cmd_retobj,
                          lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallCommandObject(void *implementor, lldb::DebuggerSP &debugger,
                                const char *args,
                                lldb_private::CommandReturnObject &cmd_retobj,
                                lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallModuleInit(const char *python_module_name,
                             const char *session_dictionary_name,
                             lldb::DebuggerSP &debugger);

extern "C" void *
LLDBSWIGPythonCreateOSPlugin(const char *python_class_name,
                             const char *session_dictionary_name,
                             const lldb::ProcessSP &process_sp);

extern "C" void *
LLDBSWIGPython_CreateFrameRecognizer(const char *python_class_name,
                                     const char *session_dictionary_name);

extern "C" void *
LLDBSwigPython_GetRecognizedArguments(void *implementor,
                                      const lldb::StackFrameSP &frame_sp);

extern "C" bool LLDBSWIGPythonRunScriptKeywordProcess(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ProcessSP &process, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordThread(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ThreadSP &thread, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordTarget(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::TargetSP &target, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordFrame(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::StackFrameSP &frame, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordValue(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ValueObjectSP &value, std::string &output);

extern "C" void *
LLDBSWIGPython_GetDynamicSetting(void *module, const char *setting,
                                 const lldb::TargetSP &target_sp);

static ScriptInterpreterPythonImpl *GetPythonInterpreter(Debugger &debugger) {
  ScriptInterpreter *script_interpreter =
      debugger.GetScriptInterpreter(true, lldb::eScriptLanguagePython);
  return static_cast<ScriptInterpreterPythonImpl *>(script_interpreter);
}

static bool g_initialized = false;

namespace {

// Initializing Python is not a straightforward process.  We cannot control
// what external code may have done before getting to this point in LLDB,
// including potentially having already initialized Python, so we need to do a
// lot of work to ensure that the existing state of the system is maintained
// across our initialization.  We do this by using an RAII pattern where we
// save off initial state at the beginning, and restore it at the end
struct InitializePythonRAII {
public:
  InitializePythonRAII()
      : m_gil_state(PyGILState_UNLOCKED), m_was_already_initialized(false) {
    InitializePythonHome();

#ifdef LLDB_USE_LIBEDIT_READLINE_COMPAT_MODULE
    // Python's readline is incompatible with libedit being linked into lldb.
    // Provide a patched version local to the embedded interpreter.
    bool ReadlinePatched = false;
    for (auto *p = PyImport_Inittab; p->name != NULL; p++) {
      if (strcmp(p->name, "readline") == 0) {
        p->initfunc = initlldb_readline;
        break;
      }
    }
    if (!ReadlinePatched) {
      PyImport_AppendInittab("readline", initlldb_readline);
      ReadlinePatched = true;
    }
#endif

    // Register _lldb as a built-in module.
    PyImport_AppendInittab("_lldb", LLDBSwigPyInit);

// Python < 3.2 and Python >= 3.2 reversed the ordering requirements for
// calling `Py_Initialize` and `PyEval_InitThreads`.  < 3.2 requires that you
// call `PyEval_InitThreads` first, and >= 3.2 requires that you call it last.
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 2) || (PY_MAJOR_VERSION > 3)
    Py_InitializeEx(0);
    InitializeThreadsPrivate();
#else
    InitializeThreadsPrivate();
    Py_InitializeEx(0);
#endif
  }

  ~InitializePythonRAII() {
    if (m_was_already_initialized) {
      Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));
      LLDB_LOGV(log, "Releasing PyGILState. Returning to state = {0}locked",
                m_gil_state == PyGILState_UNLOCKED ? "un" : "");
      PyGILState_Release(m_gil_state);
    } else {
      // We initialized the threads in this function, just unlock the GIL.
      PyEval_SaveThread();
    }
  }

private:
  void InitializePythonHome() {
#if LLDB_EMBED_PYTHON_HOME
#if PY_MAJOR_VERSION >= 3
    typedef wchar_t* str_type;
#else
    typedef char* str_type;
#endif
    static str_type g_python_home = []() -> str_type {
      const char *lldb_python_home = LLDB_PYTHON_HOME;
      const char *absolute_python_home = nullptr;
      llvm::SmallString<64> path;
      if (llvm::sys::path::is_absolute(lldb_python_home)) {
        absolute_python_home = lldb_python_home;
      } else {
        FileSpec spec = HostInfo::GetShlibDir();
        if (!spec)
          return nullptr;
        spec.GetPath(path);
        llvm::sys::path::append(path, lldb_python_home);
        absolute_python_home = path.c_str();
      }
#if PY_MAJOR_VERSION >= 3
      size_t size = 0;
      return Py_DecodeLocale(absolute_python_home, &size);
#else
      return strdup(absolute_python_home);
#endif
    }();
    if (g_python_home != nullptr) {
      Py_SetPythonHome(g_python_home);
    }
#else
#if defined(__APPLE__) && PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    // For Darwin, the only Python version supported is the one shipped in the
    // OS OS and linked with lldb. Other installation of Python may have higher
    // priorities in the path, overriding PYTHONHOME and causing
    // problems/incompatibilities. In order to avoid confusion, always hardcode
    // the PythonHome to be right, as it's not going to change.
    static char path[] =
        "/System/Library/Frameworks/Python.framework/Versions/2.7";
    Py_SetPythonHome(path);
#endif
#endif
  }

  void InitializeThreadsPrivate() {
// Since Python 3.7 `Py_Initialize` calls `PyEval_InitThreads` inside itself,
// so there is no way to determine whether the embedded interpreter
// was already initialized by some external code. `PyEval_ThreadsInitialized`
// would always return `true` and `PyGILState_Ensure/Release` flow would be
// executed instead of unlocking GIL with `PyEval_SaveThread`. When
// an another thread calls `PyGILState_Ensure` it would get stuck in deadlock.
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 7) || (PY_MAJOR_VERSION > 3)
    // The only case we should go further and acquire the GIL: it is unlocked.
    if (PyGILState_Check())
      return;
#endif

    if (PyEval_ThreadsInitialized()) {
      Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));

      m_was_already_initialized = true;
      m_gil_state = PyGILState_Ensure();
      LLDB_LOGV(log, "Ensured PyGILState. Previous state = {0}locked\n",
                m_gil_state == PyGILState_UNLOCKED ? "un" : "");
      return;
    }

    // InitThreads acquires the GIL if it hasn't been called before.
    PyEval_InitThreads();
  }

  TerminalState m_stdin_tty_state;
  PyGILState_STATE m_gil_state;
  bool m_was_already_initialized;
};
} // namespace

void ScriptInterpreterPython::ComputePythonDirForApple(
    llvm::SmallVectorImpl<char> &path) {
  auto style = llvm::sys::path::Style::posix;

  llvm::StringRef path_ref(path.begin(), path.size());
  auto rbegin = llvm::sys::path::rbegin(path_ref, style);
  auto rend = llvm::sys::path::rend(path_ref);
  auto framework = std::find(rbegin, rend, "LLDB.framework");
  if (framework == rend) {
    ComputePythonDir(path);
    return;
  }
  path.resize(framework - rend);
  llvm::sys::path::append(path, style, "LLDB.framework", "Resources", "Python");
}

void ScriptInterpreterPython::ComputePythonDir(
    llvm::SmallVectorImpl<char> &path) {
  // Build the path by backing out of the lib dir, then building with whatever
  // the real python interpreter uses.  (e.g. lib for most, lib64 on RHEL
  // x86_64, or bin on Windows).
  llvm::sys::path::remove_filename(path);
  llvm::sys::path::append(path, LLDB_PYTHON_RELATIVE_LIBDIR);

#if defined(_WIN32)
  // This will be injected directly through FileSpec.GetDirectory().SetString(),
  // so we need to normalize manually.
  std::replace(path.begin(), path.end(), '\\', '/');
#endif
}

FileSpec ScriptInterpreterPython::GetPythonDir() {
  static FileSpec g_spec = []() {
    FileSpec spec = HostInfo::GetShlibDir();
    if (!spec)
      return FileSpec();
    llvm::SmallString<64> path;
    spec.GetPath(path);

#if defined(__APPLE__)
    ComputePythonDirForApple(path);
#else
    ComputePythonDir(path);
#endif
    spec.GetDirectory().SetString(path);
    return spec;
  }();
  return g_spec;
}

void ScriptInterpreterPython::SharedLibraryDirectoryHelper(
    FileSpec &this_file) {
  // When we're loaded from python, this_file will point to the file inside the
  // python package directory. Replace it with the one in the lib directory.
#ifdef _WIN32
  // On windows, we need to manually back out of the python tree, and go into
  // the bin directory. This is pretty much the inverse of what ComputePythonDir
  // does.
  if (this_file.GetFileNameExtension() == ConstString(".pyd")) {
    this_file.RemoveLastPathComponent(); // _lldb.pyd or _lldb_d.pyd
    this_file.RemoveLastPathComponent(); // lldb
    for (auto it = llvm::sys::path::begin(LLDB_PYTHON_RELATIVE_LIBDIR),
              end = llvm::sys::path::end(LLDB_PYTHON_RELATIVE_LIBDIR);
         it != end; ++it)
      this_file.RemoveLastPathComponent();
    this_file.AppendPathComponent("bin");
    this_file.AppendPathComponent("liblldb.dll");
  }
#else
  // The python file is a symlink, so we can find the real library by resolving
  // it. We can do this unconditionally.
  FileSystem::Instance().ResolveSymbolicLink(this_file, this_file);
#endif
}

lldb_private::ConstString ScriptInterpreterPython::GetPluginNameStatic() {
  static ConstString g_name("script-python");
  return g_name;
}

const char *ScriptInterpreterPython::GetPluginDescriptionStatic() {
  return "Embedded Python interpreter";
}

void ScriptInterpreterPython::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  lldb::eScriptLanguagePython,
                                  ScriptInterpreterPythonImpl::CreateInstance);
  });
}

void ScriptInterpreterPython::Terminate() {}

ScriptInterpreterPythonImpl::Locker::Locker(
    ScriptInterpreterPythonImpl *py_interpreter, uint16_t on_entry,
    uint16_t on_leave, FileSP in, FileSP out, FileSP err)
    : ScriptInterpreterLocker(),
      m_teardown_session((on_leave & TearDownSession) == TearDownSession),
      m_python_interpreter(py_interpreter) {
  repro::Recorder::PrivateThread();
  DoAcquireLock();
  if ((on_entry & InitSession) == InitSession) {
    if (!DoInitSession(on_entry, in, out, err)) {
      // Don't teardown the session if we didn't init it.
      m_teardown_session = false;
    }
  }
}

bool ScriptInterpreterPythonImpl::Locker::DoAcquireLock() {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));
  m_GILState = PyGILState_Ensure();
  LLDB_LOGV(log, "Ensured PyGILState. Previous state = {0}locked",
            m_GILState == PyGILState_UNLOCKED ? "un" : "");

  // we need to save the thread state when we first start the command because
  // we might decide to interrupt it while some action is taking place outside
  // of Python (e.g. printing to screen, waiting for the network, ...) in that
  // case, _PyThreadState_Current will be NULL - and we would be unable to set
  // the asynchronous exception - not a desirable situation
  m_python_interpreter->SetThreadState(PyThreadState_Get());
  m_python_interpreter->IncrementLockCount();
  return true;
}

bool ScriptInterpreterPythonImpl::Locker::DoInitSession(uint16_t on_entry_flags,
                                                        FileSP in, FileSP out,
                                                        FileSP err) {
  if (!m_python_interpreter)
    return false;
  return m_python_interpreter->EnterSession(on_entry_flags, in, out, err);
}

bool ScriptInterpreterPythonImpl::Locker::DoFreeLock() {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));
  LLDB_LOGV(log, "Releasing PyGILState. Returning to state = {0}locked",
            m_GILState == PyGILState_UNLOCKED ? "un" : "");
  PyGILState_Release(m_GILState);
  m_python_interpreter->DecrementLockCount();
  return true;
}

bool ScriptInterpreterPythonImpl::Locker::DoTearDownSession() {
  if (!m_python_interpreter)
    return false;
  m_python_interpreter->LeaveSession();
  return true;
}

ScriptInterpreterPythonImpl::Locker::~Locker() {
  if (m_teardown_session)
    DoTearDownSession();
  DoFreeLock();
}

ScriptInterpreterPythonImpl::ScriptInterpreterPythonImpl(Debugger &debugger)
    : ScriptInterpreterPython(debugger), m_saved_stdin(), m_saved_stdout(),
      m_saved_stderr(), m_main_module(),
      m_session_dict(PyInitialValue::Invalid),
      m_sys_module_dict(PyInitialValue::Invalid), m_run_one_line_function(),
      m_run_one_line_str_global(),
      m_dictionary_name(m_debugger.GetInstanceName().AsCString()),
      m_active_io_handler(eIOHandlerNone), m_session_is_active(false),
      m_pty_secondary_is_open(false), m_valid_session(true), m_lock_count(0),
      m_command_thread_state(nullptr) {
  InitializePrivate();

  m_scripted_process_interface_up =
      std::make_unique<ScriptedProcessPythonInterface>(*this);

  m_dictionary_name.append("_dict");
  StreamString run_string;
  run_string.Printf("%s = dict()", m_dictionary_name.c_str());

  Locker locker(this, Locker::AcquireLock, Locker::FreeAcquiredLock);
  PyRun_SimpleString(run_string.GetData());

  run_string.Clear();
  run_string.Printf(
      "run_one_line (%s, 'import copy, keyword, os, re, sys, uuid, lldb')",
      m_dictionary_name.c_str());
  PyRun_SimpleString(run_string.GetData());

  // Reloading modules requires a different syntax in Python 2 and Python 3.
  // This provides a consistent syntax no matter what version of Python.
  run_string.Clear();
  run_string.Printf("run_one_line (%s, 'from six.moves import reload_module')",
                    m_dictionary_name.c_str());
  PyRun_SimpleString(run_string.GetData());

  // WARNING: temporary code that loads Cocoa formatters - this should be done
  // on a per-platform basis rather than loading the whole set and letting the
  // individual formatter classes exploit APIs to check whether they can/cannot
  // do their task
  run_string.Clear();
  run_string.Printf(
      "run_one_line (%s, 'import lldb.formatters, lldb.formatters.cpp, pydoc')",
      m_dictionary_name.c_str());
  PyRun_SimpleString(run_string.GetData());
  run_string.Clear();

  run_string.Printf("run_one_line (%s, 'import lldb.embedded_interpreter; from "
                    "lldb.embedded_interpreter import run_python_interpreter; "
                    "from lldb.embedded_interpreter import run_one_line')",
                    m_dictionary_name.c_str());
  PyRun_SimpleString(run_string.GetData());
  run_string.Clear();

  run_string.Printf("run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64
                    "; pydoc.pager = pydoc.plainpager')",
                    m_dictionary_name.c_str(), m_debugger.GetID());
  PyRun_SimpleString(run_string.GetData());
}

ScriptInterpreterPythonImpl::~ScriptInterpreterPythonImpl() {
  // the session dictionary may hold objects with complex state which means
  // that they may need to be torn down with some level of smarts and that, in
  // turn, requires a valid thread state force Python to procure itself such a
  // thread state, nuke the session dictionary and then release it for others
  // to use and proceed with the rest of the shutdown
  auto gil_state = PyGILState_Ensure();
  m_session_dict.Reset();
  PyGILState_Release(gil_state);
}

lldb_private::ConstString ScriptInterpreterPythonImpl::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ScriptInterpreterPythonImpl::GetPluginVersion() { return 1; }

void ScriptInterpreterPythonImpl::IOHandlerActivated(IOHandler &io_handler,
                                                     bool interactive) {
  const char *instructions = nullptr;

  switch (m_active_io_handler) {
  case eIOHandlerNone:
    break;
  case eIOHandlerBreakpoint:
    instructions = R"(Enter your Python command(s). Type 'DONE' to end.
def function (frame, bp_loc, internal_dict):
    """frame: the lldb.SBFrame for the location at which you stopped
       bp_loc: an lldb.SBBreakpointLocation for the breakpoint location information
       internal_dict: an LLDB support object not to be used"""
)";
    break;
  case eIOHandlerWatchpoint:
    instructions = "Enter your Python command(s). Type 'DONE' to end.\n";
    break;
  }

  if (instructions) {
    StreamFileSP output_sp(io_handler.GetOutputStreamFileSP());
    if (output_sp && interactive) {
      output_sp->PutCString(instructions);
      output_sp->Flush();
    }
  }
}

void ScriptInterpreterPythonImpl::IOHandlerInputComplete(IOHandler &io_handler,
                                                         std::string &data) {
  io_handler.SetIsDone(true);
  bool batch_mode = m_debugger.GetCommandInterpreter().GetBatchCommandMode();

  switch (m_active_io_handler) {
  case eIOHandlerNone:
    break;
  case eIOHandlerBreakpoint: {
    std::vector<BreakpointOptions *> *bp_options_vec =
        (std::vector<BreakpointOptions *> *)io_handler.GetUserData();
    for (auto bp_options : *bp_options_vec) {
      if (!bp_options)
        continue;

      auto data_up = std::make_unique<CommandDataPython>();
      if (!data_up)
        break;
      data_up->user_source.SplitIntoLines(data);

      StructuredData::ObjectSP empty_args_sp;
      if (GenerateBreakpointCommandCallbackData(data_up->user_source,
                                                data_up->script_source,
                                                false)
              .Success()) {
        auto baton_sp = std::make_shared<BreakpointOptions::CommandBaton>(
            std::move(data_up));
        bp_options->SetCallback(
            ScriptInterpreterPythonImpl::BreakpointCallbackFunction, baton_sp);
      } else if (!batch_mode) {
        StreamFileSP error_sp = io_handler.GetErrorStreamFileSP();
        if (error_sp) {
          error_sp->Printf("Warning: No command attached to breakpoint.\n");
          error_sp->Flush();
        }
      }
    }
    m_active_io_handler = eIOHandlerNone;
  } break;
  case eIOHandlerWatchpoint: {
    WatchpointOptions *wp_options =
        (WatchpointOptions *)io_handler.GetUserData();
    auto data_up = std::make_unique<WatchpointOptions::CommandData>();
    data_up->user_source.SplitIntoLines(data);

    if (GenerateWatchpointCommandCallbackData(data_up->user_source,
                                              data_up->script_source)) {
      auto baton_sp =
          std::make_shared<WatchpointOptions::CommandBaton>(std::move(data_up));
      wp_options->SetCallback(
          ScriptInterpreterPythonImpl::WatchpointCallbackFunction, baton_sp);
    } else if (!batch_mode) {
      StreamFileSP error_sp = io_handler.GetErrorStreamFileSP();
      if (error_sp) {
        error_sp->Printf("Warning: No command attached to breakpoint.\n");
        error_sp->Flush();
      }
    }
    m_active_io_handler = eIOHandlerNone;
  } break;
  }
}

lldb::ScriptInterpreterSP
ScriptInterpreterPythonImpl::CreateInstance(Debugger &debugger) {
  return std::make_shared<ScriptInterpreterPythonImpl>(debugger);
}

void ScriptInterpreterPythonImpl::LeaveSession() {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));
  if (log)
    log->PutCString("ScriptInterpreterPythonImpl::LeaveSession()");

  // Unset the LLDB global variables.
  PyRun_SimpleString("lldb.debugger = None; lldb.target = None; lldb.process "
                     "= None; lldb.thread = None; lldb.frame = None");

  // checking that we have a valid thread state - since we use our own
  // threading and locking in some (rare) cases during cleanup Python may end
  // up believing we have no thread state and PyImport_AddModule will crash if
  // that is the case - since that seems to only happen when destroying the
  // SBDebugger, we can make do without clearing up stdout and stderr

  // rdar://problem/11292882
  // When the current thread state is NULL, PyThreadState_Get() issues a fatal
  // error.
  if (PyThreadState_GetDict()) {
    PythonDictionary &sys_module_dict = GetSysModuleDictionary();
    if (sys_module_dict.IsValid()) {
      if (m_saved_stdin.IsValid()) {
        sys_module_dict.SetItemForKey(PythonString("stdin"), m_saved_stdin);
        m_saved_stdin.Reset();
      }
      if (m_saved_stdout.IsValid()) {
        sys_module_dict.SetItemForKey(PythonString("stdout"), m_saved_stdout);
        m_saved_stdout.Reset();
      }
      if (m_saved_stderr.IsValid()) {
        sys_module_dict.SetItemForKey(PythonString("stderr"), m_saved_stderr);
        m_saved_stderr.Reset();
      }
    }
  }

  m_session_is_active = false;
}

bool ScriptInterpreterPythonImpl::SetStdHandle(FileSP file_sp,
                                               const char *py_name,
                                               PythonObject &save_file,
                                               const char *mode) {
  if (!file_sp || !*file_sp) {
    save_file.Reset();
    return false;
  }
  File &file = *file_sp;

  // Flush the file before giving it to python to avoid interleaved output.
  file.Flush();

  PythonDictionary &sys_module_dict = GetSysModuleDictionary();

  auto new_file = PythonFile::FromFile(file, mode);
  if (!new_file) {
    llvm::consumeError(new_file.takeError());
    return false;
  }

  save_file = sys_module_dict.GetItemForKey(PythonString(py_name));

  sys_module_dict.SetItemForKey(PythonString(py_name), new_file.get());
  return true;
}

bool ScriptInterpreterPythonImpl::EnterSession(uint16_t on_entry_flags,
                                               FileSP in_sp, FileSP out_sp,
                                               FileSP err_sp) {
  // If we have already entered the session, without having officially 'left'
  // it, then there is no need to 'enter' it again.
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));
  if (m_session_is_active) {
    LLDB_LOGF(
        log,
        "ScriptInterpreterPythonImpl::EnterSession(on_entry_flags=0x%" PRIx16
        ") session is already active, returning without doing anything",
        on_entry_flags);
    return false;
  }

  LLDB_LOGF(
      log,
      "ScriptInterpreterPythonImpl::EnterSession(on_entry_flags=0x%" PRIx16 ")",
      on_entry_flags);

  m_session_is_active = true;

  StreamString run_string;

  if (on_entry_flags & Locker::InitGlobals) {
    run_string.Printf("run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64,
                      m_dictionary_name.c_str(), m_debugger.GetID());
    run_string.Printf(
        "; lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%" PRIu64 ")",
        m_debugger.GetID());
    run_string.PutCString("; lldb.target = lldb.debugger.GetSelectedTarget()");
    run_string.PutCString("; lldb.process = lldb.target.GetProcess()");
    run_string.PutCString("; lldb.thread = lldb.process.GetSelectedThread ()");
    run_string.PutCString("; lldb.frame = lldb.thread.GetSelectedFrame ()");
    run_string.PutCString("')");
  } else {
    // If we aren't initing the globals, we should still always set the
    // debugger (since that is always unique.)
    run_string.Printf("run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64,
                      m_dictionary_name.c_str(), m_debugger.GetID());
    run_string.Printf(
        "; lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%" PRIu64 ")",
        m_debugger.GetID());
    run_string.PutCString("')");
  }

  PyRun_SimpleString(run_string.GetData());
  run_string.Clear();

  PythonDictionary &sys_module_dict = GetSysModuleDictionary();
  if (sys_module_dict.IsValid()) {
    lldb::FileSP top_in_sp;
    lldb::StreamFileSP top_out_sp, top_err_sp;
    if (!in_sp || !out_sp || !err_sp || !*in_sp || !*out_sp || !*err_sp)
      m_debugger.AdoptTopIOHandlerFilesIfInvalid(top_in_sp, top_out_sp,
                                                 top_err_sp);

    if (on_entry_flags & Locker::NoSTDIN) {
      m_saved_stdin.Reset();
    } else {
      if (!SetStdHandle(in_sp, "stdin", m_saved_stdin, "r")) {
        if (top_in_sp)
          SetStdHandle(top_in_sp, "stdin", m_saved_stdin, "r");
      }
    }

    if (!SetStdHandle(out_sp, "stdout", m_saved_stdout, "w")) {
      if (top_out_sp)
        SetStdHandle(top_out_sp->GetFileSP(), "stdout", m_saved_stdout, "w");
    }

    if (!SetStdHandle(err_sp, "stderr", m_saved_stderr, "w")) {
      if (top_err_sp)
        SetStdHandle(top_err_sp->GetFileSP(), "stderr", m_saved_stderr, "w");
    }
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  return true;
}

PythonModule &ScriptInterpreterPythonImpl::GetMainModule() {
  if (!m_main_module.IsValid())
    m_main_module = unwrapIgnoringErrors(PythonModule::Import("__main__"));
  return m_main_module;
}

PythonDictionary &ScriptInterpreterPythonImpl::GetSessionDictionary() {
  if (m_session_dict.IsValid())
    return m_session_dict;

  PythonObject &main_module = GetMainModule();
  if (!main_module.IsValid())
    return m_session_dict;

  PythonDictionary main_dict(PyRefType::Borrowed,
                             PyModule_GetDict(main_module.get()));
  if (!main_dict.IsValid())
    return m_session_dict;

  m_session_dict = unwrapIgnoringErrors(
      As<PythonDictionary>(main_dict.GetItem(m_dictionary_name)));
  return m_session_dict;
}

PythonDictionary &ScriptInterpreterPythonImpl::GetSysModuleDictionary() {
  if (m_sys_module_dict.IsValid())
    return m_sys_module_dict;
  PythonModule sys_module = unwrapIgnoringErrors(PythonModule::Import("sys"));
  m_sys_module_dict = sys_module.GetDictionary();
  return m_sys_module_dict;
}

llvm::Expected<unsigned>
ScriptInterpreterPythonImpl::GetMaxPositionalArgumentsForCallable(
    const llvm::StringRef &callable_name) {
  if (callable_name.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "called with empty callable name.");
  }
  Locker py_lock(this, Locker::AcquireLock |
                 Locker::InitSession |
                 Locker::NoSTDIN);
  auto dict = PythonModule::MainModule()
      .ResolveName<PythonDictionary>(m_dictionary_name);
  auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(
      callable_name, dict);
  if (!pfunc.IsAllocated()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "can't find callable: %s", callable_name.str().c_str());
  }
  llvm::Expected<PythonCallable::ArgInfo> arg_info = pfunc.GetArgInfo();
  if (!arg_info)
    return arg_info.takeError();
  return arg_info.get().max_positional_args;
}

static std::string GenerateUniqueName(const char *base_name_wanted,
                                      uint32_t &functions_counter,
                                      const void *name_token = nullptr) {
  StreamString sstr;

  if (!base_name_wanted)
    return std::string();

  if (!name_token)
    sstr.Printf("%s_%d", base_name_wanted, functions_counter++);
  else
    sstr.Printf("%s_%p", base_name_wanted, name_token);

  return std::string(sstr.GetString());
}

bool ScriptInterpreterPythonImpl::GetEmbeddedInterpreterModuleObjects() {
  if (m_run_one_line_function.IsValid())
    return true;

  PythonObject module(PyRefType::Borrowed,
                      PyImport_AddModule("lldb.embedded_interpreter"));
  if (!module.IsValid())
    return false;

  PythonDictionary module_dict(PyRefType::Borrowed,
                               PyModule_GetDict(module.get()));
  if (!module_dict.IsValid())
    return false;

  m_run_one_line_function =
      module_dict.GetItemForKey(PythonString("run_one_line"));
  m_run_one_line_str_global =
      module_dict.GetItemForKey(PythonString("g_run_one_line_str"));
  return m_run_one_line_function.IsValid();
}

bool ScriptInterpreterPythonImpl::ExecuteOneLine(
    llvm::StringRef command, CommandReturnObject *result,
    const ExecuteScriptOptions &options) {
  std::string command_str = command.str();

  if (!m_valid_session)
    return false;

  if (!command.empty()) {
    // We want to call run_one_line, passing in the dictionary and the command
    // string.  We cannot do this through PyRun_SimpleString here because the
    // command string may contain escaped characters, and putting it inside
    // another string to pass to PyRun_SimpleString messes up the escaping.  So
    // we use the following more complicated method to pass the command string
    // directly down to Python.
    llvm::Expected<std::unique_ptr<ScriptInterpreterIORedirect>>
        io_redirect_or_error = ScriptInterpreterIORedirect::Create(
            options.GetEnableIO(), m_debugger, result);
    if (!io_redirect_or_error) {
      if (result)
        result->AppendErrorWithFormatv(
            "failed to redirect I/O: {0}\n",
            llvm::fmt_consume(io_redirect_or_error.takeError()));
      else
        llvm::consumeError(io_redirect_or_error.takeError());
      return false;
    }

    ScriptInterpreterIORedirect &io_redirect = **io_redirect_or_error;

    bool success = false;
    {
      // WARNING!  It's imperative that this RAII scope be as tight as
      // possible. In particular, the scope must end *before* we try to join
      // the read thread.  The reason for this is that a pre-requisite for
      // joining the read thread is that we close the write handle (to break
      // the pipe and cause it to wake up and exit).  But acquiring the GIL as
      // below will redirect Python's stdio to use this same handle.  If we
      // close the handle while Python is still using it, bad things will
      // happen.
      Locker locker(
          this,
          Locker::AcquireLock | Locker::InitSession |
              (options.GetSetLLDBGlobals() ? Locker::InitGlobals : 0) |
              ((result && result->GetInteractive()) ? 0 : Locker::NoSTDIN),
          Locker::FreeAcquiredLock | Locker::TearDownSession,
          io_redirect.GetInputFile(), io_redirect.GetOutputFile(),
          io_redirect.GetErrorFile());

      // Find the correct script interpreter dictionary in the main module.
      PythonDictionary &session_dict = GetSessionDictionary();
      if (session_dict.IsValid()) {
        if (GetEmbeddedInterpreterModuleObjects()) {
          if (PyCallable_Check(m_run_one_line_function.get())) {
            PythonObject pargs(
                PyRefType::Owned,
                Py_BuildValue("(Os)", session_dict.get(), command_str.c_str()));
            if (pargs.IsValid()) {
              PythonObject return_value(
                  PyRefType::Owned,
                  PyObject_CallObject(m_run_one_line_function.get(),
                                      pargs.get()));
              if (return_value.IsValid())
                success = true;
              else if (options.GetMaskoutErrors() && PyErr_Occurred()) {
                PyErr_Print();
                PyErr_Clear();
              }
            }
          }
        }
      }

      io_redirect.Flush();
    }

    if (success)
      return true;

    // The one-liner failed.  Append the error message.
    if (result) {
      result->AppendErrorWithFormat(
          "python failed attempting to evaluate '%s'\n", command_str.c_str());
    }
    return false;
  }

  if (result)
    result->AppendError("empty command passed to python\n");
  return false;
}

void ScriptInterpreterPythonImpl::ExecuteInterpreterLoop() {
  LLDB_SCOPED_TIMER();

  Debugger &debugger = m_debugger;

  // At the moment, the only time the debugger does not have an input file
  // handle is when this is called directly from Python, in which case it is
  // both dangerous and unnecessary (not to mention confusing) to try to embed
  // a running interpreter loop inside the already running Python interpreter
  // loop, so we won't do it.

  if (!debugger.GetInputFile().IsValid())
    return;

  IOHandlerSP io_handler_sp(new IOHandlerPythonInterpreter(debugger, this));
  if (io_handler_sp) {
    debugger.RunIOHandlerAsync(io_handler_sp);
  }
}

bool ScriptInterpreterPythonImpl::Interrupt() {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SCRIPT));

  if (IsExecutingPython()) {
    PyThreadState *state = PyThreadState_GET();
    if (!state)
      state = GetThreadState();
    if (state) {
      long tid = state->thread_id;
      PyThreadState_Swap(state);
      int num_threads = PyThreadState_SetAsyncExc(tid, PyExc_KeyboardInterrupt);
      LLDB_LOGF(log,
                "ScriptInterpreterPythonImpl::Interrupt() sending "
                "PyExc_KeyboardInterrupt (tid = %li, num_threads = %i)...",
                tid, num_threads);
      return true;
    }
  }
  LLDB_LOGF(log,
            "ScriptInterpreterPythonImpl::Interrupt() python code not running, "
            "can't interrupt");
  return false;
}

bool ScriptInterpreterPythonImpl::ExecuteOneLineWithReturn(
    llvm::StringRef in_string, ScriptInterpreter::ScriptReturnType return_type,
    void *ret_value, const ExecuteScriptOptions &options) {

  Locker locker(this,
                Locker::AcquireLock | Locker::InitSession |
                    (options.GetSetLLDBGlobals() ? Locker::InitGlobals : 0) |
                    Locker::NoSTDIN,
                Locker::FreeAcquiredLock | Locker::TearDownSession);

  PythonModule &main_module = GetMainModule();
  PythonDictionary globals = main_module.GetDictionary();

  PythonDictionary locals = GetSessionDictionary();
  if (!locals.IsValid())
    locals = unwrapIgnoringErrors(
        As<PythonDictionary>(globals.GetAttribute(m_dictionary_name)));
  if (!locals.IsValid())
    locals = globals;

  Expected<PythonObject> maybe_py_return =
      runStringOneLine(in_string, globals, locals);

  if (!maybe_py_return) {
    llvm::handleAllErrors(
        maybe_py_return.takeError(),
        [&](PythonException &E) {
          E.Restore();
          if (options.GetMaskoutErrors()) {
            if (E.Matches(PyExc_SyntaxError)) {
              PyErr_Print();
            }
            PyErr_Clear();
          }
        },
        [](const llvm::ErrorInfoBase &E) {});
    return false;
  }

  PythonObject py_return = std::move(maybe_py_return.get());
  assert(py_return.IsValid());

  switch (return_type) {
  case eScriptReturnTypeCharPtr: // "char *"
  {
    const char format[3] = "s#";
    return PyArg_Parse(py_return.get(), format, (char **)ret_value);
  }
  case eScriptReturnTypeCharStrOrNone: // char* or NULL if py_return ==
                                       // Py_None
  {
    const char format[3] = "z";
    return PyArg_Parse(py_return.get(), format, (char **)ret_value);
  }
  case eScriptReturnTypeBool: {
    const char format[2] = "b";
    return PyArg_Parse(py_return.get(), format, (bool *)ret_value);
  }
  case eScriptReturnTypeShortInt: {
    const char format[2] = "h";
    return PyArg_Parse(py_return.get(), format, (short *)ret_value);
  }
  case eScriptReturnTypeShortIntUnsigned: {
    const char format[2] = "H";
    return PyArg_Parse(py_return.get(), format, (unsigned short *)ret_value);
  }
  case eScriptReturnTypeInt: {
    const char format[2] = "i";
    return PyArg_Parse(py_return.get(), format, (int *)ret_value);
  }
  case eScriptReturnTypeIntUnsigned: {
    const char format[2] = "I";
    return PyArg_Parse(py_return.get(), format, (unsigned int *)ret_value);
  }
  case eScriptReturnTypeLongInt: {
    const char format[2] = "l";
    return PyArg_Parse(py_return.get(), format, (long *)ret_value);
  }
  case eScriptReturnTypeLongIntUnsigned: {
    const char format[2] = "k";
    return PyArg_Parse(py_return.get(), format, (unsigned long *)ret_value);
  }
  case eScriptReturnTypeLongLong: {
    const char format[2] = "L";
    return PyArg_Parse(py_return.get(), format, (long long *)ret_value);
  }
  case eScriptReturnTypeLongLongUnsigned: {
    const char format[2] = "K";
    return PyArg_Parse(py_return.get(), format,
                       (unsigned long long *)ret_value);
  }
  case eScriptReturnTypeFloat: {
    const char format[2] = "f";
    return PyArg_Parse(py_return.get(), format, (float *)ret_value);
  }
  case eScriptReturnTypeDouble: {
    const char format[2] = "d";
    return PyArg_Parse(py_return.get(), format, (double *)ret_value);
  }
  case eScriptReturnTypeChar: {
    const char format[2] = "c";
    return PyArg_Parse(py_return.get(), format, (char *)ret_value);
  }
  case eScriptReturnTypeOpaqueObject: {
    *((PyObject **)ret_value) = py_return.release();
    return true;
  }
  }
  llvm_unreachable("Fully covered switch!");
}

Status ScriptInterpreterPythonImpl::ExecuteMultipleLines(
    const char *in_string, const ExecuteScriptOptions &options) {

  if (in_string == nullptr)
    return Status();

  Locker locker(this,
                Locker::AcquireLock | Locker::InitSession |
                    (options.GetSetLLDBGlobals() ? Locker::InitGlobals : 0) |
                    Locker::NoSTDIN,
                Locker::FreeAcquiredLock | Locker::TearDownSession);

  PythonModule &main_module = GetMainModule();
  PythonDictionary globals = main_module.GetDictionary();

  PythonDictionary locals = GetSessionDictionary();
  if (!locals.IsValid())
    locals = unwrapIgnoringErrors(
        As<PythonDictionary>(globals.GetAttribute(m_dictionary_name)));
  if (!locals.IsValid())
    locals = globals;

  Expected<PythonObject> return_value =
      runStringMultiLine(in_string, globals, locals);

  if (!return_value) {
    llvm::Error error =
        llvm::handleErrors(return_value.takeError(), [&](PythonException &E) {
          llvm::Error error = llvm::createStringError(
              llvm::inconvertibleErrorCode(), E.ReadBacktrace());
          if (!options.GetMaskoutErrors())
            E.Restore();
          return error;
        });
    return Status(std::move(error));
  }

  return Status();
}

void ScriptInterpreterPythonImpl::CollectDataForBreakpointCommandCallback(
    std::vector<BreakpointOptions *> &bp_options_vec,
    CommandReturnObject &result) {
  m_active_io_handler = eIOHandlerBreakpoint;
  m_debugger.GetCommandInterpreter().GetPythonCommandsFromIOHandler(
      "    ", *this, &bp_options_vec);
}

void ScriptInterpreterPythonImpl::CollectDataForWatchpointCommandCallback(
    WatchpointOptions *wp_options, CommandReturnObject &result) {
  m_active_io_handler = eIOHandlerWatchpoint;
  m_debugger.GetCommandInterpreter().GetPythonCommandsFromIOHandler(
      "    ", *this, wp_options);
}

Status ScriptInterpreterPythonImpl::SetBreakpointCommandCallbackFunction(
    BreakpointOptions *bp_options, const char *function_name,
    StructuredData::ObjectSP extra_args_sp) {
  Status error;
  // For now just cons up a oneliner that calls the provided function.
  std::string oneliner("return ");
  oneliner += function_name;

  llvm::Expected<unsigned> maybe_args =
      GetMaxPositionalArgumentsForCallable(function_name);
  if (!maybe_args) {
    error.SetErrorStringWithFormat(
        "could not get num args: %s",
        llvm::toString(maybe_args.takeError()).c_str());
    return error;
  }
  size_t max_args = *maybe_args;

  bool uses_extra_args = false;
  if (max_args >= 4) {
    uses_extra_args = true;
    oneliner += "(frame, bp_loc, extra_args, internal_dict)";
  } else if (max_args >= 3) {
    if (extra_args_sp) {
      error.SetErrorString("cannot pass extra_args to a three argument callback"
                          );
      return error;
    }
    uses_extra_args = false;
    oneliner += "(frame, bp_loc, internal_dict)";
  } else {
    error.SetErrorStringWithFormat("expected 3 or 4 argument "
                                   "function, %s can only take %zu",
                                   function_name, max_args);
    return error;
  }

  SetBreakpointCommandCallback(bp_options, oneliner.c_str(), extra_args_sp,
                               uses_extra_args);
  return error;
}

Status ScriptInterpreterPythonImpl::SetBreakpointCommandCallback(
    BreakpointOptions *bp_options,
    std::unique_ptr<BreakpointOptions::CommandData> &cmd_data_up) {
  Status error;
  error = GenerateBreakpointCommandCallbackData(cmd_data_up->user_source,
                                                cmd_data_up->script_source,
                                                false);
  if (error.Fail()) {
    return error;
  }
  auto baton_sp =
      std::make_shared<BreakpointOptions::CommandBaton>(std::move(cmd_data_up));
  bp_options->SetCallback(
      ScriptInterpreterPythonImpl::BreakpointCallbackFunction, baton_sp);
  return error;
}

Status ScriptInterpreterPythonImpl::SetBreakpointCommandCallback(
    BreakpointOptions *bp_options, const char *command_body_text) {
  return SetBreakpointCommandCallback(bp_options, command_body_text, {},false);
}

// Set a Python one-liner as the callback for the breakpoint.
Status ScriptInterpreterPythonImpl::SetBreakpointCommandCallback(
    BreakpointOptions *bp_options, const char *command_body_text,
    StructuredData::ObjectSP extra_args_sp,
    bool uses_extra_args) {
  auto data_up = std::make_unique<CommandDataPython>(extra_args_sp);
  // Split the command_body_text into lines, and pass that to
  // GenerateBreakpointCommandCallbackData.  That will wrap the body in an
  // auto-generated function, and return the function name in script_source.
  // That is what the callback will actually invoke.

  data_up->user_source.SplitIntoLines(command_body_text);
  Status error = GenerateBreakpointCommandCallbackData(data_up->user_source,
                                                       data_up->script_source,
                                                       uses_extra_args);
  if (error.Success()) {
    auto baton_sp =
        std::make_shared<BreakpointOptions::CommandBaton>(std::move(data_up));
    bp_options->SetCallback(
        ScriptInterpreterPythonImpl::BreakpointCallbackFunction, baton_sp);
    return error;
  }
  return error;
}

// Set a Python one-liner as the callback for the watchpoint.
void ScriptInterpreterPythonImpl::SetWatchpointCommandCallback(
    WatchpointOptions *wp_options, const char *oneliner) {
  auto data_up = std::make_unique<WatchpointOptions::CommandData>();

  // It's necessary to set both user_source and script_source to the oneliner.
  // The former is used to generate callback description (as in watchpoint
  // command list) while the latter is used for Python to interpret during the
  // actual callback.

  data_up->user_source.AppendString(oneliner);
  data_up->script_source.assign(oneliner);

  if (GenerateWatchpointCommandCallbackData(data_up->user_source,
                                            data_up->script_source)) {
    auto baton_sp =
        std::make_shared<WatchpointOptions::CommandBaton>(std::move(data_up));
    wp_options->SetCallback(
        ScriptInterpreterPythonImpl::WatchpointCallbackFunction, baton_sp);
  }

  return;
}

Status ScriptInterpreterPythonImpl::ExportFunctionDefinitionToInterpreter(
    StringList &function_def) {
  // Convert StringList to one long, newline delimited, const char *.
  std::string function_def_string(function_def.CopyList());

  Status error = ExecuteMultipleLines(
      function_def_string.c_str(),
      ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false));
  return error;
}

Status ScriptInterpreterPythonImpl::GenerateFunction(const char *signature,
                                                     const StringList &input) {
  Status error;
  int num_lines = input.GetSize();
  if (num_lines == 0) {
    error.SetErrorString("No input data.");
    return error;
  }

  if (!signature || *signature == 0) {
    error.SetErrorString("No output function name.");
    return error;
  }

  StreamString sstr;
  StringList auto_generated_function;
  auto_generated_function.AppendString(signature);
  auto_generated_function.AppendString(
      "     global_dict = globals()"); // Grab the global dictionary
  auto_generated_function.AppendString(
      "     new_keys = internal_dict.keys()"); // Make a list of keys in the
                                               // session dict
  auto_generated_function.AppendString(
      "     old_keys = global_dict.keys()"); // Save list of keys in global dict
  auto_generated_function.AppendString(
      "     global_dict.update (internal_dict)"); // Add the session dictionary
                                                  // to the
  // global dictionary.

  // Wrap everything up inside the function, increasing the indentation.

  auto_generated_function.AppendString("     if True:");
  for (int i = 0; i < num_lines; ++i) {
    sstr.Clear();
    sstr.Printf("       %s", input.GetStringAtIndex(i));
    auto_generated_function.AppendString(sstr.GetData());
  }
  auto_generated_function.AppendString(
      "     for key in new_keys:"); // Iterate over all the keys from session
                                    // dict
  auto_generated_function.AppendString(
      "         internal_dict[key] = global_dict[key]"); // Update session dict
                                                         // values
  auto_generated_function.AppendString(
      "         if key not in old_keys:"); // If key was not originally in
                                           // global dict
  auto_generated_function.AppendString(
      "             del global_dict[key]"); //  ...then remove key/value from
                                            //  global dict

  // Verify that the results are valid Python.

  error = ExportFunctionDefinitionToInterpreter(auto_generated_function);

  return error;
}

bool ScriptInterpreterPythonImpl::GenerateTypeScriptFunction(
    StringList &user_input, std::string &output, const void *name_token) {
  static uint32_t num_created_functions = 0;
  user_input.RemoveBlankLines();
  StreamString sstr;

  // Check to see if we have any data; if not, just return.
  if (user_input.GetSize() == 0)
    return false;

  // Take what the user wrote, wrap it all up inside one big auto-generated
  // Python function, passing in the ValueObject as parameter to the function.

  std::string auto_generated_function_name(
      GenerateUniqueName("lldb_autogen_python_type_print_func",
                         num_created_functions, name_token));
  sstr.Printf("def %s (valobj, internal_dict):",
              auto_generated_function_name.c_str());

  if (!GenerateFunction(sstr.GetData(), user_input).Success())
    return false;

  // Store the name of the auto-generated function to be called.
  output.assign(auto_generated_function_name);
  return true;
}

bool ScriptInterpreterPythonImpl::GenerateScriptAliasFunction(
    StringList &user_input, std::string &output) {
  static uint32_t num_created_functions = 0;
  user_input.RemoveBlankLines();
  StreamString sstr;

  // Check to see if we have any data; if not, just return.
  if (user_input.GetSize() == 0)
    return false;

  std::string auto_generated_function_name(GenerateUniqueName(
      "lldb_autogen_python_cmd_alias_func", num_created_functions));

  sstr.Printf("def %s (debugger, args, result, internal_dict):",
              auto_generated_function_name.c_str());

  if (!GenerateFunction(sstr.GetData(), user_input).Success())
    return false;

  // Store the name of the auto-generated function to be called.
  output.assign(auto_generated_function_name);
  return true;
}

bool ScriptInterpreterPythonImpl::GenerateTypeSynthClass(
    StringList &user_input, std::string &output, const void *name_token) {
  static uint32_t num_created_classes = 0;
  user_input.RemoveBlankLines();
  int num_lines = user_input.GetSize();
  StreamString sstr;

  // Check to see if we have any data; if not, just return.
  if (user_input.GetSize() == 0)
    return false;

  // Wrap all user input into a Python class

  std::string auto_generated_class_name(GenerateUniqueName(
      "lldb_autogen_python_type_synth_class", num_created_classes, name_token));

  StringList auto_generated_class;

  // Create the function name & definition string.

  sstr.Printf("class %s:", auto_generated_class_name.c_str());
  auto_generated_class.AppendString(sstr.GetString());

  // Wrap everything up inside the class, increasing the indentation. we don't
  // need to play any fancy indentation tricks here because there is no
  // surrounding code whose indentation we need to honor
  for (int i = 0; i < num_lines; ++i) {
    sstr.Clear();
    sstr.Printf("     %s", user_input.GetStringAtIndex(i));
    auto_generated_class.AppendString(sstr.GetString());
  }

  // Verify that the results are valid Python. (even though the method is
  // ExportFunctionDefinitionToInterpreter, a class will actually be exported)
  // (TODO: rename that method to ExportDefinitionToInterpreter)
  if (!ExportFunctionDefinitionToInterpreter(auto_generated_class).Success())
    return false;

  // Store the name of the auto-generated class

  output.assign(auto_generated_class_name);
  return true;
}

StructuredData::GenericSP
ScriptInterpreterPythonImpl::CreateFrameRecognizer(const char *class_name) {
  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::GenericSP();

  void *ret_val;

  {
    Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);
    ret_val = LLDBSWIGPython_CreateFrameRecognizer(class_name,
                                                   m_dictionary_name.c_str());
  }

  return StructuredData::GenericSP(new StructuredPythonObject(ret_val));
}

lldb::ValueObjectListSP ScriptInterpreterPythonImpl::GetRecognizedArguments(
    const StructuredData::ObjectSP &os_plugin_object_sp,
    lldb::StackFrameSP frame_sp) {
  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  if (!os_plugin_object_sp)
    return ValueObjectListSP();

  StructuredData::Generic *generic = os_plugin_object_sp->GetAsGeneric();
  if (!generic)
    return nullptr;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());

  if (!implementor.IsAllocated())
    return ValueObjectListSP();

  PythonObject py_return(PyRefType::Owned,
                         (PyObject *)LLDBSwigPython_GetRecognizedArguments(
                             implementor.get(), frame_sp));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }
  if (py_return.get()) {
    PythonList result_list(PyRefType::Borrowed, py_return.get());
    ValueObjectListSP result = ValueObjectListSP(new ValueObjectList());
    for (size_t i = 0; i < result_list.GetSize(); i++) {
      PyObject *item = result_list.GetItemAtIndex(i).get();
      lldb::SBValue *sb_value_ptr =
          (lldb::SBValue *)LLDBSWIGPython_CastPyObjectToSBValue(item);
      auto valobj_sp = LLDBSWIGPython_GetValueObjectSPFromSBValue(sb_value_ptr);
      if (valobj_sp)
        result->Append(valobj_sp);
    }
    return result;
  }
  return ValueObjectListSP();
}

StructuredData::GenericSP
ScriptInterpreterPythonImpl::OSPlugin_CreatePluginObject(
    const char *class_name, lldb::ProcessSP process_sp) {
  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::GenericSP();

  if (!process_sp)
    return StructuredData::GenericSP();

  void *ret_val;

  {
    Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);
    ret_val = LLDBSWIGPythonCreateOSPlugin(
        class_name, m_dictionary_name.c_str(), process_sp);
  }

  return StructuredData::GenericSP(new StructuredPythonObject(ret_val));
}

StructuredData::DictionarySP ScriptInterpreterPythonImpl::OSPlugin_RegisterInfo(
    StructuredData::ObjectSP os_plugin_object_sp) {
  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_register_info";

  if (!os_plugin_object_sp)
    return StructuredData::DictionarySP();

  StructuredData::Generic *generic = os_plugin_object_sp->GetAsGeneric();
  if (!generic)
    return nullptr;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());

  if (!implementor.IsAllocated())
    return StructuredData::DictionarySP();

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return StructuredData::DictionarySP();

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();

    return StructuredData::DictionarySP();
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, nullptr));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }
  if (py_return.get()) {
    PythonDictionary result_dict(PyRefType::Borrowed, py_return.get());
    return result_dict.CreateStructuredDictionary();
  }
  return StructuredData::DictionarySP();
}

StructuredData::ArraySP ScriptInterpreterPythonImpl::OSPlugin_ThreadsInfo(
    StructuredData::ObjectSP os_plugin_object_sp) {
  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_thread_info";

  if (!os_plugin_object_sp)
    return StructuredData::ArraySP();

  StructuredData::Generic *generic = os_plugin_object_sp->GetAsGeneric();
  if (!generic)
    return nullptr;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());

  if (!implementor.IsAllocated())
    return StructuredData::ArraySP();

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return StructuredData::ArraySP();

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();

    return StructuredData::ArraySP();
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, nullptr));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.get()) {
    PythonList result_list(PyRefType::Borrowed, py_return.get());
    return result_list.CreateStructuredArray();
  }
  return StructuredData::ArraySP();
}

StructuredData::StringSP
ScriptInterpreterPythonImpl::OSPlugin_RegisterContextData(
    StructuredData::ObjectSP os_plugin_object_sp, lldb::tid_t tid) {
  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_register_data";
  static char *param_format =
      const_cast<char *>(GetPythonValueFormatString(tid));

  if (!os_plugin_object_sp)
    return StructuredData::StringSP();

  StructuredData::Generic *generic = os_plugin_object_sp->GetAsGeneric();
  if (!generic)
    return nullptr;
  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());

  if (!implementor.IsAllocated())
    return StructuredData::StringSP();

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return StructuredData::StringSP();

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return StructuredData::StringSP();
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, param_format, tid));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.get()) {
    PythonBytes result(PyRefType::Borrowed, py_return.get());
    return result.CreateStructuredString();
  }
  return StructuredData::StringSP();
}

StructuredData::DictionarySP ScriptInterpreterPythonImpl::OSPlugin_CreateThread(
    StructuredData::ObjectSP os_plugin_object_sp, lldb::tid_t tid,
    lldb::addr_t context) {
  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "create_thread";
  std::string param_format;
  param_format += GetPythonValueFormatString(tid);
  param_format += GetPythonValueFormatString(context);

  if (!os_plugin_object_sp)
    return StructuredData::DictionarySP();

  StructuredData::Generic *generic = os_plugin_object_sp->GetAsGeneric();
  if (!generic)
    return nullptr;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());

  if (!implementor.IsAllocated())
    return StructuredData::DictionarySP();

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return StructuredData::DictionarySP();

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return StructuredData::DictionarySP();
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(PyRefType::Owned,
                         PyObject_CallMethod(implementor.get(), callee_name,
                                             &param_format[0], tid, context));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.get()) {
    PythonDictionary result_dict(PyRefType::Borrowed, py_return.get());
    return result_dict.CreateStructuredDictionary();
  }
  return StructuredData::DictionarySP();
}

StructuredData::ObjectSP ScriptInterpreterPythonImpl::CreateScriptedThreadPlan(
    const char *class_name, StructuredDataImpl *args_data,
    std::string &error_str, lldb::ThreadPlanSP thread_plan_sp) {
  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::ObjectSP();

  if (!thread_plan_sp.get())
    return {};

  Debugger &debugger = thread_plan_sp->GetTarget().GetDebugger();
  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(debugger);

  if (!python_interpreter)
    return {};

  void *ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPythonCreateScriptedThreadPlan(
        class_name, python_interpreter->m_dictionary_name.c_str(),
        args_data, error_str, thread_plan_sp);
    if (!ret_val)
      return {};
  }

  return StructuredData::ObjectSP(new StructuredPythonObject(ret_val));
}

bool ScriptInterpreterPythonImpl::ScriptedThreadPlanExplainsStop(
    StructuredData::ObjectSP implementor_sp, Event *event, bool &script_error) {
  bool explains_stop = true;
  StructuredData::Generic *generic = nullptr;
  if (implementor_sp)
    generic = implementor_sp->GetAsGeneric();
  if (generic) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    explains_stop = LLDBSWIGPythonCallThreadPlan(
        generic->GetValue(), "explains_stop", event, script_error);
    if (script_error)
      return true;
  }
  return explains_stop;
}

bool ScriptInterpreterPythonImpl::ScriptedThreadPlanShouldStop(
    StructuredData::ObjectSP implementor_sp, Event *event, bool &script_error) {
  bool should_stop = true;
  StructuredData::Generic *generic = nullptr;
  if (implementor_sp)
    generic = implementor_sp->GetAsGeneric();
  if (generic) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    should_stop = LLDBSWIGPythonCallThreadPlan(
        generic->GetValue(), "should_stop", event, script_error);
    if (script_error)
      return true;
  }
  return should_stop;
}

bool ScriptInterpreterPythonImpl::ScriptedThreadPlanIsStale(
    StructuredData::ObjectSP implementor_sp, bool &script_error) {
  bool is_stale = true;
  StructuredData::Generic *generic = nullptr;
  if (implementor_sp)
    generic = implementor_sp->GetAsGeneric();
  if (generic) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    is_stale = LLDBSWIGPythonCallThreadPlan(generic->GetValue(), "is_stale",
                                            nullptr, script_error);
    if (script_error)
      return true;
  }
  return is_stale;
}

lldb::StateType ScriptInterpreterPythonImpl::ScriptedThreadPlanGetRunState(
    StructuredData::ObjectSP implementor_sp, bool &script_error) {
  bool should_step = false;
  StructuredData::Generic *generic = nullptr;
  if (implementor_sp)
    generic = implementor_sp->GetAsGeneric();
  if (generic) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    should_step = LLDBSWIGPythonCallThreadPlan(
        generic->GetValue(), "should_step", nullptr, script_error);
    if (script_error)
      should_step = true;
  }
  if (should_step)
    return lldb::eStateStepping;
  return lldb::eStateRunning;
}

StructuredData::GenericSP
ScriptInterpreterPythonImpl::CreateScriptedBreakpointResolver(
    const char *class_name, StructuredDataImpl *args_data,
    lldb::BreakpointSP &bkpt_sp) {

  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::GenericSP();

  if (!bkpt_sp.get())
    return StructuredData::GenericSP();

  Debugger &debugger = bkpt_sp->GetTarget().GetDebugger();
  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(debugger);

  if (!python_interpreter)
    return StructuredData::GenericSP();

  void *ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);

    ret_val = LLDBSwigPythonCreateScriptedBreakpointResolver(
        class_name, python_interpreter->m_dictionary_name.c_str(), args_data,
        bkpt_sp);
  }

  return StructuredData::GenericSP(new StructuredPythonObject(ret_val));
}

bool ScriptInterpreterPythonImpl::ScriptedBreakpointResolverSearchCallback(
    StructuredData::GenericSP implementor_sp, SymbolContext *sym_ctx) {
  bool should_continue = false;

  if (implementor_sp) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    should_continue = LLDBSwigPythonCallBreakpointResolver(
        implementor_sp->GetValue(), "__callback__", sym_ctx);
    if (PyErr_Occurred()) {
      PyErr_Print();
      PyErr_Clear();
    }
  }
  return should_continue;
}

lldb::SearchDepth
ScriptInterpreterPythonImpl::ScriptedBreakpointResolverSearchDepth(
    StructuredData::GenericSP implementor_sp) {
  int depth_as_int = lldb::eSearchDepthModule;
  if (implementor_sp) {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    depth_as_int = LLDBSwigPythonCallBreakpointResolver(
        implementor_sp->GetValue(), "__get_depth__", nullptr);
    if (PyErr_Occurred()) {
      PyErr_Print();
      PyErr_Clear();
    }
  }
  if (depth_as_int == lldb::eSearchDepthInvalid)
    return lldb::eSearchDepthModule;

  if (depth_as_int <= lldb::kLastSearchDepthKind)
    return (lldb::SearchDepth)depth_as_int;
  return lldb::eSearchDepthModule;
}

StructuredData::GenericSP ScriptInterpreterPythonImpl::CreateScriptedStopHook(
    TargetSP target_sp, const char *class_name, StructuredDataImpl *args_data,
    Status &error) {

  if (!target_sp) {
    error.SetErrorString("No target for scripted stop-hook.");
    return StructuredData::GenericSP();
  }

  if (class_name == nullptr || class_name[0] == '\0') {
    error.SetErrorString("No class name for scripted stop-hook.");
    return StructuredData::GenericSP();
  }

  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(m_debugger);

  if (!python_interpreter) {
    error.SetErrorString("No script interpreter for scripted stop-hook.");
    return StructuredData::GenericSP();
  }

  void *ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);

    ret_val = LLDBSwigPythonCreateScriptedStopHook(
        target_sp, class_name, python_interpreter->m_dictionary_name.c_str(),
        args_data, error);
  }

  return StructuredData::GenericSP(new StructuredPythonObject(ret_val));
}

bool ScriptInterpreterPythonImpl::ScriptedStopHookHandleStop(
    StructuredData::GenericSP implementor_sp, ExecutionContext &exc_ctx,
    lldb::StreamSP stream_sp) {
  assert(implementor_sp &&
         "can't call a stop hook with an invalid implementor");
  assert(stream_sp && "can't call a stop hook with an invalid stream");

  Locker py_lock(this,
                 Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);

  lldb::ExecutionContextRefSP exc_ctx_ref_sp(new ExecutionContextRef(exc_ctx));

  bool ret_val = LLDBSwigPythonStopHookCallHandleStop(
      implementor_sp->GetValue(), exc_ctx_ref_sp, stream_sp);
  return ret_val;
}

StructuredData::ObjectSP
ScriptInterpreterPythonImpl::LoadPluginModule(const FileSpec &file_spec,
                                              lldb_private::Status &error) {
  if (!FileSystem::Instance().Exists(file_spec)) {
    error.SetErrorString("no such file");
    return StructuredData::ObjectSP();
  }

  StructuredData::ObjectSP module_sp;

  if (LoadScriptingModule(file_spec.GetPath().c_str(), true, error, &module_sp))
    return module_sp;

  return StructuredData::ObjectSP();
}

StructuredData::DictionarySP ScriptInterpreterPythonImpl::GetDynamicSettings(
    StructuredData::ObjectSP plugin_module_sp, Target *target,
    const char *setting_name, lldb_private::Status &error) {
  if (!plugin_module_sp || !target || !setting_name || !setting_name[0])
    return StructuredData::DictionarySP();
  StructuredData::Generic *generic = plugin_module_sp->GetAsGeneric();
  if (!generic)
    return StructuredData::DictionarySP();

  Locker py_lock(this,
                 Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
  TargetSP target_sp(target->shared_from_this());

  auto setting = (PyObject *)LLDBSWIGPython_GetDynamicSetting(
      generic->GetValue(), setting_name, target_sp);

  if (!setting)
    return StructuredData::DictionarySP();

  PythonDictionary py_dict =
      unwrapIgnoringErrors(As<PythonDictionary>(Take<PythonObject>(setting)));

  if (!py_dict)
    return StructuredData::DictionarySP();

  return py_dict.CreateStructuredDictionary();
}

StructuredData::ObjectSP
ScriptInterpreterPythonImpl::CreateSyntheticScriptedProvider(
    const char *class_name, lldb::ValueObjectSP valobj) {
  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::ObjectSP();

  if (!valobj.get())
    return StructuredData::ObjectSP();

  ExecutionContext exe_ctx(valobj->GetExecutionContextRef());
  Target *target = exe_ctx.GetTargetPtr();

  if (!target)
    return StructuredData::ObjectSP();

  Debugger &debugger = target->GetDebugger();
  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(debugger);

  if (!python_interpreter)
    return StructuredData::ObjectSP();

  void *ret_val = nullptr;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPythonCreateSyntheticProvider(
        class_name, python_interpreter->m_dictionary_name.c_str(), valobj);
  }

  return StructuredData::ObjectSP(new StructuredPythonObject(ret_val));
}

StructuredData::GenericSP
ScriptInterpreterPythonImpl::CreateScriptCommandObject(const char *class_name) {
  DebuggerSP debugger_sp(m_debugger.shared_from_this());

  if (class_name == nullptr || class_name[0] == '\0')
    return StructuredData::GenericSP();

  if (!debugger_sp.get())
    return StructuredData::GenericSP();

  void *ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPythonCreateCommandObject(
        class_name, m_dictionary_name.c_str(), debugger_sp);
  }

  return StructuredData::GenericSP(new StructuredPythonObject(ret_val));
}

bool ScriptInterpreterPythonImpl::GenerateTypeScriptFunction(
    const char *oneliner, std::string &output, const void *name_token) {
  StringList input;
  input.SplitIntoLines(oneliner, strlen(oneliner));
  return GenerateTypeScriptFunction(input, output, name_token);
}

bool ScriptInterpreterPythonImpl::GenerateTypeSynthClass(
    const char *oneliner, std::string &output, const void *name_token) {
  StringList input;
  input.SplitIntoLines(oneliner, strlen(oneliner));
  return GenerateTypeSynthClass(input, output, name_token);
}

Status ScriptInterpreterPythonImpl::GenerateBreakpointCommandCallbackData(
    StringList &user_input, std::string &output,
    bool has_extra_args) {
  static uint32_t num_created_functions = 0;
  user_input.RemoveBlankLines();
  StreamString sstr;
  Status error;
  if (user_input.GetSize() == 0) {
    error.SetErrorString("No input data.");
    return error;
  }

  std::string auto_generated_function_name(GenerateUniqueName(
      "lldb_autogen_python_bp_callback_func_", num_created_functions));
  if (has_extra_args)
    sstr.Printf("def %s (frame, bp_loc, extra_args, internal_dict):",
                auto_generated_function_name.c_str());
  else
    sstr.Printf("def %s (frame, bp_loc, internal_dict):",
                auto_generated_function_name.c_str());

  error = GenerateFunction(sstr.GetData(), user_input);
  if (!error.Success())
    return error;

  // Store the name of the auto-generated function to be called.
  output.assign(auto_generated_function_name);
  return error;
}

bool ScriptInterpreterPythonImpl::GenerateWatchpointCommandCallbackData(
    StringList &user_input, std::string &output) {
  static uint32_t num_created_functions = 0;
  user_input.RemoveBlankLines();
  StreamString sstr;

  if (user_input.GetSize() == 0)
    return false;

  std::string auto_generated_function_name(GenerateUniqueName(
      "lldb_autogen_python_wp_callback_func_", num_created_functions));
  sstr.Printf("def %s (frame, wp, internal_dict):",
              auto_generated_function_name.c_str());

  if (!GenerateFunction(sstr.GetData(), user_input).Success())
    return false;

  // Store the name of the auto-generated function to be called.
  output.assign(auto_generated_function_name);
  return true;
}

bool ScriptInterpreterPythonImpl::GetScriptedSummary(
    const char *python_function_name, lldb::ValueObjectSP valobj,
    StructuredData::ObjectSP &callee_wrapper_sp,
    const TypeSummaryOptions &options, std::string &retval) {

  LLDB_SCOPED_TIMER();

  if (!valobj.get()) {
    retval.assign("<no object>");
    return false;
  }

  void *old_callee = nullptr;
  StructuredData::Generic *generic = nullptr;
  if (callee_wrapper_sp) {
    generic = callee_wrapper_sp->GetAsGeneric();
    if (generic)
      old_callee = generic->GetValue();
  }
  void *new_callee = old_callee;

  bool ret_val;
  if (python_function_name && *python_function_name) {
    {
      Locker py_lock(this, Locker::AcquireLock | Locker::InitSession |
                               Locker::NoSTDIN);
      {
        TypeSummaryOptionsSP options_sp(new TypeSummaryOptions(options));

        static Timer::Category func_cat("LLDBSwigPythonCallTypeScript");
        Timer scoped_timer(func_cat, "LLDBSwigPythonCallTypeScript");
        ret_val = LLDBSwigPythonCallTypeScript(
            python_function_name, GetSessionDictionary().get(), valobj,
            &new_callee, options_sp, retval);
      }
    }
  } else {
    retval.assign("<no function name>");
    return false;
  }

  if (new_callee && old_callee != new_callee)
    callee_wrapper_sp = std::make_shared<StructuredPythonObject>(new_callee);

  return ret_val;
}

bool ScriptInterpreterPythonImpl::BreakpointCallbackFunction(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  CommandDataPython *bp_option_data = (CommandDataPython *)baton;
  const char *python_function_name = bp_option_data->script_source.c_str();

  if (!context)
    return true;

  ExecutionContext exe_ctx(context->exe_ctx_ref);
  Target *target = exe_ctx.GetTargetPtr();

  if (!target)
    return true;

  Debugger &debugger = target->GetDebugger();
  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(debugger);

  if (!python_interpreter)
    return true;

  if (python_function_name && python_function_name[0]) {
    const StackFrameSP stop_frame_sp(exe_ctx.GetFrameSP());
    BreakpointSP breakpoint_sp = target->GetBreakpointByID(break_id);
    if (breakpoint_sp) {
      const BreakpointLocationSP bp_loc_sp(
          breakpoint_sp->FindLocationByID(break_loc_id));

      if (stop_frame_sp && bp_loc_sp) {
        bool ret_val = true;
        {
          Locker py_lock(python_interpreter, Locker::AcquireLock |
                                                 Locker::InitSession |
                                                 Locker::NoSTDIN);
          Expected<bool> maybe_ret_val =
              LLDBSwigPythonBreakpointCallbackFunction(
                  python_function_name,
                  python_interpreter->m_dictionary_name.c_str(), stop_frame_sp,
                  bp_loc_sp, bp_option_data->m_extra_args_up.get());

          if (!maybe_ret_val) {

            llvm::handleAllErrors(
                maybe_ret_val.takeError(),
                [&](PythonException &E) {
                  debugger.GetErrorStream() << E.ReadBacktrace();
                },
                [&](const llvm::ErrorInfoBase &E) {
                  debugger.GetErrorStream() << E.message();
                });

          } else {
            ret_val = maybe_ret_val.get();
          }
        }
        return ret_val;
      }
    }
  }
  // We currently always true so we stop in case anything goes wrong when
  // trying to call the script function
  return true;
}

bool ScriptInterpreterPythonImpl::WatchpointCallbackFunction(
    void *baton, StoppointCallbackContext *context, user_id_t watch_id) {
  WatchpointOptions::CommandData *wp_option_data =
      (WatchpointOptions::CommandData *)baton;
  const char *python_function_name = wp_option_data->script_source.c_str();

  if (!context)
    return true;

  ExecutionContext exe_ctx(context->exe_ctx_ref);
  Target *target = exe_ctx.GetTargetPtr();

  if (!target)
    return true;

  Debugger &debugger = target->GetDebugger();
  ScriptInterpreterPythonImpl *python_interpreter =
      GetPythonInterpreter(debugger);

  if (!python_interpreter)
    return true;

  if (python_function_name && python_function_name[0]) {
    const StackFrameSP stop_frame_sp(exe_ctx.GetFrameSP());
    WatchpointSP wp_sp = target->GetWatchpointList().FindByID(watch_id);
    if (wp_sp) {
      if (stop_frame_sp && wp_sp) {
        bool ret_val = true;
        {
          Locker py_lock(python_interpreter, Locker::AcquireLock |
                                                 Locker::InitSession |
                                                 Locker::NoSTDIN);
          ret_val = LLDBSwigPythonWatchpointCallbackFunction(
              python_function_name,
              python_interpreter->m_dictionary_name.c_str(), stop_frame_sp,
              wp_sp);
        }
        return ret_val;
      }
    }
  }
  // We currently always true so we stop in case anything goes wrong when
  // trying to call the script function
  return true;
}

size_t ScriptInterpreterPythonImpl::CalculateNumChildren(
    const StructuredData::ObjectSP &implementor_sp, uint32_t max) {
  if (!implementor_sp)
    return 0;
  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return 0;
  void *implementor = generic->GetValue();
  if (!implementor)
    return 0;

  size_t ret_val = 0;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPython_CalculateNumChildren(implementor, max);
  }

  return ret_val;
}

lldb::ValueObjectSP ScriptInterpreterPythonImpl::GetChildAtIndex(
    const StructuredData::ObjectSP &implementor_sp, uint32_t idx) {
  if (!implementor_sp)
    return lldb::ValueObjectSP();

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return lldb::ValueObjectSP();
  void *implementor = generic->GetValue();
  if (!implementor)
    return lldb::ValueObjectSP();

  lldb::ValueObjectSP ret_val;
  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    void *child_ptr = LLDBSwigPython_GetChildAtIndex(implementor, idx);
    if (child_ptr != nullptr && child_ptr != Py_None) {
      lldb::SBValue *sb_value_ptr =
          (lldb::SBValue *)LLDBSWIGPython_CastPyObjectToSBValue(child_ptr);
      if (sb_value_ptr == nullptr)
        Py_XDECREF(child_ptr);
      else
        ret_val = LLDBSWIGPython_GetValueObjectSPFromSBValue(sb_value_ptr);
    } else {
      Py_XDECREF(child_ptr);
    }
  }

  return ret_val;
}

int ScriptInterpreterPythonImpl::GetIndexOfChildWithName(
    const StructuredData::ObjectSP &implementor_sp, const char *child_name) {
  if (!implementor_sp)
    return UINT32_MAX;

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return UINT32_MAX;
  void *implementor = generic->GetValue();
  if (!implementor)
    return UINT32_MAX;

  int ret_val = UINT32_MAX;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPython_GetIndexOfChildWithName(implementor, child_name);
  }

  return ret_val;
}

bool ScriptInterpreterPythonImpl::UpdateSynthProviderInstance(
    const StructuredData::ObjectSP &implementor_sp) {
  bool ret_val = false;

  if (!implementor_sp)
    return ret_val;

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return ret_val;
  void *implementor = generic->GetValue();
  if (!implementor)
    return ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSwigPython_UpdateSynthProviderInstance(implementor);
  }

  return ret_val;
}

bool ScriptInterpreterPythonImpl::MightHaveChildrenSynthProviderInstance(
    const StructuredData::ObjectSP &implementor_sp) {
  bool ret_val = false;

  if (!implementor_sp)
    return ret_val;

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return ret_val;
  void *implementor = generic->GetValue();
  if (!implementor)
    return ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val =
        LLDBSwigPython_MightHaveChildrenSynthProviderInstance(implementor);
  }

  return ret_val;
}

lldb::ValueObjectSP ScriptInterpreterPythonImpl::GetSyntheticValue(
    const StructuredData::ObjectSP &implementor_sp) {
  lldb::ValueObjectSP ret_val(nullptr);

  if (!implementor_sp)
    return ret_val;

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return ret_val;
  void *implementor = generic->GetValue();
  if (!implementor)
    return ret_val;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    void *child_ptr = LLDBSwigPython_GetValueSynthProviderInstance(implementor);
    if (child_ptr != nullptr && child_ptr != Py_None) {
      lldb::SBValue *sb_value_ptr =
          (lldb::SBValue *)LLDBSWIGPython_CastPyObjectToSBValue(child_ptr);
      if (sb_value_ptr == nullptr)
        Py_XDECREF(child_ptr);
      else
        ret_val = LLDBSWIGPython_GetValueObjectSPFromSBValue(sb_value_ptr);
    } else {
      Py_XDECREF(child_ptr);
    }
  }

  return ret_val;
}

ConstString ScriptInterpreterPythonImpl::GetSyntheticTypeName(
    const StructuredData::ObjectSP &implementor_sp) {
  Locker py_lock(this,
                 Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);

  static char callee_name[] = "get_type_name";

  ConstString ret_val;
  bool got_string = false;
  std::string buffer;

  if (!implementor_sp)
    return ret_val;

  StructuredData::Generic *generic = implementor_sp->GetAsGeneric();
  if (!generic)
    return ret_val;
  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)generic->GetValue());
  if (!implementor.IsAllocated())
    return ret_val;

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return ret_val;

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return ret_val;
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, nullptr));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.IsAllocated() && PythonString::Check(py_return.get())) {
    PythonString py_string(PyRefType::Borrowed, py_return.get());
    llvm::StringRef return_data(py_string.GetString());
    if (!return_data.empty()) {
      buffer.assign(return_data.data(), return_data.size());
      got_string = true;
    }
  }

  if (got_string)
    ret_val.SetCStringWithLength(buffer.c_str(), buffer.size());

  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptFormatKeyword(
    const char *impl_function, Process *process, std::string &output,
    Status &error) {
  bool ret_val;
  if (!process) {
    error.SetErrorString("no process");
    return false;
  }
  if (!impl_function || !impl_function[0]) {
    error.SetErrorString("no function to execute");
    return false;
  }

  {
    ProcessSP process_sp(process->shared_from_this());
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSWIGPythonRunScriptKeywordProcess(
        impl_function, m_dictionary_name.c_str(), process_sp, output);
    if (!ret_val)
      error.SetErrorString("python script evaluation failed");
  }
  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptFormatKeyword(
    const char *impl_function, Thread *thread, std::string &output,
    Status &error) {
  bool ret_val;
  if (!thread) {
    error.SetErrorString("no thread");
    return false;
  }
  if (!impl_function || !impl_function[0]) {
    error.SetErrorString("no function to execute");
    return false;
  }

  {
    ThreadSP thread_sp(thread->shared_from_this());
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSWIGPythonRunScriptKeywordThread(
        impl_function, m_dictionary_name.c_str(), thread_sp, output);
    if (!ret_val)
      error.SetErrorString("python script evaluation failed");
  }
  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptFormatKeyword(
    const char *impl_function, Target *target, std::string &output,
    Status &error) {
  bool ret_val;
  if (!target) {
    error.SetErrorString("no thread");
    return false;
  }
  if (!impl_function || !impl_function[0]) {
    error.SetErrorString("no function to execute");
    return false;
  }

  {
    TargetSP target_sp(target->shared_from_this());
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSWIGPythonRunScriptKeywordTarget(
        impl_function, m_dictionary_name.c_str(), target_sp, output);
    if (!ret_val)
      error.SetErrorString("python script evaluation failed");
  }
  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptFormatKeyword(
    const char *impl_function, StackFrame *frame, std::string &output,
    Status &error) {
  bool ret_val;
  if (!frame) {
    error.SetErrorString("no frame");
    return false;
  }
  if (!impl_function || !impl_function[0]) {
    error.SetErrorString("no function to execute");
    return false;
  }

  {
    StackFrameSP frame_sp(frame->shared_from_this());
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSWIGPythonRunScriptKeywordFrame(
        impl_function, m_dictionary_name.c_str(), frame_sp, output);
    if (!ret_val)
      error.SetErrorString("python script evaluation failed");
  }
  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptFormatKeyword(
    const char *impl_function, ValueObject *value, std::string &output,
    Status &error) {
  bool ret_val;
  if (!value) {
    error.SetErrorString("no value");
    return false;
  }
  if (!impl_function || !impl_function[0]) {
    error.SetErrorString("no function to execute");
    return false;
  }

  {
    ValueObjectSP value_sp(value->GetSP());
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
    ret_val = LLDBSWIGPythonRunScriptKeywordValue(
        impl_function, m_dictionary_name.c_str(), value_sp, output);
    if (!ret_val)
      error.SetErrorString("python script evaluation failed");
  }
  return ret_val;
}

uint64_t replace_all(std::string &str, const std::string &oldStr,
                     const std::string &newStr) {
  size_t pos = 0;
  uint64_t matches = 0;
  while ((pos = str.find(oldStr, pos)) != std::string::npos) {
    matches++;
    str.replace(pos, oldStr.length(), newStr);
    pos += newStr.length();
  }
  return matches;
}

bool ScriptInterpreterPythonImpl::LoadScriptingModule(
    const char *pathname, bool init_session, lldb_private::Status &error,
    StructuredData::ObjectSP *module_sp, FileSpec extra_search_dir) {
  namespace fs = llvm::sys::fs;
  namespace path = llvm::sys::path;

  if (!pathname || !pathname[0]) {
    error.SetErrorString("invalid pathname");
    return false;
  }

  lldb::DebuggerSP debugger_sp = m_debugger.shared_from_this();

  // Before executing Python code, lock the GIL.
  Locker py_lock(this,
                 Locker::AcquireLock |
                     (init_session ? Locker::InitSession : 0) | Locker::NoSTDIN,
                 Locker::FreeAcquiredLock |
                     (init_session ? Locker::TearDownSession : 0));

  auto ExtendSysPath = [this](std::string directory) -> llvm::Error {
    if (directory.empty()) {
      return llvm::make_error<llvm::StringError>(
          "invalid directory name", llvm::inconvertibleErrorCode());
    }

    replace_all(directory, "\\", "\\\\");
    replace_all(directory, "'", "\\'");

    // Make sure that Python has "directory" in the search path.
    StreamString command_stream;
    command_stream.Printf("if not (sys.path.__contains__('%s')):\n    "
                          "sys.path.insert(1,'%s');\n\n",
                          directory.c_str(), directory.c_str());
    bool syspath_retval =
        ExecuteMultipleLines(command_stream.GetData(),
                             ScriptInterpreter::ExecuteScriptOptions()
                                 .SetEnableIO(false)
                                 .SetSetLLDBGlobals(false))
            .Success();
    if (!syspath_retval) {
      return llvm::make_error<llvm::StringError>(
          "Python sys.path handling failed", llvm::inconvertibleErrorCode());
    }

    return llvm::Error::success();
  };

  std::string module_name(pathname);
  bool possible_package = false;

  if (extra_search_dir) {
    if (llvm::Error e = ExtendSysPath(extra_search_dir.GetPath())) {
      error = std::move(e);
      return false;
    }
  } else {
    FileSpec module_file(pathname);
    FileSystem::Instance().Resolve(module_file);
    FileSystem::Instance().Collect(module_file);

    fs::file_status st;
    std::error_code ec = status(module_file.GetPath(), st);

    if (ec || st.type() == fs::file_type::status_error ||
        st.type() == fs::file_type::type_unknown ||
        st.type() == fs::file_type::file_not_found) {
      // if not a valid file of any sort, check if it might be a filename still
      // dot can't be used but / and \ can, and if either is found, reject
      if (strchr(pathname, '\\') || strchr(pathname, '/')) {
        error.SetErrorString("invalid pathname");
        return false;
      }
      // Not a filename, probably a package of some sort, let it go through.
      possible_package = true;
    } else if (is_directory(st) || is_regular_file(st)) {
      if (module_file.GetDirectory().IsEmpty()) {
        error.SetErrorString("invalid directory name");
        return false;
      }
      if (llvm::Error e =
              ExtendSysPath(module_file.GetDirectory().GetCString())) {
        error = std::move(e);
        return false;
      }
      module_name = module_file.GetFilename().GetCString();
    } else {
      error.SetErrorString("no known way to import this module specification");
      return false;
    }
  }

  // Strip .py or .pyc extension
  llvm::StringRef extension = llvm::sys::path::extension(module_name);
  if (!extension.empty()) {
    if (extension == ".py")
      module_name.resize(module_name.length() - 3);
    else if (extension == ".pyc")
      module_name.resize(module_name.length() - 4);
  }

  if (!possible_package && module_name.find('.') != llvm::StringRef::npos) {
    error.SetErrorStringWithFormat(
        "Python does not allow dots in module names: %s", module_name.c_str());
    return false;
  }

  if (module_name.find('-') != llvm::StringRef::npos) {
    error.SetErrorStringWithFormat(
        "Python discourages dashes in module names: %s", module_name.c_str());
    return false;
  }

  // check if the module is already import-ed
  StreamString command_stream;
  command_stream.Clear();
  command_stream.Printf("sys.modules.__contains__('%s')", module_name.c_str());
  bool does_contain = false;
  // this call will succeed if the module was ever imported in any Debugger
  // in the lifetime of the process in which this LLDB framework is living
  const bool was_imported_globally =
      (ExecuteOneLineWithReturn(
           command_stream.GetData(),
           ScriptInterpreterPythonImpl::eScriptReturnTypeBool, &does_contain,
           ScriptInterpreter::ExecuteScriptOptions()
               .SetEnableIO(false)
               .SetSetLLDBGlobals(false)) &&
       does_contain);
  const bool was_imported_locally =
      GetSessionDictionary()
          .GetItemForKey(PythonString(module_name))
          .IsAllocated();

  // now actually do the import
  command_stream.Clear();

  if (was_imported_globally || was_imported_locally) {
    if (!was_imported_locally)
      command_stream.Printf("import %s ; reload_module(%s)",
                            module_name.c_str(), module_name.c_str());
    else
      command_stream.Printf("reload_module(%s)", module_name.c_str());
  } else
    command_stream.Printf("import %s", module_name.c_str());

  error = ExecuteMultipleLines(command_stream.GetData(),
                               ScriptInterpreter::ExecuteScriptOptions()
                                   .SetEnableIO(false)
                                   .SetSetLLDBGlobals(false));
  if (error.Fail())
    return false;

  // if we are here, everything worked
  // call __lldb_init_module(debugger,dict)
  if (!LLDBSwigPythonCallModuleInit(module_name.c_str(),
                                    m_dictionary_name.c_str(), debugger_sp)) {
    error.SetErrorString("calling __lldb_init_module failed");
    return false;
  }

  if (module_sp) {
    // everything went just great, now set the module object
    command_stream.Clear();
    command_stream.Printf("%s", module_name.c_str());
    void *module_pyobj = nullptr;
    if (ExecuteOneLineWithReturn(
            command_stream.GetData(),
            ScriptInterpreter::eScriptReturnTypeOpaqueObject, &module_pyobj) &&
        module_pyobj)
      *module_sp = std::make_shared<StructuredPythonObject>(module_pyobj);
  }

  return true;
}

bool ScriptInterpreterPythonImpl::IsReservedWord(const char *word) {
  if (!word || !word[0])
    return false;

  llvm::StringRef word_sr(word);

  // filter out a few characters that would just confuse us and that are
  // clearly not keyword material anyway
  if (word_sr.find('"') != llvm::StringRef::npos ||
      word_sr.find('\'') != llvm::StringRef::npos)
    return false;

  StreamString command_stream;
  command_stream.Printf("keyword.iskeyword('%s')", word);
  bool result;
  ExecuteScriptOptions options;
  options.SetEnableIO(false);
  options.SetMaskoutErrors(true);
  options.SetSetLLDBGlobals(false);
  if (ExecuteOneLineWithReturn(command_stream.GetData(),
                               ScriptInterpreter::eScriptReturnTypeBool,
                               &result, options))
    return result;
  return false;
}

ScriptInterpreterPythonImpl::SynchronicityHandler::SynchronicityHandler(
    lldb::DebuggerSP debugger_sp, ScriptedCommandSynchronicity synchro)
    : m_debugger_sp(debugger_sp), m_synch_wanted(synchro),
      m_old_asynch(debugger_sp->GetAsyncExecution()) {
  if (m_synch_wanted == eScriptedCommandSynchronicitySynchronous)
    m_debugger_sp->SetAsyncExecution(false);
  else if (m_synch_wanted == eScriptedCommandSynchronicityAsynchronous)
    m_debugger_sp->SetAsyncExecution(true);
}

ScriptInterpreterPythonImpl::SynchronicityHandler::~SynchronicityHandler() {
  if (m_synch_wanted != eScriptedCommandSynchronicityCurrentValue)
    m_debugger_sp->SetAsyncExecution(m_old_asynch);
}

bool ScriptInterpreterPythonImpl::RunScriptBasedCommand(
    const char *impl_function, llvm::StringRef args,
    ScriptedCommandSynchronicity synchronicity,
    lldb_private::CommandReturnObject &cmd_retobj, Status &error,
    const lldb_private::ExecutionContext &exe_ctx) {
  if (!impl_function) {
    error.SetErrorString("no function to execute");
    return false;
  }

  lldb::DebuggerSP debugger_sp = m_debugger.shared_from_this();
  lldb::ExecutionContextRefSP exe_ctx_ref_sp(new ExecutionContextRef(exe_ctx));

  if (!debugger_sp.get()) {
    error.SetErrorString("invalid Debugger pointer");
    return false;
  }

  bool ret_val = false;

  std::string err_msg;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession |
                       (cmd_retobj.GetInteractive() ? 0 : Locker::NoSTDIN),
                   Locker::FreeLock | Locker::TearDownSession);

    SynchronicityHandler synch_handler(debugger_sp, synchronicity);

    std::string args_str = args.str();
    ret_val = LLDBSwigPythonCallCommand(
        impl_function, m_dictionary_name.c_str(), debugger_sp, args_str.c_str(),
        cmd_retobj, exe_ctx_ref_sp);
  }

  if (!ret_val)
    error.SetErrorString("unable to execute script function");
  else
    error.Clear();

  return ret_val;
}

bool ScriptInterpreterPythonImpl::RunScriptBasedCommand(
    StructuredData::GenericSP impl_obj_sp, llvm::StringRef args,
    ScriptedCommandSynchronicity synchronicity,
    lldb_private::CommandReturnObject &cmd_retobj, Status &error,
    const lldb_private::ExecutionContext &exe_ctx) {
  if (!impl_obj_sp || !impl_obj_sp->IsValid()) {
    error.SetErrorString("no function to execute");
    return false;
  }

  lldb::DebuggerSP debugger_sp = m_debugger.shared_from_this();
  lldb::ExecutionContextRefSP exe_ctx_ref_sp(new ExecutionContextRef(exe_ctx));

  if (!debugger_sp.get()) {
    error.SetErrorString("invalid Debugger pointer");
    return false;
  }

  bool ret_val = false;

  std::string err_msg;

  {
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::InitSession |
                       (cmd_retobj.GetInteractive() ? 0 : Locker::NoSTDIN),
                   Locker::FreeLock | Locker::TearDownSession);

    SynchronicityHandler synch_handler(debugger_sp, synchronicity);

    std::string args_str = args.str();
    ret_val = LLDBSwigPythonCallCommandObject(impl_obj_sp->GetValue(),
                                              debugger_sp, args_str.c_str(),
                                              cmd_retobj, exe_ctx_ref_sp);
  }

  if (!ret_val)
    error.SetErrorString("unable to execute script function");
  else
    error.Clear();

  return ret_val;
}

/// In Python, a special attribute __doc__ contains the docstring for an object
/// (function, method, class, ...) if any is defined Otherwise, the attribute's
/// value is None.
bool ScriptInterpreterPythonImpl::GetDocumentationForItem(const char *item,
                                                          std::string &dest) {
  dest.clear();

  if (!item || !*item)
    return false;

  std::string command(item);
  command += ".__doc__";

  // Python is going to point this to valid data if ExecuteOneLineWithReturn
  // returns successfully.
  char *result_ptr = nullptr;

  if (ExecuteOneLineWithReturn(
          command, ScriptInterpreter::eScriptReturnTypeCharStrOrNone,
          &result_ptr,
          ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false))) {
    if (result_ptr)
      dest.assign(result_ptr);
    return true;
  }

  StreamString str_stream;
  str_stream << "Function " << item
             << " was not found. Containing module might be missing.";
  dest = std::string(str_stream.GetString());

  return false;
}

bool ScriptInterpreterPythonImpl::GetShortHelpForCommandObject(
    StructuredData::GenericSP cmd_obj_sp, std::string &dest) {
  dest.clear();

  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_short_help";

  if (!cmd_obj_sp)
    return false;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)cmd_obj_sp->GetValue());

  if (!implementor.IsAllocated())
    return false;

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return false;

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return false;
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // Right now we know this function exists and is callable.
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, nullptr));

  // If it fails, print the error but otherwise go on.
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.IsAllocated() && PythonString::Check(py_return.get())) {
    PythonString py_string(PyRefType::Borrowed, py_return.get());
    llvm::StringRef return_data(py_string.GetString());
    dest.assign(return_data.data(), return_data.size());
    return true;
  }

  return false;
}

uint32_t ScriptInterpreterPythonImpl::GetFlagsForCommandObject(
    StructuredData::GenericSP cmd_obj_sp) {
  uint32_t result = 0;

  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_flags";

  if (!cmd_obj_sp)
    return result;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)cmd_obj_sp->GetValue());

  if (!implementor.IsAllocated())
    return result;

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return result;

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return result;
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  long long py_return = unwrapOrSetPythonException(
      As<long long>(implementor.CallMethod(callee_name)));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  } else {
    result = py_return;
  }

  return result;
}

bool ScriptInterpreterPythonImpl::GetLongHelpForCommandObject(
    StructuredData::GenericSP cmd_obj_sp, std::string &dest) {
  bool got_string = false;
  dest.clear();

  Locker py_lock(this, Locker::AcquireLock | Locker::NoSTDIN, Locker::FreeLock);

  static char callee_name[] = "get_long_help";

  if (!cmd_obj_sp)
    return false;

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)cmd_obj_sp->GetValue());

  if (!implementor.IsAllocated())
    return false;

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return false;

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();

    return false;
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  // right now we know this function exists and is callable..
  PythonObject py_return(
      PyRefType::Owned,
      PyObject_CallMethod(implementor.get(), callee_name, nullptr));

  // if it fails, print the error but otherwise go on
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.IsAllocated() && PythonString::Check(py_return.get())) {
    PythonString str(PyRefType::Borrowed, py_return.get());
    llvm::StringRef str_data(str.GetString());
    dest.assign(str_data.data(), str_data.size());
    got_string = true;
  }

  return got_string;
}

std::unique_ptr<ScriptInterpreterLocker>
ScriptInterpreterPythonImpl::AcquireInterpreterLock() {
  std::unique_ptr<ScriptInterpreterLocker> py_lock(new Locker(
      this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN,
      Locker::FreeLock | Locker::TearDownSession));
  return py_lock;
}

void ScriptInterpreterPythonImpl::InitializePrivate() {
  if (g_initialized)
    return;

  g_initialized = true;

  LLDB_SCOPED_TIMER();

  // RAII-based initialization which correctly handles multiple-initialization,
  // version- specific differences among Python 2 and Python 3, and saving and
  // restoring various other pieces of state that can get mucked with during
  // initialization.
  InitializePythonRAII initialize_guard;

  LLDBSwigPyInit();

  // Update the path python uses to search for modules to include the current
  // directory.

  PyRun_SimpleString("import sys");
  AddToSysPath(AddLocation::End, ".");

  // Don't denormalize paths when calling file_spec.GetPath().  On platforms
  // that use a backslash as the path separator, this will result in executing
  // python code containing paths with unescaped backslashes.  But Python also
  // accepts forward slashes, so to make life easier we just use that.
  if (FileSpec file_spec = GetPythonDir())
    AddToSysPath(AddLocation::Beginning, file_spec.GetPath(false));
  if (FileSpec file_spec = HostInfo::GetShlibDir())
    AddToSysPath(AddLocation::Beginning, file_spec.GetPath(false));

  PyRun_SimpleString("sys.dont_write_bytecode = 1; import "
                     "lldb.embedded_interpreter; from "
                     "lldb.embedded_interpreter import run_python_interpreter; "
                     "from lldb.embedded_interpreter import run_one_line");
}

void ScriptInterpreterPythonImpl::AddToSysPath(AddLocation location,
                                               std::string path) {
  std::string path_copy;

  std::string statement;
  if (location == AddLocation::Beginning) {
    statement.assign("sys.path.insert(0,\"");
    statement.append(path);
    statement.append("\")");
  } else {
    statement.assign("sys.path.append(\"");
    statement.append(path);
    statement.append("\")");
  }
  PyRun_SimpleString(statement.c_str());
}

// We are intentionally NOT calling Py_Finalize here (this would be the logical
// place to call it).  Calling Py_Finalize here causes test suite runs to seg
// fault:  The test suite runs in Python.  It registers SBDebugger::Terminate to
// be called 'at_exit'.  When the test suite Python harness finishes up, it
// calls Py_Finalize, which calls all the 'at_exit' registered functions.
// SBDebugger::Terminate calls Debugger::Terminate, which calls lldb::Terminate,
// which calls ScriptInterpreter::Terminate, which calls
// ScriptInterpreterPythonImpl::Terminate.  So if we call Py_Finalize here, we
// end up with Py_Finalize being called from within Py_Finalize, which results
// in a seg fault. Since this function only gets called when lldb is shutting
// down and going away anyway, the fact that we don't actually call Py_Finalize
// should not cause any problems (everything should shut down/go away anyway
// when the process exits).
//
// void ScriptInterpreterPythonImpl::Terminate() { Py_Finalize (); }

#endif
