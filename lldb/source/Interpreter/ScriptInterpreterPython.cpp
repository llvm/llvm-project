//===-- ScriptInterpreterPython.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In order to guarantee correct working with Python, Python.h *MUST* be
// the *FIRST* header file included here.
#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include <stdlib.h>
#include <stdio.h>

#include <string>

#include "lldb/API/SBValue.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Breakpoint/WatchpointOptions.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


static ScriptInterpreter::SWIGInitCallback g_swig_init_callback = NULL;
static ScriptInterpreter::SWIGBreakpointCallbackFunction g_swig_breakpoint_callback = NULL;
static ScriptInterpreter::SWIGWatchpointCallbackFunction g_swig_watchpoint_callback = NULL;
static ScriptInterpreter::SWIGPythonTypeScriptCallbackFunction g_swig_typescript_callback = NULL;
static ScriptInterpreter::SWIGPythonCreateSyntheticProvider g_swig_synthetic_script = NULL;
static ScriptInterpreter::SWIGPythonCalculateNumChildren g_swig_calc_children = NULL;
static ScriptInterpreter::SWIGPythonGetChildAtIndex g_swig_get_child_index = NULL;
static ScriptInterpreter::SWIGPythonGetIndexOfChildWithName g_swig_get_index_child = NULL;
static ScriptInterpreter::SWIGPythonCastPyObjectToSBValue g_swig_cast_to_sbvalue  = NULL;
static ScriptInterpreter::SWIGPythonUpdateSynthProviderInstance g_swig_update_provider = NULL;
static ScriptInterpreter::SWIGPythonMightHaveChildrenSynthProviderInstance g_swig_mighthavechildren_provider = NULL;
static ScriptInterpreter::SWIGPythonCallCommand g_swig_call_command = NULL;
static ScriptInterpreter::SWIGPythonCallModuleInit g_swig_call_module_init = NULL;
static ScriptInterpreter::SWIGPythonCreateOSPlugin g_swig_create_os_plugin = NULL;

// these are the Pythonic implementations of the required callbacks
// these are scripting-language specific, which is why they belong here
// we still need to use function pointers to them instead of relying
// on linkage-time resolution because the SWIG stuff and this file
// get built at different times
extern "C" bool
LLDBSwigPythonBreakpointCallbackFunction 
(
 const char *python_function_name,
 const char *session_dictionary_name,
 const lldb::StackFrameSP& sb_frame, 
 const lldb::BreakpointLocationSP& sb_bp_loc
 );

extern "C" bool
LLDBSwigPythonWatchpointCallbackFunction 
(
 const char *python_function_name,
 const char *session_dictionary_name,
 const lldb::StackFrameSP& sb_frame, 
 const lldb::WatchpointSP& sb_wp
 );

extern "C" bool
LLDBSwigPythonCallTypeScript 
(
 const char *python_function_name,
 void *session_dictionary,
 const lldb::ValueObjectSP& valobj_sp,
 void** pyfunct_wrapper,
 std::string& retval
 );

extern "C" void*
LLDBSwigPythonCreateSyntheticProvider 
(
 const std::string python_class_name,
 const char *session_dictionary_name,
 const lldb::ValueObjectSP& valobj_sp
 );


extern "C" uint32_t       LLDBSwigPython_CalculateNumChildren                   (void *implementor);
extern "C" void*          LLDBSwigPython_GetChildAtIndex                        (void *implementor, uint32_t idx);
extern "C" int            LLDBSwigPython_GetIndexOfChildWithName                (void *implementor, const char* child_name);
extern "C" void*          LLDBSWIGPython_CastPyObjectToSBValue                  (void* data);
extern "C" bool           LLDBSwigPython_UpdateSynthProviderInstance            (void* implementor);
extern "C" bool           LLDBSwigPython_MightHaveChildrenSynthProviderInstance (void* implementor);

extern "C" bool           LLDBSwigPythonCallCommand 
(
 const char *python_function_name,
 const char *session_dictionary_name,
 lldb::DebuggerSP& debugger,
 const char* args,
 std::string& err_msg,
 lldb_private::CommandReturnObject& cmd_retobj
 );

extern "C" bool           LLDBSwigPythonCallModuleInit 
(
 const std::string python_module_name,
 const char *session_dictionary_name,
 lldb::DebuggerSP& debugger
 );

extern "C" void*        LLDBSWIGPythonCreateOSPlugin
(
 const std::string python_class_name,
 const char *session_dictionary_name,
 const lldb::ProcessSP& process_sp
);

static int
_check_and_flush (FILE *stream)
{
  int prev_fail = ferror (stream);
  return fflush (stream) || prev_fail ? EOF : 0;
}

ScriptInterpreterPython::Locker::Locker (ScriptInterpreterPython *py_interpreter,
                                         uint16_t on_entry,
                                         uint16_t on_leave,
                                         FILE* wait_msg_handle) :
    m_need_session( (on_leave & TearDownSession) == TearDownSession ),
    m_python_interpreter(py_interpreter),
    m_tmp_fh(wait_msg_handle)
{
    if (m_python_interpreter && !m_tmp_fh)
        m_tmp_fh = (m_python_interpreter->m_dbg_stdout ? m_python_interpreter->m_dbg_stdout : stdout);

    DoAcquireLock();
    if ( (on_entry & InitSession) == InitSession )
        DoInitSession();
}

bool
ScriptInterpreterPython::Locker::DoAcquireLock()
{
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    m_GILState = PyGILState_Ensure();
    if (log)
        log->Printf("Ensured PyGILState. Previous state = %slocked\n", m_GILState == PyGILState_UNLOCKED ? "un" : "");
    return true;
}

bool
ScriptInterpreterPython::Locker::DoInitSession()
{
    if (!m_python_interpreter)
        return false;
    m_python_interpreter->EnterSession ();
    return true;
}

bool
ScriptInterpreterPython::Locker::DoFreeLock()
{
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    if (log)
        log->Printf("Releasing PyGILState. Returning to state = %slocked\n", m_GILState == PyGILState_UNLOCKED ? "un" : "");
    PyGILState_Release(m_GILState);
    return true;
}

bool
ScriptInterpreterPython::Locker::DoTearDownSession()
{
    if (!m_python_interpreter)
        return false;
    m_python_interpreter->LeaveSession ();
    return true;
}

ScriptInterpreterPython::Locker::~Locker()
{
    if (m_need_session)
        DoTearDownSession();
    DoFreeLock();
}

ScriptInterpreterPython::PythonInputReaderManager::PythonInputReaderManager (ScriptInterpreterPython *interpreter) :
m_interpreter(interpreter),
m_debugger_sp(),
m_reader_sp(),
m_error(false)
{
    if (m_interpreter == NULL)
    {
        m_error = true;
        return;
    }
    
    m_debugger_sp = m_interpreter->GetCommandInterpreter().GetDebugger().shared_from_this();
    
    if (!m_debugger_sp)
    {
        m_error = true;
        return;
    }

    m_reader_sp = InputReaderSP(new InputReader(*m_debugger_sp.get()));
    
    if (!m_reader_sp)
    {
        m_error = true;
        return;
    }
    
    Error error (m_reader_sp->Initialize (ScriptInterpreterPython::PythonInputReaderManager::InputReaderCallback,
                                          m_interpreter,                // baton
                                          eInputReaderGranularityLine,  // token size, to pass to callback function
                                          NULL,                         // end token
                                          NULL,                         // prompt
                                          true));                       // echo input
    if (error.Fail())
        m_error = true;
    else
    {
        m_debugger_sp->PushInputReader (m_reader_sp);
        m_interpreter->m_embedded_thread_input_reader_sp = m_reader_sp;
    }
}

ScriptInterpreterPython::PythonInputReaderManager::~PythonInputReaderManager()
{
    // Nothing to do if either m_interpreter or m_reader_sp is invalid.
    if (!m_interpreter || !m_reader_sp)
        return;

    m_reader_sp->SetIsDone (true);
    if (m_debugger_sp)
        m_debugger_sp->PopInputReader(m_reader_sp);

    // Only mess with m_interpreter's counterpart if, indeed, they are the same object.
    if (m_reader_sp.get() == m_interpreter->m_embedded_thread_input_reader_sp.get())
    {
        m_interpreter->m_embedded_thread_pty.CloseSlaveFileDescriptor();
        m_interpreter->m_embedded_thread_input_reader_sp.reset();
    }
}

size_t
ScriptInterpreterPython::PythonInputReaderManager::InputReaderCallback
(
 void *baton, 
 InputReader &reader, 
 InputReaderAction notification,
 const char *bytes, 
 size_t bytes_len
 )
{
    lldb::thread_t embedded_interpreter_thread;
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));
    
    if (baton == NULL)
        return 0;
    
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;
    
    if (script_interpreter->m_script_lang != eScriptLanguagePython)
        return 0;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    
    switch (notification)
    {
        case eInputReaderActivate:
        {
            // Save terminal settings if we can
            int input_fd = reader.GetDebugger().GetInputFile().GetDescriptor();
            if (input_fd == File::kInvalidDescriptor)
                input_fd = STDIN_FILENO;
            
            script_interpreter->SaveTerminalState(input_fd);

            char error_str[1024];
            if (script_interpreter->m_embedded_thread_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, error_str, 
                                                                                    sizeof(error_str)))
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, Activate, succeeded in opening master pty (fd = %d).",
                                 script_interpreter->m_embedded_thread_pty.GetMasterFileDescriptor());
                {
                    StreamString run_string;
                    char error_str[1024];
                    const char *pty_slave_name = script_interpreter->m_embedded_thread_pty.GetSlaveName (error_str, sizeof (error_str));
                    if (pty_slave_name != NULL && PyThreadState_GetDict() != NULL)
                    {
                        ScriptInterpreterPython::Locker locker(script_interpreter,
                                                               ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
                                                               ScriptInterpreterPython::Locker::FreeAcquiredLock);
                        run_string.Printf ("run_one_line (%s, 'save_stderr = sys.stderr')", script_interpreter->m_dictionary_name.c_str());
                        PyRun_SimpleString (run_string.GetData());
                        run_string.Clear ();
                        
                        run_string.Printf ("run_one_line (%s, 'sys.stderr = sys.stdout')", script_interpreter->m_dictionary_name.c_str());
                        PyRun_SimpleString (run_string.GetData());
                        run_string.Clear ();
                        
                        run_string.Printf ("run_one_line (%s, 'save_stdin = sys.stdin')", script_interpreter->m_dictionary_name.c_str());
                        PyRun_SimpleString (run_string.GetData());
                        run_string.Clear ();
                        
                        run_string.Printf ("run_one_line (%s, \"sys.stdin = open ('%s', 'r')\")", script_interpreter->m_dictionary_name.c_str(),
                                           pty_slave_name);
                        PyRun_SimpleString (run_string.GetData());
                        run_string.Clear ();
                    }
                }
                embedded_interpreter_thread = Host::ThreadCreate ("<lldb.script-interpreter.noninteractive-python>",
                                                                  ScriptInterpreterPython::PythonInputReaderManager::RunPythonInputReader,
                                                                  script_interpreter, NULL);
                if (IS_VALID_LLDB_HOST_THREAD(embedded_interpreter_thread))
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, Activate, succeeded in creating thread (thread_t = %p)", embedded_interpreter_thread);
                    Error detach_error;
                    Host::ThreadDetach (embedded_interpreter_thread, &detach_error);
                }
                else
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, Activate, failed in creating thread");
                    reader.SetIsDone (true);
                }
            }
            else
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, Activate, failed to open master pty ");
                reader.SetIsDone (true);
            }
        }
            break;
            
        case eInputReaderDeactivate:
			// When another input reader is pushed, don't leave the session...
            //script_interpreter->LeaveSession ();
            break;
            
        case eInputReaderReactivate:
//        {
//            ScriptInterpreterPython::Locker locker(script_interpreter,
//                                                   ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
//                                                   ScriptInterpreterPython::Locker::FreeAcquiredLock);
//        }
            break;
            
        case eInputReaderAsynchronousOutputWritten:
            break;
            
        case eInputReaderInterrupt:
            reader.SetIsDone(true);
            break;
            
        case eInputReaderEndOfFile:
            reader.SetIsDone(true);
            break;
            
        case eInputReaderGotToken:
            if (script_interpreter->m_embedded_thread_pty.GetMasterFileDescriptor() != -1)
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, GotToken, bytes='%s', byte_len = %lu", bytes,
                                 bytes_len);
                if (bytes && bytes_len)
                    ::write (script_interpreter->m_embedded_thread_pty.GetMasterFileDescriptor(), bytes, bytes_len);
                ::write (script_interpreter->m_embedded_thread_pty.GetMasterFileDescriptor(), "\n", 1);
            }
            else
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, GotToken, bytes='%s', byte_len = %lu, Master File Descriptor is bad.", 
                                 bytes,
                                 bytes_len);
                reader.SetIsDone (true);
            }
            
            break;
            
        case eInputReaderDone:
        {
            StreamString run_string;
            char error_str[1024];
            const char *pty_slave_name = script_interpreter->m_embedded_thread_pty.GetSlaveName (error_str, sizeof (error_str));
            if (pty_slave_name != NULL && PyThreadState_GetDict() != NULL)
            {
                ScriptInterpreterPython::Locker locker(script_interpreter,
                                                       ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
                                                       ScriptInterpreterPython::Locker::FreeAcquiredLock);
                run_string.Printf ("run_one_line (%s, 'sys.stdin = save_stdin')", script_interpreter->m_dictionary_name.c_str());
                PyRun_SimpleString (run_string.GetData());
                run_string.Clear();
                
                run_string.Printf ("run_one_line (%s, 'sys.stderr = save_stderr')", script_interpreter->m_dictionary_name.c_str());
                PyRun_SimpleString (run_string.GetData());
                run_string.Clear();
            }
        }
            
            // Restore terminal settings if they were validly saved
            if (log)
                log->Printf ("ScriptInterpreterPython::NonInteractiveInputReaderCallback, Done, closing down input reader.");
            
            script_interpreter->RestoreTerminalState ();
            
            script_interpreter->m_embedded_thread_pty.CloseMasterFileDescriptor();
            break;
    }
    
    return bytes_len;
}

ScriptInterpreterPython::ScriptInterpreterPython (CommandInterpreter &interpreter) :
    ScriptInterpreter (interpreter, eScriptLanguagePython),
    m_embedded_thread_pty (),
    m_embedded_python_pty (),
    m_embedded_thread_input_reader_sp (),
    m_embedded_python_input_reader_sp (),
    m_dbg_stdout (interpreter.GetDebugger().GetOutputFile().GetStream()),
    m_new_sysout (NULL),
    m_old_sysout (NULL),
    m_old_syserr (NULL),
    m_run_one_line (NULL),
    m_dictionary_name (interpreter.GetDebugger().GetInstanceName().AsCString()),
    m_terminal_state (),
    m_session_is_active (false),
    m_valid_session (true)
{

    static int g_initialized = false;
    
    if (!g_initialized)
    {
        g_initialized = true;
        ScriptInterpreterPython::InitializePrivate ();
    }

    m_dictionary_name.append("_dict");
    StreamString run_string;
    run_string.Printf ("%s = dict()", m_dictionary_name.c_str());

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock,
                  ScriptInterpreterPython::Locker::FreeAcquiredLock);
    PyRun_SimpleString (run_string.GetData());

    run_string.Clear();

    // Importing 'lldb' module calls SBDebugger::Initialize, which calls Debugger::Initialize, which increments a
    // global debugger ref-count; therefore we need to check the ref-count before and after importing lldb, and if the
    // ref-count increased we need to call Debugger::Terminate here to decrement the ref-count so that when the final 
    // call to Debugger::Terminate is made, the ref-count has the correct value. 
    //
    // Bonus question:  Why doesn't the ref-count always increase?  Because sometimes lldb has already been imported, in
    // which case the code inside it, including the call to SBDebugger::Initialize(), does not get executed.
    
    int old_count = Debugger::TestDebuggerRefCount();
    
    run_string.Printf ("run_one_line (%s, 'import copy, os, re, sys, uuid, lldb')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    // WARNING: temporary code that loads Cocoa formatters - this should be done on a per-platform basis rather than loading the whole set
    // and letting the individual formatter classes exploit APIs to check whether they can/cannot do their task
    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'import lldb.runtime.objc, lldb.formatters, lldb.formatters.objc, lldb.formatters.cpp, pydoc')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    int new_count = Debugger::TestDebuggerRefCount();
    
    if (new_count > old_count)
        Debugger::Terminate();

    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'lldb.debugger_unique_id = %llu; pydoc.pager = pydoc.plainpager')", m_dictionary_name.c_str(),
                       interpreter.GetDebugger().GetID());
    PyRun_SimpleString (run_string.GetData());
    
    if (m_dbg_stdout != NULL)
    {
        m_new_sysout = PyFile_FromFile (m_dbg_stdout, (char *) "", (char *) "w", _check_and_flush);
    }
    
    // get the output file handle from the debugger (if any)
    File& out_file = interpreter.GetDebugger().GetOutputFile();
    if (out_file.IsValid())
        ResetOutputFileHandle(out_file.GetStream());
}

ScriptInterpreterPython::~ScriptInterpreterPython ()
{
    Debugger &debugger = GetCommandInterpreter().GetDebugger();

    if (m_embedded_thread_input_reader_sp.get() != NULL)
    {
        m_embedded_thread_input_reader_sp->SetIsDone (true);
        m_embedded_thread_pty.CloseSlaveFileDescriptor();
        const InputReaderSP reader_sp = m_embedded_thread_input_reader_sp;
        debugger.PopInputReader (reader_sp);
        m_embedded_thread_input_reader_sp.reset();
    }
    
    if (m_embedded_python_input_reader_sp.get() != NULL)
    {
        m_embedded_python_input_reader_sp->SetIsDone (true);
        m_embedded_python_pty.CloseSlaveFileDescriptor();
        const InputReaderSP reader_sp = m_embedded_python_input_reader_sp;
        debugger.PopInputReader (reader_sp);
        m_embedded_python_input_reader_sp.reset();
    }
    
    if (m_new_sysout)
    {
        Locker locker(this,
                      ScriptInterpreterPython::Locker::AcquireLock,
                      ScriptInterpreterPython::Locker::FreeLock);
        Py_DECREF ((PyObject*)m_new_sysout);
    }
}

void
ScriptInterpreterPython::ResetOutputFileHandle (FILE *fh)
{
    if (fh == NULL)
        return;
        
    m_dbg_stdout = fh;

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock,
                  ScriptInterpreterPython::Locker::FreeAcquiredLock);

    m_new_sysout = PyFile_FromFile (m_dbg_stdout, (char *) "", (char *) "w", _check_and_flush);
}

void
ScriptInterpreterPython::SaveTerminalState (int fd)
{
    // Python mucks with the terminal state of STDIN. If we can possibly avoid
    // this by setting the file handles up correctly prior to entering the
    // interpreter we should. For now we save and restore the terminal state
    // on the input file handle.
    m_terminal_state.Save (fd, false);
}

void
ScriptInterpreterPython::RestoreTerminalState ()
{
    // Python mucks with the terminal state of STDIN. If we can possibly avoid
    // this by setting the file handles up correctly prior to entering the
    // interpreter we should. For now we save and restore the terminal state
    // on the input file handle.
    m_terminal_state.Restore();
}

void
ScriptInterpreterPython::LeaveSession ()
{
    // checking that we have a valid thread state - since we use our own threading and locking
    // in some (rare) cases during cleanup Python may end up believing we have no thread state
    // and PyImport_AddModule will crash if that is the case - since that seems to only happen
    // when destroying the SBDebugger, we can make do without clearing up stdout and stderr

    // rdar://problem/11292882
    // When the current thread state is NULL, PyThreadState_Get() issues a fatal error.
    if (PyThreadState_GetDict())
    {
        PyObject *sysmod = PyImport_AddModule ("sys");
        PyObject *sysdict = PyModule_GetDict (sysmod);

        if (m_new_sysout && sysmod && sysdict)
        {
            if (m_old_sysout)
                PyDict_SetItemString (sysdict, "stdout", (PyObject*)m_old_sysout);
            if (m_old_syserr)
                PyDict_SetItemString (sysdict, "stderr", (PyObject*)m_old_syserr);
        }
    }

    m_session_is_active = false;
}

void
ScriptInterpreterPython::EnterSession ()
{
    // If we have already entered the session, without having officially 'left' it, then there is no need to 
    // 'enter' it again.
    
    if (m_session_is_active)
        return;

    m_session_is_active = true;

    StreamString run_string;

    run_string.Printf (    "run_one_line (%s, 'lldb.debugger_unique_id = %llu", m_dictionary_name.c_str(), GetCommandInterpreter().GetDebugger().GetID());
    run_string.Printf (    "; lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%llu)", GetCommandInterpreter().GetDebugger().GetID());
    run_string.PutCString ("; lldb.target = lldb.debugger.GetSelectedTarget()");
    run_string.PutCString ("; lldb.process = lldb.target.GetProcess()");
    run_string.PutCString ("; lldb.thread = lldb.process.GetSelectedThread ()");
    run_string.PutCString ("; lldb.frame = lldb.thread.GetSelectedFrame ()");
    // Make sure STDIN is closed since when we run this as an embedded 
    // interpreter we don't want someone to call "line = sys.stdin.readline()"
    // and lock up. We don't have multiple windows and when the interpreter is
    // embedded we don't know we should be feeding input to the embedded 
    // interpreter or to the python sys.stdin. We also don't want to let python
    // play with the real stdin from this process, so we need to close it...
    //run_string.PutCString ("; sys.stdin.close()");
    run_string.PutCString ("')");

    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();

    PyObject *sysmod = PyImport_AddModule ("sys");
    PyObject *sysdict = PyModule_GetDict (sysmod);

    if (m_new_sysout && sysmod && sysdict)
    {
        m_old_sysout = PyDict_GetItemString(sysdict, "stdout");
        m_old_syserr = PyDict_GetItemString(sysdict, "stderr");
        if (m_new_sysout)
        {
            PyDict_SetItemString (sysdict, "stdout", (PyObject*)m_new_sysout);
            PyDict_SetItemString (sysdict, "stderr", (PyObject*)m_new_sysout);
        }
    }

    if (PyErr_Occurred())
        PyErr_Clear ();
}

static PyObject*
FindSessionDictionary (const char* dict_name)
{
    static std::map<ConstString,PyObject*> g_dict_map;
    
    ConstString dict(dict_name);
    
    std::map<ConstString,PyObject*>::iterator iter = g_dict_map.find(dict);
    
    if (iter != g_dict_map.end())
        return iter->second;
    
    PyObject *main_mod = PyImport_AddModule ("__main__");
    if (main_mod != NULL)
    {
        PyObject *main_dict = PyModule_GetDict (main_mod);
        if ((main_dict != NULL)
            && PyDict_Check (main_dict))
        {
            // Go through the main dictionary looking for the correct python script interpreter dictionary
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            
            while (PyDict_Next (main_dict, &pos, &key, &value))
            {
                // We have stolen references to the key and value objects in the dictionary; we need to increment 
                // them now so that Python's garbage collector doesn't collect them out from under us.
                Py_INCREF (key);
                Py_INCREF (value);
                if (strcmp (PyString_AsString (key), dict_name) == 0)
                {
                    g_dict_map[dict] = value;
                    return value;
                }
            }
        }
    }
    return NULL;
}

static std::string
GenerateUniqueName (const char* base_name_wanted,
                    uint32_t& functions_counter,
                    void* name_token = NULL)
{
    StreamString sstr;
    
    if (!base_name_wanted)
        return std::string();
    
    if (!name_token)
        sstr.Printf ("%s_%d", base_name_wanted, functions_counter++);
    else
        sstr.Printf ("%s_%p", base_name_wanted, name_token);
    
    return sstr.GetString();
}

bool
ScriptInterpreterPython::ExecuteOneLine (const char *command, CommandReturnObject *result, const ExecuteScriptOptions &options)
{
    if (!m_valid_session)
        return false;
        
    // We want to call run_one_line, passing in the dictionary and the command string.  We cannot do this through
    // PyRun_SimpleString here because the command string may contain escaped characters, and putting it inside
    // another string to pass to PyRun_SimpleString messes up the escaping.  So we use the following more complicated
    // method to pass the command string directly down to Python.

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitSession : 0),
                  ScriptInterpreterPython::Locker::FreeAcquiredLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::TearDownSession : 0));

    bool success = false;

    if (command)
    {
        // Find the correct script interpreter dictionary in the main module.
        PyObject *script_interpreter_dict = FindSessionDictionary(m_dictionary_name.c_str());
        if (script_interpreter_dict != NULL)
        {
            PyObject *pfunc = (PyObject*)m_run_one_line;
            PyObject *pmod = PyImport_AddModule ("lldb.embedded_interpreter");
            if (pmod != NULL)
            {
                PyObject *pmod_dict = PyModule_GetDict (pmod);
                if ((pmod_dict != NULL)
                    && PyDict_Check (pmod_dict))
                {
                    if (!pfunc)
                    {
                        PyObject *key, *value;
                        Py_ssize_t pos = 0;
                        
                        while (PyDict_Next (pmod_dict, &pos, &key, &value))
                        {
                            Py_INCREF (key);
                            Py_INCREF (value);
                            if (strcmp (PyString_AsString (key), "run_one_line") == 0)
                            {
                                pfunc = value;
                                break;
                            }
                        }
                        m_run_one_line = pfunc;
                    }
                    
                    if (pfunc && PyCallable_Check (pfunc))
                    {
                        PyObject *pargs = Py_BuildValue("(Os)",script_interpreter_dict,command);
                        if (pargs != NULL)
                        {
                            PyObject *pvalue = NULL;
                            { // scope for PythonInputReaderManager
                                PythonInputReaderManager py_input(options.GetEnableIO() ? this : NULL);
                                pvalue = PyObject_CallObject (pfunc, pargs);
                            }
                            Py_DECREF (pargs);
                            if (pvalue != NULL)
                            {
                                Py_DECREF (pvalue);
                                success = true;
                            }
                            else if (options.GetMaskoutErrors() && PyErr_Occurred ())
                            {
                                PyErr_Print();
                                PyErr_Clear();
                            }
                        }
                    }
                }
            }
            Py_INCREF (script_interpreter_dict);
        }

        if (success)
            return true;

        // The one-liner failed.  Append the error message.
        if (result)
            result->AppendErrorWithFormat ("python failed attempting to evaluate '%s'\n", command);
        return false;
    }

    if (result)
        result->AppendError ("empty command passed to python\n");
    return false;
}

size_t
ScriptInterpreterPython::InputReaderCallback
(
    void *baton, 
    InputReader &reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    lldb::thread_t embedded_interpreter_thread;
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));

    if (baton == NULL)
        return 0;
        
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;
    
    if (script_interpreter->m_script_lang != eScriptLanguagePython)
        return 0;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        {
            if (!batch_mode)
            {
                out_stream->Printf ("Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.\n");
                out_stream->Flush();
            }

            // Save terminal settings if we can
            int input_fd = reader.GetDebugger().GetInputFile().GetDescriptor();
            if (input_fd == File::kInvalidDescriptor)
                input_fd = STDIN_FILENO;

            script_interpreter->SaveTerminalState(input_fd);

            {
                ScriptInterpreterPython::Locker locker(script_interpreter,
                                                       ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
                                                       ScriptInterpreterPython::Locker::FreeAcquiredLock);
            }

            char error_str[1024];
            if (script_interpreter->m_embedded_python_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, error_str, 
                                                                                    sizeof(error_str)))
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, succeeded in opening master pty (fd = %d).",
                                  script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor());
                embedded_interpreter_thread = Host::ThreadCreate ("<lldb.script-interpreter.embedded-python-loop>",
                                                                  ScriptInterpreterPython::RunEmbeddedPythonInterpreter,
                                                                  script_interpreter, NULL);
                if (IS_VALID_LLDB_HOST_THREAD(embedded_interpreter_thread))
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, succeeded in creating thread (thread_t = %p)", embedded_interpreter_thread);
                    Error detach_error;
                    Host::ThreadDetach (embedded_interpreter_thread, &detach_error);
                }
                else
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, failed in creating thread");
                    reader.SetIsDone (true);
                }
            }
            else
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, failed to open master pty ");
                reader.SetIsDone (true);
            }
        }
        break;

    case eInputReaderDeactivate:
			// When another input reader is pushed, don't leave the session...
            //script_interpreter->LeaveSession ();
        break;

    case eInputReaderReactivate:
        {
            ScriptInterpreterPython::Locker locker(script_interpreter,
                                                   ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
                                                   ScriptInterpreterPython::Locker::FreeAcquiredLock);
        }
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderInterrupt:
        ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "raise KeyboardInterrupt\n", 24);
        break;
        
    case eInputReaderEndOfFile:
        ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "quit()\n", 7);
        break;

    case eInputReaderGotToken:
        if (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor() != -1)
        {
            if (log)
                log->Printf ("ScriptInterpreterPython::InputReaderCallback, GotToken, bytes='%s', byte_len = %lu", bytes,
                             bytes_len);
            if (bytes && bytes_len)
            {
                if ((int) bytes[0] == 4)
                    ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "quit()", 6);
                else
                    ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), bytes, bytes_len);
            }
            ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "\n", 1);
        }
        else
        {
            if (log)
                log->Printf ("ScriptInterpreterPython::InputReaderCallback, GotToken, bytes='%s', byte_len = %lu, Master File Descriptor is bad.", 
                             bytes,
                             bytes_len);
            reader.SetIsDone (true);
        }

        break;
        
    case eInputReaderDone:
        {
            Locker locker(script_interpreter,
                          ScriptInterpreterPython::Locker::AcquireLock,
                          ScriptInterpreterPython::Locker::FreeAcquiredLock);
            script_interpreter->LeaveSession ();
        }

        // Restore terminal settings if they were validly saved
        if (log)
            log->Printf ("ScriptInterpreterPython::InputReaderCallback, Done, closing down input reader.");
            
        script_interpreter->RestoreTerminalState ();

        script_interpreter->m_embedded_python_pty.CloseMasterFileDescriptor();
        break;
    }

    return bytes_len;
}


void
ScriptInterpreterPython::ExecuteInterpreterLoop ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    Debugger &debugger = GetCommandInterpreter().GetDebugger();

    // At the moment, the only time the debugger does not have an input file handle is when this is called
    // directly from Python, in which case it is both dangerous and unnecessary (not to mention confusing) to
    // try to embed a running interpreter loop inside the already running Python interpreter loop, so we won't
    // do it.

    if (!debugger.GetInputFile().IsValid())
        return;

    InputReaderSP reader_sp (new InputReader(debugger));
    if (reader_sp)
    {
        Error error (reader_sp->Initialize (ScriptInterpreterPython::InputReaderCallback,
                                            this,                         // baton
                                            eInputReaderGranularityLine,  // token size, to pass to callback function
                                            NULL,                         // end token
                                            NULL,                         // prompt
                                            true));                       // echo input
     
        if (error.Success())
        {
            debugger.PushInputReader (reader_sp);
            m_embedded_python_input_reader_sp = reader_sp;
        }
    }
}

bool
ScriptInterpreterPython::ExecuteOneLineWithReturn (const char *in_string,
                                                   ScriptInterpreter::ScriptReturnType return_type,
                                                   void *ret_value,
                                                   const ExecuteScriptOptions &options)
{

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitSession : 0),
                  ScriptInterpreterPython::Locker::FreeAcquiredLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::TearDownSession : 0));

    PyObject *py_return = NULL;
    PyObject *mainmod = PyImport_AddModule ("__main__");
    PyObject *globals = PyModule_GetDict (mainmod);
    PyObject *locals = NULL;
    PyObject *py_error = NULL;
    bool ret_success = false;
    bool should_decrement_locals = false;
    int success;
    
    locals = FindSessionDictionary(m_dictionary_name.c_str());
    
    if (locals == NULL)
    {
        locals = PyObject_GetAttrString (globals, m_dictionary_name.c_str());
        should_decrement_locals = true;
    }
        
    if (locals == NULL)
    {
        locals = globals;
        should_decrement_locals = false;
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
        PyErr_Clear();
    
    if (in_string != NULL)
    {
        { // scope for PythonInputReaderManager
            PythonInputReaderManager py_input(options.GetEnableIO() ? this : NULL);
            py_return = PyRun_String (in_string, Py_eval_input, globals, locals);
            if (py_return == NULL)
            { 
                py_error = PyErr_Occurred ();
                if (py_error != NULL)
                    PyErr_Clear ();

                py_return = PyRun_String (in_string, Py_single_input, globals, locals);
            }
        }

        if (locals != NULL
            && should_decrement_locals)
            Py_DECREF (locals);

        if (py_return != NULL)
        {
            switch (return_type)
            {
                case eScriptReturnTypeCharPtr: // "char *"
                {
                    const char format[3] = "s#";
                    success = PyArg_Parse (py_return, format, (char **) ret_value);
                    break;
                }
                case eScriptReturnTypeCharStrOrNone: // char* or NULL if py_return == Py_None
                {
                    const char format[3] = "z";
                    success = PyArg_Parse (py_return, format, (char **) ret_value);
                    break;
                }
                case eScriptReturnTypeBool:
                {
                    const char format[2] = "b";
                    success = PyArg_Parse (py_return, format, (bool *) ret_value);
                    break;
                }
                case eScriptReturnTypeShortInt:
                {
                    const char format[2] = "h";
                    success = PyArg_Parse (py_return, format, (short *) ret_value);
                    break;
                }
                case eScriptReturnTypeShortIntUnsigned:
                {
                    const char format[2] = "H";
                    success = PyArg_Parse (py_return, format, (unsigned short *) ret_value);
                    break;
                }
                case eScriptReturnTypeInt:
                {
                    const char format[2] = "i";
                    success = PyArg_Parse (py_return, format, (int *) ret_value);
                    break;
                }
                case eScriptReturnTypeIntUnsigned:
                {
                    const char format[2] = "I";
                    success = PyArg_Parse (py_return, format, (unsigned int *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongInt:
                {
                    const char format[2] = "l";
                    success = PyArg_Parse (py_return, format, (long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongIntUnsigned:
                {
                    const char format[2] = "k";
                    success = PyArg_Parse (py_return, format, (unsigned long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongLong:
                {
                    const char format[2] = "L";
                    success = PyArg_Parse (py_return, format, (long long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongLongUnsigned:
                {
                    const char format[2] = "K";
                    success = PyArg_Parse (py_return, format, (unsigned long long *) ret_value);
                    break;
                }
                case eScriptReturnTypeFloat:
                {
                    const char format[2] = "f";
                    success = PyArg_Parse (py_return, format, (float *) ret_value);
                    break;
                }
                case eScriptReturnTypeDouble:
                {
                    const char format[2] = "d";
                    success = PyArg_Parse (py_return, format, (double *) ret_value);
                    break;
                }
                case eScriptReturnTypeChar:
                {
                    const char format[2] = "c";
                    success = PyArg_Parse (py_return, format, (char *) ret_value);
                    break;
                }
                default:
                  {}
            }
            Py_DECREF (py_return);
            if (success)
                ret_success = true;
            else
                ret_success = false;
        }
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
    {
        ret_success = false;
        if (options.GetMaskoutErrors())
        {
            if (PyErr_GivenExceptionMatches (py_error, PyExc_SyntaxError))
                PyErr_Print ();
            PyErr_Clear();
        }
    }

    return ret_success;
}

bool
ScriptInterpreterPython::ExecuteMultipleLines (const char *in_string, const ExecuteScriptOptions &options)
{
    
    
    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitSession : 0),
                  ScriptInterpreterPython::Locker::FreeAcquiredLock | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::TearDownSession : 0));

    bool success = false;
    PyObject *py_return = NULL;
    PyObject *mainmod = PyImport_AddModule ("__main__");
    PyObject *globals = PyModule_GetDict (mainmod);
    PyObject *locals = NULL;
    PyObject *py_error = NULL;
    bool should_decrement_locals = false;

    locals = FindSessionDictionary(m_dictionary_name.c_str());
    
    if (locals == NULL)
    {
        locals = PyObject_GetAttrString (globals, m_dictionary_name.c_str());
        should_decrement_locals = true;
    }

    if (locals == NULL)
    {
        locals = globals;
        should_decrement_locals = false;
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
        PyErr_Clear();
    
    if (in_string != NULL)
    {
        struct _node *compiled_node = PyParser_SimpleParseString (in_string, Py_file_input);
        if (compiled_node)
        {
            PyCodeObject *compiled_code = PyNode_Compile (compiled_node, "temp.py");
            if (compiled_code)
            {
                { // scope for PythonInputReaderManager
                    PythonInputReaderManager py_input(options.GetEnableIO() ? this : NULL);
                    py_return = PyEval_EvalCode (compiled_code, globals, locals);
                }
                if (py_return != NULL)
                {
                    success = true;
                    Py_DECREF (py_return);
                }
                if (locals && should_decrement_locals)
                    Py_DECREF (locals);
            }
        }
    }

    py_error = PyErr_Occurred ();
    if (py_error != NULL)
    {
        success = false;
        if (options.GetMaskoutErrors())
        {
            if (PyErr_GivenExceptionMatches (py_error, PyExc_SyntaxError))
                PyErr_Print ();
            PyErr_Clear();
        }
    }

    return success;
}

static const char *g_reader_instructions = "Enter your Python command(s). Type 'DONE' to end.";

size_t
ScriptInterpreterPython::GenerateBreakpointOptionsCommandCallback
(
    void *baton, 
    InputReader &reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    static StringList commands_in_progress;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        {
            commands_in_progress.Clear();
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", g_reader_instructions);
                if (reader.GetPrompt())
                    out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush ();
        }
        break;

    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        {
            std::string temp_string (bytes, bytes_len);
            commands_in_progress.AppendString (temp_string.c_str());
            if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderEndOfFile:
    case eInputReaderInterrupt:
        // Control-c (SIGINT) & control-d both mean finish & exit.
        reader.SetIsDone(true);
        
        // Control-c (SIGINT) ALSO means cancel; do NOT create a breakpoint command.
        if (notification == eInputReaderInterrupt)
            commands_in_progress.Clear();  
        
        // Fall through here...

    case eInputReaderDone:
        {
            BreakpointOptions *bp_options = (BreakpointOptions *)baton;
            std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());
            data_ap->user_source.AppendList (commands_in_progress);
            if (data_ap.get())
            {
                ScriptInterpreter *interpreter = reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                if (interpreter)
                {
                    if (interpreter->GenerateBreakpointCommandCallbackData (data_ap->user_source, 
                                                                            data_ap->script_source))
                    {
                        BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
                        bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
                    }
                    else if (!batch_mode)
                    {
                        out_stream->Printf ("Warning: No command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
                else
                {
		            if (!batch_mode)
                    {
                        out_stream->Printf ("Warning:  Unable to find script intepreter; no command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
            }
        }
        break;
        
    }

    return bytes_len;
}

size_t
ScriptInterpreterPython::GenerateWatchpointOptionsCommandCallback
(
    void *baton, 
    InputReader &reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    static StringList commands_in_progress;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        {
            commands_in_progress.Clear();
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", g_reader_instructions);
                if (reader.GetPrompt())
                    out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush ();
        }
        break;

    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        {
            std::string temp_string (bytes, bytes_len);
            commands_in_progress.AppendString (temp_string.c_str());
            if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderEndOfFile:
    case eInputReaderInterrupt:
        // Control-c (SIGINT) & control-d both mean finish & exit.
        reader.SetIsDone(true);
        
        // Control-c (SIGINT) ALSO means cancel; do NOT create a breakpoint command.
        if (notification == eInputReaderInterrupt)
            commands_in_progress.Clear();  
        
        // Fall through here...

    case eInputReaderDone:
        {
            WatchpointOptions *wp_options = (WatchpointOptions *)baton;
            std::auto_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());
            data_ap->user_source.AppendList (commands_in_progress);
            if (data_ap.get())
            {
                ScriptInterpreter *interpreter = reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                if (interpreter)
                {
                    if (interpreter->GenerateWatchpointCommandCallbackData (data_ap->user_source, 
                                                                            data_ap->script_source))
                    {
                        BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
                        wp_options->SetCallback (ScriptInterpreterPython::WatchpointCallbackFunction, baton_sp);
                    }
                    else if (!batch_mode)
                    {
                        out_stream->Printf ("Warning: No command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
                else
                {
		            if (!batch_mode)
                    {
                        out_stream->Printf ("Warning:  Unable to find script intepreter; no command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
            }
        }
        break;
        
    }

    return bytes_len;
}

void
ScriptInterpreterPython::CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                                  CommandReturnObject &result)
{
    Debugger &debugger = GetCommandInterpreter().GetDebugger();
    
    InputReaderSP reader_sp (new InputReader (debugger));

    if (reader_sp)
    {
        Error err = reader_sp->Initialize (
                ScriptInterpreterPython::GenerateBreakpointOptionsCommandCallback,
                bp_options,                 // baton
                eInputReaderGranularityLine, // token size, for feeding data to callback function
                "DONE",                     // end token
                "> ",                       // prompt
                true);                      // echo input
    
        if (err.Success())
            debugger.PushInputReader (reader_sp);
        else
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError("out of memory");
        result.SetStatus (eReturnStatusFailed);
    }
}

void
ScriptInterpreterPython::CollectDataForWatchpointCommandCallback (WatchpointOptions *wp_options,
                                                                  CommandReturnObject &result)
{
    Debugger &debugger = GetCommandInterpreter().GetDebugger();
    
    InputReaderSP reader_sp (new InputReader (debugger));

    if (reader_sp)
    {
        Error err = reader_sp->Initialize (
                ScriptInterpreterPython::GenerateWatchpointOptionsCommandCallback,
                wp_options,                 // baton
                eInputReaderGranularityLine, // token size, for feeding data to callback function
                "DONE",                     // end token
                "> ",                       // prompt
                true);                      // echo input
    
        if (err.Success())
            debugger.PushInputReader (reader_sp);
        else
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError("out of memory");
        result.SetStatus (eReturnStatusFailed);
    }
}

// Set a Python one-liner as the callback for the breakpoint.
void
ScriptInterpreterPython::SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                       const char *oneliner)
{
    std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in breakpoint command list)
    // while the latter is used for Python to interpret during the actual callback.

    data_ap->user_source.AppendString (oneliner);
    data_ap->script_source.assign (oneliner);

    if (GenerateBreakpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
    {
        BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
        bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
    }
    
    return;
}

// Set a Python one-liner as the callback for the watchpoint.
void
ScriptInterpreterPython::SetWatchpointCommandCallback (WatchpointOptions *wp_options,
                                                       const char *oneliner)
{
    std::auto_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in watchpoint command list)
    // while the latter is used for Python to interpret during the actual callback.

    data_ap->user_source.AppendString (oneliner);
    data_ap->script_source.assign (oneliner);

    if (GenerateWatchpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
    {
        BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
        wp_options->SetCallback (ScriptInterpreterPython::WatchpointCallbackFunction, baton_sp);
    }
    
    return;
}

bool
ScriptInterpreterPython::ExportFunctionDefinitionToInterpreter (StringList &function_def)
{
    // Convert StringList to one long, newline delimited, const char *.
    std::string function_def_string(function_def.CopyList());

    return ExecuteMultipleLines (function_def_string.c_str(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false));
}

bool
ScriptInterpreterPython::GenerateFunction(const char *signature, const StringList &input)
{
    int num_lines = input.GetSize ();
    if (num_lines == 0)
        return false;
    
    if (!signature || *signature == 0)
        return false;

    StreamString sstr;
    StringList auto_generated_function;
    auto_generated_function.AppendString (signature);
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = internal_dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (internal_dict)"); // Add the session dictionary to the 
    // global dictionary.
    
    // Wrap everything up inside the function, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         internal_dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict
    
    // Verify that the results are valid Python.
    
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
        return false;
    
    return true;

}

bool
ScriptInterpreterPython::GenerateTypeScriptFunction (StringList &user_input, std::string& output, void* name_token)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Take what the user wrote, wrap it all up inside one big auto-generated Python function, passing in the
    // ValueObject as parameter to the function.
    
    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_type_print_func", num_created_functions, name_token));
    sstr.Printf ("def %s (valobj, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;

    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GenerateScriptAliasFunction (StringList &user_input, std::string &output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_cmd_alias_func", num_created_functions));

    sstr.Printf ("def %s (debugger, args, result, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(),user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}


bool
ScriptInterpreterPython::GenerateTypeSynthClass (StringList &user_input, std::string &output, void* name_token)
{
    static uint32_t num_created_classes = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Wrap all user input into a Python class
    
    std::string auto_generated_class_name(GenerateUniqueName("lldb_autogen_python_type_synth_class",num_created_classes,name_token));
    
    StringList auto_generated_class;
    
    // Create the function name & definition string.
    
    sstr.Printf ("class %s:", auto_generated_class_name.c_str());
    auto_generated_class.AppendString (sstr.GetData());
        
    // Wrap everything up inside the class, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", user_input.GetStringAtIndex (i));
        auto_generated_class.AppendString (sstr.GetData());
    }
    
    
    // Verify that the results are valid Python.
    // (even though the method is ExportFunctionDefinitionToInterpreter, a class will actually be exported)
    // (TODO: rename that method to ExportDefinitionToInterpreter)
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_class))
        return false;
    
    // Store the name of the auto-generated class
    
    output.assign(auto_generated_class_name);
    return true;
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::CreateOSPlugin (std::string class_name,
                lldb::ProcessSP process_sp)
{
    if (class_name.empty())
        return lldb::ScriptInterpreterObjectSP();
    
    if (!process_sp)
        return lldb::ScriptInterpreterObjectSP();
        
    void* ret_val;
    
    {
        Locker py_lock(this,Locker::AcquireLock,Locker::FreeLock);
        ret_val = g_swig_create_os_plugin    (class_name,
                                              m_dictionary_name.c_str(),
                                              process_sp);
    }
    
    return MakeScriptObject(ret_val);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_QueryForRegisterInfo (lldb::ScriptInterpreterObjectSP object)
{
    Locker py_lock(this,Locker::AcquireLock,Locker::FreeLock);
    
    static char callee_name[] = "get_register_info";
    
    if (!object)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)object->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_QueryForThreadsInfo (lldb::ScriptInterpreterObjectSP object)
{
    Locker py_lock(this,Locker::AcquireLock,Locker::FreeLock);

    static char callee_name[] = "get_thread_info";
    
    if (!object)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)object->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_QueryForRegisterContextData (lldb::ScriptInterpreterObjectSP object,
                                                               lldb::tid_t thread_id)
{
    Locker py_lock(this,Locker::AcquireLock,Locker::FreeLock);

    static char callee_name[] = "get_register_data";
    static char param_format[] = "l";
    
    if (!object)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)object->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();

    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, thread_id);

    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::CreateSyntheticScriptedProvider (std::string class_name,
                                                          lldb::ValueObjectSP valobj)
{
    if (class_name.empty())
        return lldb::ScriptInterpreterObjectSP();
    
    if (!valobj.get())
        return lldb::ScriptInterpreterObjectSP();
    
    ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return lldb::ScriptInterpreterObjectSP();
    
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return lldb::ScriptInterpreterObjectSP();
    
    void* ret_val;

    {
        Locker py_lock(this);
        ret_val = g_swig_synthetic_script (class_name,
                                           python_interpreter->m_dictionary_name.c_str(),
                                           valobj);
    }
    
    return MakeScriptObject(ret_val);
}

bool
ScriptInterpreterPython::GenerateTypeScriptFunction (const char* oneliner, std::string& output, void* name_token)
{
    StringList input;
    input.SplitIntoLines(oneliner, strlen(oneliner));
    return GenerateTypeScriptFunction(input, output, name_token);
}

bool
ScriptInterpreterPython::GenerateTypeSynthClass (const char* oneliner, std::string& output, void* name_token)
{
    StringList input;
    input.SplitIntoLines(oneliner, strlen(oneliner));
    return GenerateTypeSynthClass(input, output, name_token);
}


bool
ScriptInterpreterPython::GenerateBreakpointCommandCallbackData (StringList &user_input, std::string& output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;

    if (user_input.GetSize() == 0)
        return false;

    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_bp_callback_func_",num_created_functions));
    sstr.Printf ("def %s (frame, bp_loc, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GenerateWatchpointCommandCallbackData (StringList &user_input, std::string& output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;

    if (user_input.GetSize() == 0)
        return false;

    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_wp_callback_func_",num_created_functions));
    sstr.Printf ("def %s (frame, wp, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GetScriptedSummary (const char *python_function_name,
                                             lldb::ValueObjectSP valobj,
                                             lldb::ScriptInterpreterObjectSP& callee_wrapper_sp,
                                             std::string& retval)
{
    
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    
    if (!valobj.get())
    {
        retval.assign("<no object>");
        return false;
    }
        
    void* old_callee = (callee_wrapper_sp ? callee_wrapper_sp->GetObject() : NULL);
    void* new_callee = old_callee;
    
    bool ret_val;
    if (python_function_name 
        && *python_function_name)
    {
        {
            Locker py_lock(this);
            {
            Timer scoped_timer ("g_swig_typescript_callback","g_swig_typescript_callback");
            ret_val = g_swig_typescript_callback (python_function_name,
                                                  FindSessionDictionary(m_dictionary_name.c_str()),
                                                  valobj,
                                                  &new_callee,
                                                  retval);
            }
        }
    }
    else
    {
        retval.assign("<no function name>");
        return false;
    }
    
    if (new_callee && old_callee != new_callee)
        callee_wrapper_sp = MakeScriptObject(new_callee);
    
    return ret_val;
    
}

bool
ScriptInterpreterPython::BreakpointCallbackFunction 
(
    void *baton,
    StoppointCallbackContext *context,
    user_id_t break_id,
    user_id_t break_loc_id
)
{
    BreakpointOptions::CommandData *bp_option_data = (BreakpointOptions::CommandData *) baton;
    const char *python_function_name = bp_option_data->script_source.c_str();

    if (!context)
        return true;
        
    ExecutionContext exe_ctx (context->exe_ctx_ref);
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return true;
        
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return true;
    
    if (python_function_name != NULL 
        && python_function_name[0] != '\0')
    {
        const StackFrameSP stop_frame_sp (exe_ctx.GetFrameSP());
        BreakpointSP breakpoint_sp = target->GetBreakpointByID (break_id);
        if (breakpoint_sp)
        {
            const BreakpointLocationSP bp_loc_sp (breakpoint_sp->FindLocationByID (break_loc_id));

            if (stop_frame_sp && bp_loc_sp)
            {
                bool ret_val = true;
                {
                    Locker py_lock(python_interpreter);
                    ret_val = g_swig_breakpoint_callback (python_function_name, 
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
                                                          bp_loc_sp);
                }
                return ret_val;
            }
        }
    }
    // We currently always true so we stop in case anything goes wrong when
    // trying to call the script function
    return true;
}

bool
ScriptInterpreterPython::WatchpointCallbackFunction 
(
    void *baton,
    StoppointCallbackContext *context,
    user_id_t watch_id
)
{
    WatchpointOptions::CommandData *wp_option_data = (WatchpointOptions::CommandData *) baton;
    const char *python_function_name = wp_option_data->script_source.c_str();

    if (!context)
        return true;
        
    ExecutionContext exe_ctx (context->exe_ctx_ref);
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return true;
        
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return true;
    
    if (python_function_name != NULL 
        && python_function_name[0] != '\0')
    {
        const StackFrameSP stop_frame_sp (exe_ctx.GetFrameSP());
        WatchpointSP wp_sp = target->GetWatchpointList().FindByID (watch_id);
        if (wp_sp)
        {
            if (stop_frame_sp && wp_sp)
            {
                bool ret_val = true;
                {
                    Locker py_lock(python_interpreter);
                    ret_val = g_swig_watchpoint_callback (python_function_name, 
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
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

lldb::thread_result_t
ScriptInterpreterPython::RunEmbeddedPythonInterpreter (lldb::thread_arg_t baton)
{
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;
    
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));
    
    if (log)
        log->Printf ("%p ScriptInterpreterPython::RunEmbeddedPythonInterpreter () thread starting...", baton);
    
    char error_str[1024];
    const char *pty_slave_name = script_interpreter->m_embedded_python_pty.GetSlaveName (error_str, sizeof (error_str));

    if (pty_slave_name != NULL)
    {
        StreamString run_string;

        // Ensure we have the GIL before running any Python code.
        // Since we're only running a few one-liners and then dropping to the interpreter (which will release the GIL when needed),
        // we can just release the GIL after finishing our work.
        // If finer-grained locking is desirable, we can lock and unlock the GIL only when calling a python function.
        Locker locker(script_interpreter,
                      ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession,
                      ScriptInterpreterPython::Locker::FreeAcquiredLock | ScriptInterpreterPython::Locker::TearDownSession);

        run_string.Printf ("run_one_line (%s, 'save_stderr = sys.stderr')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, 'sys.stderr = sys.stdout')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, 'save_stdin = sys.stdin')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, \"sys.stdin = open ('%s', 'r')\")", script_interpreter->m_dictionary_name.c_str(),
                           pty_slave_name);
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();

        // The following call drops into the embedded interpreter loop and stays there until the
        // user chooses to exit from the Python interpreter.
        // This embedded interpreter will, as any Python code that performs I/O, unlock the GIL before
        // a system call that can hang, and lock it when the syscall has returned.

        // We need to surround the call to the embedded interpreter with calls to PyGILState_Ensure and 
        // PyGILState_Release (using the Locker above). This is because Python has a global lock which must be held whenever we want
        // to touch any Python objects. Otherwise, if the user calls Python code, the interpreter state will be off,
        // and things could hang (it's happened before).

        run_string.Printf ("run_python_interpreter (%s)", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();

        run_string.Printf ("run_one_line (%s, 'sys.stdin = save_stdin')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear();

        run_string.Printf ("run_one_line (%s, 'sys.stderr = save_stderr')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear();
    }
    
    if (script_interpreter->m_embedded_python_input_reader_sp)
        script_interpreter->m_embedded_python_input_reader_sp->SetIsDone (true);
    
    script_interpreter->m_embedded_python_pty.CloseSlaveFileDescriptor();

    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT);
    if (log)
        log->Printf ("%p ScriptInterpreterPython::RunEmbeddedPythonInterpreter () thread exiting...", baton);
    

    // Clean up the input reader and make the debugger pop it off the stack.
    Debugger &debugger = script_interpreter->GetCommandInterpreter().GetDebugger();
    const InputReaderSP reader_sp = script_interpreter->m_embedded_python_input_reader_sp;
    if (reader_sp)
    {
        debugger.PopInputReader (reader_sp);
        script_interpreter->m_embedded_python_input_reader_sp.reset();
    }
    
    return NULL;
}

lldb::thread_result_t
ScriptInterpreterPython::PythonInputReaderManager::RunPythonInputReader (lldb::thread_arg_t baton)
{
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;

    const InputReaderSP reader_sp = script_interpreter->m_embedded_thread_input_reader_sp;
    
    if (reader_sp)
        reader_sp->WaitOnReaderIsDone();
    
    return NULL;
}

uint32_t
ScriptInterpreterPython::CalculateNumChildren (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    if (!implementor_sp)
        return 0;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return 0;
    
    if (!g_swig_calc_children)
        return 0;

    uint32_t ret_val = 0;
    
    {
        Locker py_lock(this);
        ret_val = g_swig_calc_children (implementor);
    }
    
    return ret_val;
}

lldb::ValueObjectSP
ScriptInterpreterPython::GetChildAtIndex (const lldb::ScriptInterpreterObjectSP& implementor_sp, uint32_t idx)
{
    if (!implementor_sp)
        return lldb::ValueObjectSP();
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return lldb::ValueObjectSP();
    
    if (!g_swig_get_child_index || !g_swig_cast_to_sbvalue)
        return lldb::ValueObjectSP();
    
    void* child_ptr = NULL;
    lldb::SBValue* value_sb = NULL;
    lldb::ValueObjectSP ret_val;
    
    {
        Locker py_lock(this);
        child_ptr = g_swig_get_child_index (implementor,idx);
        if (child_ptr != NULL && child_ptr != Py_None)
        {
            value_sb = (lldb::SBValue*)g_swig_cast_to_sbvalue(child_ptr);
            if (value_sb == NULL)
                Py_XDECREF(child_ptr);
            else
                ret_val = value_sb->GetSP();
        }
        else
        {
            Py_XDECREF(child_ptr);
        }
    }
    
    return ret_val;
}

int
ScriptInterpreterPython::GetIndexOfChildWithName (const lldb::ScriptInterpreterObjectSP& implementor_sp, const char* child_name)
{
    if (!implementor_sp)
        return UINT32_MAX;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return UINT32_MAX;
    
    if (!g_swig_get_index_child)
        return UINT32_MAX;
    
    int ret_val = UINT32_MAX;
    
    {
        Locker py_lock(this);
        ret_val = g_swig_get_index_child (implementor, child_name);
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::UpdateSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    bool ret_val = false;
    
    if (!implementor_sp)
        return ret_val;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return ret_val;
    
    if (!g_swig_update_provider)
        return ret_val;
    
    {
        Locker py_lock(this);
        ret_val = g_swig_update_provider (implementor);
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::MightHaveChildrenSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    bool ret_val = false;
    
    if (!implementor_sp)
        return ret_val;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return ret_val;
    
    if (!g_swig_mighthavechildren_provider)
        return ret_val;
    
    {
        Locker py_lock(this);
        ret_val = g_swig_mighthavechildren_provider (implementor);
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::LoadScriptingModule (const char* pathname,
                                              bool can_reload,
                                              bool init_lldb_globals,
                                              lldb_private::Error& error)
{
    if (!pathname || !pathname[0])
    {
        error.SetErrorString("invalid pathname");
        return false;
    }
    
    if (!g_swig_call_module_init)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    
    lldb::DebuggerSP debugger_sp = m_interpreter.GetDebugger().shared_from_this();

    {
        FileSpec target_file(pathname, true);
        
        // TODO: would we want to reject any other value?
        if (target_file.GetFileType() == FileSpec::eFileTypeInvalid ||
            target_file.GetFileType() == FileSpec::eFileTypeUnknown)
        {
            error.SetErrorString("invalid pathname");
            return false;
        }
        
        const char* directory = target_file.GetDirectory().GetCString();
        std::string basename(target_file.GetFilename().GetCString());

        // Before executing Pyton code, lock the GIL.
        Locker py_lock (this,
                        Locker::AcquireLock      | (init_lldb_globals ? Locker::InitSession     : 0),
                        Locker::FreeAcquiredLock | (init_lldb_globals ? Locker::TearDownSession : 0));
        
        // now make sure that Python has "directory" in the search path
        StreamString command_stream;
        command_stream.Printf("if not (sys.path.__contains__('%s')):\n    sys.path.insert(1,'%s');\n\n",
                              directory,
                              directory);
        bool syspath_retval = ExecuteMultipleLines(command_stream.GetData(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false));
        if (!syspath_retval)
        {
            error.SetErrorString("Python sys.path handling failed");
            return false;
        }
        
        // strip .py or .pyc extension
        ConstString extension = target_file.GetFileNameExtension();
        if (::strcmp(extension.GetCString(), "py") == 0)
            basename.resize(basename.length()-3);
        else if(::strcmp(extension.GetCString(), "pyc") == 0)
            basename.resize(basename.length()-4);
        
        // check if the module is already import-ed
        command_stream.Clear();
        command_stream.Printf("sys.getrefcount(%s)",basename.c_str());
        int refcount = 0;
        // this call will fail if the module does not exist (because the parameter to it is not a string
        // but an actual Python module object, which is non-existant if the module was not imported before)
        bool was_imported = (ExecuteOneLineWithReturn(command_stream.GetData(),
                                                      ScriptInterpreterPython::eScriptReturnTypeInt,
                                                      &refcount,
                                                      ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false)) && refcount > 0);
        if (was_imported == true && can_reload == false)
        {
            error.SetErrorString("module already imported");
            return false;
        }

        // now actually do the import
        command_stream.Clear();
        command_stream.Printf("import %s",basename.c_str());
        bool import_retval = ExecuteMultipleLines(command_stream.GetData(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false).SetMaskoutErrors(false));
        PyObject* py_error = PyErr_Occurred(); // per Python docs: "you do not need to Py_DECREF()" the return of this function
        
        if (py_error || !import_retval) // check for failure of the import
        {
            if (py_error) // if we have a Python error..
            {
                if (PyErr_GivenExceptionMatches (py_error, PyExc_ImportError)) // and it is an ImportError
                {
                    PyObject *type,*value,*traceback;
                    PyErr_Fetch (&type,&value,&traceback);
                    
                    if (value && value != Py_None)
                        error.SetErrorString(PyString_AsString(PyObject_Str(value)));
                    else
                        error.SetErrorString("ImportError raised by imported module");
                    
                    Py_XDECREF(type);
                    Py_XDECREF(value);
                    Py_XDECREF(traceback);
                }
                else // any other error
                {
                    error.SetErrorString("Python raised an error while importing module");
                }
            }
            else // we failed but have no error to explain why
            {
                error.SetErrorString("unknown error while importing module");
            }
            
            // anyway, clear the error indicator and return false
            PyErr_Clear();
            return false;
        }
        
        // if we are here, everything worked
        // call __lldb_init_module(debugger,dict)
        if (!g_swig_call_module_init (basename,
                                        m_dictionary_name.c_str(),
                                        debugger_sp))
        {
            error.SetErrorString("calling __lldb_init_module failed");
            return false;
        }
        return true;
    }
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::MakeScriptObject (void* object)
{
    return lldb::ScriptInterpreterObjectSP(new ScriptInterpreterPythonObject(object));
}

ScriptInterpreterPython::SynchronicityHandler::SynchronicityHandler (lldb::DebuggerSP debugger_sp,
                                                                     ScriptedCommandSynchronicity synchro) :
    m_debugger_sp(debugger_sp),
    m_synch_wanted(synchro),
    m_old_asynch(debugger_sp->GetAsyncExecution())
{
    if (m_synch_wanted == eScriptedCommandSynchronicitySynchronous)
        m_debugger_sp->SetAsyncExecution(false);
    else if (m_synch_wanted == eScriptedCommandSynchronicityAsynchronous)
        m_debugger_sp->SetAsyncExecution(true);
}

ScriptInterpreterPython::SynchronicityHandler::~SynchronicityHandler()
{
    if (m_synch_wanted != eScriptedCommandSynchronicityCurrentValue)
        m_debugger_sp->SetAsyncExecution(m_old_asynch);
}

bool
ScriptInterpreterPython::RunScriptBasedCommand(const char* impl_function,
                                               const char* args,
                                               ScriptedCommandSynchronicity synchronicity,
                                               lldb_private::CommandReturnObject& cmd_retobj,
                                               Error& error)
{
    if (!impl_function)
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    
    if (!g_swig_call_command)
    {
        error.SetErrorString("no helper function to run scripted commands");
        return false;
    }
    
    lldb::DebuggerSP debugger_sp = m_interpreter.GetDebugger().shared_from_this();

    if (!debugger_sp.get())
    {
        error.SetErrorString("invalid Debugger pointer");
        return false;
    }
    
    bool ret_val;
    
    std::string err_msg;

    {
        Locker py_lock(this);
        SynchronicityHandler synch_handler(debugger_sp,
                                           synchronicity);
        
        PythonInputReaderManager py_input(this);
        
        ret_val = g_swig_call_command       (impl_function,
                                             m_dictionary_name.c_str(),
                                             debugger_sp,
                                             args,
                                             err_msg,
                                             cmd_retobj);
    }

    if (!ret_val)
        error.SetErrorString(err_msg.c_str());
    else
        error.Clear();
    
    return ret_val;
}

// in Python, a special attribute __doc__ contains the docstring
// for an object (function, method, class, ...) if any is defined
// Otherwise, the attribute's value is None
bool
ScriptInterpreterPython::GetDocumentationForItem(const char* item, std::string& dest)
{
	dest.clear();
	if (!item || !*item)
		return false;
    std::string command(item);
    command += ".__doc__";
    
    char* result_ptr = NULL; // Python is going to point this to valid data if ExecuteOneLineWithReturn returns successfully
    
    if (ExecuteOneLineWithReturn (command.c_str(),
                                 ScriptInterpreter::eScriptReturnTypeCharStrOrNone,
                                  &result_ptr, ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false) /*.SetSetLLDBGlobals(false)*/))
    {
        if (result_ptr)
            dest.assign(result_ptr);
        return true;
    }
    else
    {
        StreamString str_stream;
        str_stream.Printf("Function %s was not found. Containing module might be missing.",item);
        dest.assign(str_stream.GetData());
        return false;
    }
}

void
ScriptInterpreterPython::InitializeInterpreter (SWIGInitCallback python_swig_init_callback)
{
    g_swig_init_callback = python_swig_init_callback;
    g_swig_breakpoint_callback = LLDBSwigPythonBreakpointCallbackFunction;
    g_swig_watchpoint_callback = LLDBSwigPythonWatchpointCallbackFunction;
    g_swig_typescript_callback = LLDBSwigPythonCallTypeScript;
    g_swig_synthetic_script = LLDBSwigPythonCreateSyntheticProvider;
    g_swig_calc_children = LLDBSwigPython_CalculateNumChildren;
    g_swig_get_child_index = LLDBSwigPython_GetChildAtIndex;
    g_swig_get_index_child = LLDBSwigPython_GetIndexOfChildWithName;
    g_swig_cast_to_sbvalue = LLDBSWIGPython_CastPyObjectToSBValue;
    g_swig_update_provider = LLDBSwigPython_UpdateSynthProviderInstance;
    g_swig_mighthavechildren_provider = LLDBSwigPython_MightHaveChildrenSynthProviderInstance;
    g_swig_call_command = LLDBSwigPythonCallCommand;
    g_swig_call_module_init = LLDBSwigPythonCallModuleInit;
    g_swig_create_os_plugin = LLDBSWIGPythonCreateOSPlugin;
}

void
ScriptInterpreterPython::InitializePrivate ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    // Python will muck with STDIN terminal state, so save off any current TTY
    // settings so we can restore them.
    TerminalState stdin_tty_state;
    stdin_tty_state.Save(STDIN_FILENO, false);

    PyGILState_STATE gstate;
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    bool threads_already_initialized = false;
    if (PyEval_ThreadsInitialized ()) {
        gstate = PyGILState_Ensure ();
        if (log)
            log->Printf("Ensured PyGILState. Previous state = %slocked\n", gstate == PyGILState_UNLOCKED ? "un" : "");
        threads_already_initialized = true;
    } else {
        // InitThreads acquires the GIL if it hasn't been called before.
        PyEval_InitThreads ();
    }
    Py_InitializeEx (0);

    // Initialize SWIG after setting up python
    assert (g_swig_init_callback != NULL);
    g_swig_init_callback ();

    // Update the path python uses to search for modules to include the current directory.

    PyRun_SimpleString ("import sys");
    PyRun_SimpleString ("sys.path.append ('.')");

    // Find the module that owns this code and use that path we get to
    // set the sys.path appropriately.

    FileSpec file_spec;
    char python_dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypePythonDir, file_spec))
    {
        std::string python_path("sys.path.insert(0,\"");
        size_t orig_len = python_path.length();
        if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
        {
            python_path.append (python_dir_path);
            python_path.append ("\")");
            PyRun_SimpleString (python_path.c_str());
            python_path.resize (orig_len);
        }
        
        if (Host::GetLLDBPath (ePathTypeLLDBShlibDir, file_spec))
        {
            if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
            {
                python_path.append (python_dir_path);
                python_path.append ("\")");
                PyRun_SimpleString (python_path.c_str());
                python_path.resize (orig_len);
            }
        }
    }

    PyRun_SimpleString ("sys.dont_write_bytecode = 1; import lldb.embedded_interpreter; from lldb.embedded_interpreter import run_python_interpreter; from lldb.embedded_interpreter import run_one_line; from termios import *");

    if (threads_already_initialized) {
        if (log)
            log->Printf("Releasing PyGILState. Returning to state = %slocked\n", gstate == PyGILState_UNLOCKED ? "un" : "");
        PyGILState_Release (gstate);
    } else {
        // We initialized the threads in this function, just unlock the GIL.
        PyEval_SaveThread();
    }

    stdin_tty_state.Restore();
}

//void
//ScriptInterpreterPython::Terminate ()
//{
//    // We are intentionally NOT calling Py_Finalize here (this would be the logical place to call it).  Calling
//    // Py_Finalize here causes test suite runs to seg fault:  The test suite runs in Python.  It registers 
//    // SBDebugger::Terminate to be called 'at_exit'.  When the test suite Python harness finishes up, it calls 
//    // Py_Finalize, which calls all the 'at_exit' registered functions.  SBDebugger::Terminate calls Debugger::Terminate,
//    // which calls lldb::Terminate, which calls ScriptInterpreter::Terminate, which calls 
//    // ScriptInterpreterPython::Terminate.  So if we call Py_Finalize here, we end up with Py_Finalize being called from
//    // within Py_Finalize, which results in a seg fault.
//    //
//    // Since this function only gets called when lldb is shutting down and going away anyway, the fact that we don't
//    // actually call Py_Finalize should not cause any problems (everything should shut down/go away anyway when the
//    // process exits).
//    //
////    Py_Finalize ();
//}

#endif // #ifdef LLDB_DISABLE_PYTHON
