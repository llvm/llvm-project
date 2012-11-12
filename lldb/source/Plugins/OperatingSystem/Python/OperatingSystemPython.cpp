//===-- OperatingSystemPython.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_DISABLE_PYTHON

#include "OperatingSystemPython.h"
// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/Thread.h"
#include "Plugins/Process/Utility/DynamicRegisterInfo.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"
#include "Plugins/Process/Utility/ThreadMemory.h"

using namespace lldb;
using namespace lldb_private;

void
OperatingSystemPython::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
OperatingSystemPython::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

OperatingSystem *
OperatingSystemPython::CreateInstance (Process *process, bool force)
{
    // Python OperatingSystem plug-ins must be requested by name, so force must be true
    FileSpec python_os_plugin_spec (process->GetPythonOSPluginPath());
    if (python_os_plugin_spec && python_os_plugin_spec.Exists())
    {
        std::auto_ptr<OperatingSystemPython> os_ap (new OperatingSystemPython (process, python_os_plugin_spec));
        if (os_ap.get() && os_ap->IsValid())
            return os_ap.release();
    }
    return NULL;
}


const char *
OperatingSystemPython::GetPluginNameStatic()
{
    return "python";
}

const char *
OperatingSystemPython::GetPluginDescriptionStatic()
{
    return "Operating system plug-in that gathers OS information from a python class that implements the necessary OperatingSystem functionality.";
}


OperatingSystemPython::OperatingSystemPython (lldb_private::Process *process, const FileSpec &python_module_path) :
    OperatingSystem (process),
    m_thread_list_valobj_sp (),
    m_register_info_ap (),
    m_interpreter (NULL),
    m_python_object (NULL)
{
    if (!process)
        return;
    lldb::TargetSP target_sp = process->CalculateTarget();
    if (!target_sp)
        return;
    m_interpreter = target_sp->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    if (m_interpreter)
    {
        
        std::string os_plugin_class_name (python_module_path.GetFilename().AsCString(""));
        if (!os_plugin_class_name.empty())
        {
            const bool init_session = false;
            const bool allow_reload = true;
            char python_module_path_cstr[PATH_MAX];
            python_module_path.GetPath(python_module_path_cstr, sizeof(python_module_path_cstr));
            Error error;
            if (m_interpreter->LoadScriptingModule (python_module_path_cstr, allow_reload, init_session, error))
            {
                // Strip the ".py" extension if there is one
                size_t py_extension_pos = os_plugin_class_name.rfind(".py");
                if (py_extension_pos != std::string::npos)
                    os_plugin_class_name.erase (py_extension_pos);
                // Add ".OperatingSystemPlugIn" to the module name to get a string like "modulename.OperatingSystemPlugIn"
                os_plugin_class_name += ".OperatingSystemPlugIn";
                auto object_sp = m_interpreter->CreateOSPlugin(os_plugin_class_name.c_str(), process->CalculateProcess());
                if (object_sp)
                    m_python_object = object_sp->GetObject();
            }
        }
    }
}

OperatingSystemPython::~OperatingSystemPython ()
{
}

DynamicRegisterInfo *
OperatingSystemPython::GetDynamicRegisterInfo ()
{
    if (m_register_info_ap.get() == NULL)
    {
        if (!m_interpreter || !m_python_object)
            return NULL;
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
        
        if (log)
            log->Printf ("OperatingSystemPython::GetDynamicRegisterInfo() fetching thread register definitions from python for pid %llu", m_process->GetID());
        
        auto object_sp = m_interpreter->OSPlugin_QueryForRegisterInfo(m_interpreter->MakeScriptObject(m_python_object));
        if (!object_sp)
            return NULL;
        PythonDataObject dictionary_data_obj((PyObject*)object_sp->GetObject());
        PythonDataDictionary dictionary = dictionary_data_obj.GetDictionaryObject();
        if (!dictionary)
            return NULL;
        
        m_register_info_ap.reset (new DynamicRegisterInfo (dictionary));
        assert (m_register_info_ap->GetNumRegisters() > 0);
        assert (m_register_info_ap->GetNumRegisterSets() > 0);
    }
    return m_register_info_ap.get();
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
OperatingSystemPython::GetPluginName()
{
    return "OperatingSystemPython";
}

const char *
OperatingSystemPython::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
OperatingSystemPython::GetPluginVersion()
{
    return 1;
}

bool
OperatingSystemPython::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    if (!m_interpreter || !m_python_object)
        return NULL;
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    
    if (log)
        log->Printf ("OperatingSystemPython::UpdateThreadList() fetching thread data from python for pid %llu", m_process->GetID());

    auto object_sp = m_interpreter->OSPlugin_QueryForThreadsInfo(m_interpreter->MakeScriptObject(m_python_object));
    if (!object_sp)
        return NULL;
    PythonDataObject pyobj((PyObject*)object_sp->GetObject());
    PythonDataArray threads_array (pyobj.GetArrayObject());
    if (threads_array)
    {
//        const uint32_t num_old_threads = old_thread_list.GetSize(false);
//        for (uint32_t i=0; i<num_old_threads; ++i)
//        {
//            ThreadSP old_thread_sp(old_thread_list.GetThreadAtIndex(i, false));
//            if (old_thread_sp->GetID() < 0x10000)
//                new_thread_list.AddThread (old_thread_sp);
//        }

        PythonDataString tid_pystr("tid");
        PythonDataString name_pystr("name");
        PythonDataString queue_pystr("queue");
        PythonDataString state_pystr("state");
        PythonDataString stop_reason_pystr("stop_reason");
        PythonDataString reg_data_addr_pystr ("register_data_addr");
        
        const uint32_t num_threads = threads_array.GetSize();
        for (uint32_t i=0; i<num_threads; ++i)
        {
            PythonDataDictionary thread_dict(threads_array.GetItemAtIndex(i).GetDictionaryObject());
            if (thread_dict)
            {
                const tid_t tid = thread_dict.GetItemForKeyAsInteger (tid_pystr, LLDB_INVALID_THREAD_ID);
                const addr_t reg_data_addr = thread_dict.GetItemForKeyAsInteger (reg_data_addr_pystr, LLDB_INVALID_ADDRESS);
                const char *name = thread_dict.GetItemForKeyAsString (name_pystr);
                const char *queue = thread_dict.GetItemForKeyAsString (queue_pystr);
                //const char *state = thread_dict.GetItemForKeyAsString (state_pystr);
                //const char *stop_reason = thread_dict.GetItemForKeyAsString (stop_reason_pystr);
                
                ThreadSP thread_sp (old_thread_list.FindThreadByID (tid, false));
                if (!thread_sp)
                    thread_sp.reset (new ThreadMemory (*m_process,
                                                       tid,
                                                       name,
                                                       queue,
                                                       reg_data_addr));
                new_thread_list.AddThread(thread_sp);

            }
        }
    }
    else
    {
        new_thread_list = old_thread_list;
    }
    return new_thread_list.GetSize(false) > 0;
}

void
OperatingSystemPython::ThreadWasSelected (Thread *thread)
{
}

RegisterContextSP
OperatingSystemPython::CreateRegisterContextForThread (Thread *thread, lldb::addr_t reg_data_addr)
{
    RegisterContextSP reg_ctx_sp;
    if (!m_interpreter || !m_python_object || !thread)
        return RegisterContextSP();
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));

    if (reg_data_addr != LLDB_INVALID_ADDRESS)
    {
        // The registers data is in contiguous memory, just create the register
        // context using the address provided
        if (log)
            log->Printf ("OperatingSystemPython::CreateRegisterContextForThread (tid = 0x%llx, reg_data_addr = 0x%llx) creating memory register context", thread->GetID(), reg_data_addr);
        reg_ctx_sp.reset (new RegisterContextMemory (*thread, 0, *GetDynamicRegisterInfo (), reg_data_addr));
    }
    else
    {
        // No register data address is provided, query the python plug-in to let
        // it make up the data as it sees fit
        if (log)
            log->Printf ("OperatingSystemPython::CreateRegisterContextForThread (tid = 0x%llx) fetching register data from python", thread->GetID());

        auto object_sp = m_interpreter->OSPlugin_QueryForRegisterContextData (m_interpreter->MakeScriptObject(m_python_object),
                                                                              thread->GetID());
        
        if (!object_sp)
            return RegisterContextSP();
        
        PythonDataString reg_context_data((PyObject*)object_sp->GetObject());
        if (reg_context_data)
        {
            DataBufferSP data_sp (new DataBufferHeap (reg_context_data.GetString(),
                                                      reg_context_data.GetSize()));
            if (data_sp->GetByteSize())
            {
                RegisterContextMemory *reg_ctx_memory = new RegisterContextMemory (*thread, 0, *GetDynamicRegisterInfo (), LLDB_INVALID_ADDRESS);
                if (reg_ctx_memory)
                {
                    reg_ctx_sp.reset(reg_ctx_memory);
                    reg_ctx_memory->SetAllRegisterData (data_sp);
                }
            }
        }
    }
    return reg_ctx_sp;
}

StopInfoSP
OperatingSystemPython::CreateThreadStopReason (lldb_private::Thread *thread)
{
    // We should have gotten the thread stop info from the dictionary of data for
    // the thread in the initial call to get_thread_info(), this should have been
    // cached so we can return it here
    StopInfoSP stop_info_sp; //(StopInfo::CreateStopReasonWithSignal (*thread, SIGSTOP));
    return stop_info_sp;
}


#endif // #ifndef LLDB_DISABLE_PYTHON
