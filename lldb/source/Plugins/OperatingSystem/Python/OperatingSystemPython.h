//===-- OperatingSystemPython.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_DISABLE_PYTHON

#ifndef liblldb_OperatingSystemPython_h_
#define liblldb_OperatingSystemPython_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/OperatingSystem.h"

class DynamicRegisterInfo;

class OperatingSystemPython : public lldb_private::OperatingSystem
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static lldb_private::OperatingSystem *
    CreateInstance (lldb_private::Process *process, bool force);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static const char *
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    //------------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------------
    OperatingSystemPython (lldb_private::Process *process,
                           const lldb_private::FileSpec &python_module_path);
    
    virtual
    ~OperatingSystemPython ();
    
    //------------------------------------------------------------------
    // lldb_private::PluginInterface Methods
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    //------------------------------------------------------------------
    // lldb_private::OperatingSystem Methods
    //------------------------------------------------------------------
    virtual bool
    UpdateThreadList (lldb_private::ThreadList &old_thread_list, 
                      lldb_private::ThreadList &new_thread_list);
    
    virtual void
    ThreadWasSelected (lldb_private::Thread *thread);

    virtual lldb::RegisterContextSP
    CreateRegisterContextForThread (lldb_private::Thread *thread,
                                    lldb::addr_t reg_data_addr);

    virtual lldb::StopInfoSP
    CreateThreadStopReason (lldb_private::Thread *thread);

protected:
    
    bool IsValid() const
    {
        return m_python_object != NULL;
    }
    DynamicRegisterInfo *
    GetDynamicRegisterInfo ();

    lldb::ValueObjectSP m_thread_list_valobj_sp;
    std::auto_ptr<DynamicRegisterInfo> m_register_info_ap;
    lldb_private::ScriptInterpreter *m_interpreter;
    void* m_python_object;
    
};

#endif // #ifndef liblldb_OperatingSystemPython_h_
#endif // #ifndef LLDB_DISABLE_PYTHON
