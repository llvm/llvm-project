//===-- ProcessMachCore.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMachCore_h_
#define liblldb_ProcessMachCore_h_

// C Includes

// C++ Includes
#include <list>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/Process.h"

class ThreadKDP;

class ProcessMachCore : public lldb_private::Process
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance (lldb_private::Target& target, 
                    lldb_private::Listener &listener, 
                    const lldb_private::FileSpec *crash_file_path);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::ConstString
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ProcessMachCore(lldb_private::Target& target, 
                    lldb_private::Listener &listener,
                    const lldb_private::FileSpec &core_file);
    
    virtual
    ~ProcessMachCore();
    
    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool
    CanDebug (lldb_private::Target &target,
              bool plugin_specified_by_name);
    
    //------------------------------------------------------------------
    // Creating a new process, or attaching to an existing one
    //------------------------------------------------------------------
    virtual lldb_private::Error
    DoLoadCore ();
    
    virtual lldb_private::DynamicLoader *
    GetDynamicLoader ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    //------------------------------------------------------------------
    // Process Control
    //------------------------------------------------------------------    
    virtual lldb_private::Error
    DoDestroy ();
    
    virtual void
    RefreshStateAfterStop();
    
    //------------------------------------------------------------------
    // Process Queries
    //------------------------------------------------------------------
    virtual bool
    IsAlive ();

    virtual bool
    WarnBeforeDetach () const;

    //------------------------------------------------------------------
    // Process Memory
    //------------------------------------------------------------------
    virtual size_t
    ReadMemory (lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error);
    
    virtual size_t
    DoReadMemory (lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error);
    
    virtual lldb::addr_t
    GetImageInfoAddress ();

protected:
    friend class ThreadMachCore;
    
    void
    Clear ( );
    
    virtual bool
    UpdateThreadList (lldb_private::ThreadList &old_thread_list, 
                      lldb_private::ThreadList &new_thread_list);
    
    lldb_private::ObjectFile *
    GetCoreObjectFile ();
private:
    bool 
    GetDynamicLoaderAddress (lldb::addr_t addr);

    //------------------------------------------------------------------
    // For ProcessMachCore only
    //------------------------------------------------------------------
    typedef lldb_private::Range<lldb::addr_t, lldb::addr_t> FileRange;
    typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, FileRange> VMRangeToFileOffset;

    VMRangeToFileOffset m_core_aranges;
    lldb::ModuleSP m_core_module_sp;
    lldb_private::FileSpec m_core_file;
    lldb::addr_t m_dyld_addr;
    lldb_private::ConstString m_dyld_plugin_name;
    DISALLOW_COPY_AND_ASSIGN (ProcessMachCore);
    
};

#endif  // liblldb_ProcessMachCore_h_
