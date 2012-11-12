//===-- DynamicLoaderDarwinKernel.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderDarwinKernel_h_
#define liblldb_DynamicLoaderDarwinKernel_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

class DynamicLoaderDarwinKernel : public lldb_private::DynamicLoader
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::DynamicLoader *
    CreateInstance (lldb_private::Process *process, bool force);

    static void
    DebuggerInitialize (lldb_private::Debugger &debugger);

    DynamicLoaderDarwinKernel (lldb_private::Process *process);

    virtual
    ~DynamicLoaderDarwinKernel ();
    //------------------------------------------------------------------
    /// Called after attaching a process.
    ///
    /// Allow DynamicLoader plug-ins to execute some code after
    /// attaching to a process.
    //------------------------------------------------------------------
    virtual void
    DidAttach ();

    virtual void
    DidLaunch ();

    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (lldb_private::Thread &thread,
                                  bool stop_others);

    virtual lldb_private::Error
    CanLoadImage ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    void
    PrivateInitialize (lldb_private::Process *process);

    void
    PrivateProcessStateChanged (lldb_private::Process *process,
                                lldb::StateType state);
    
    void
    UpdateIfNeeded();

    void
    LoadKernelModuleIfNeeded ();

    void
    Clear (bool clear_process);

    void
    PutToLog (lldb_private::Log *log) const;

    static bool
    BreakpointHitCallback (void *baton,
                           lldb_private::StoppointCallbackContext *context,
                           lldb::user_id_t break_id,
                           lldb::user_id_t break_loc_id);

    bool
    BreakpointHit (lldb_private::StoppointCallbackContext *context, 
                   lldb::user_id_t break_id, 
                   lldb::user_id_t break_loc_id);
    uint32_t
    GetAddrByteSize()
    {
        return m_kernel.GetAddressByteSize();
    }

    static lldb::ByteOrder
    GetByteOrderFromMagic (uint32_t magic);

    enum
    {
        KERNEL_MODULE_MAX_NAME = 64u,
        // Versions less than 2 didn't have an entry size,
        // they had a 64 bit name, 16 byte UUID, 8 byte addr,
        // 8 byte size, 8 byte version, 4 byte load tag, and
        // 4 byte flags
        KERNEL_MODULE_ENTRY_SIZE_VERSION_1 = 64u + 16u + 8u + 8u + 8u + 4u + 4u
    };
    
    struct OSKextLoadedKextSummary
    {
        char                     name[KERNEL_MODULE_MAX_NAME];
        lldb::ModuleSP           module_sp;
        uint32_t                 load_process_stop_id;
        lldb_private::UUID       uuid;              // UUID for this dylib if it has one, else all zeros
        lldb_private::Address    so_address;        // The section offset address for this kext in case it can be read from object files
        uint64_t                 address;
        uint64_t                 size;
        uint64_t                 version;
        uint32_t                 load_tag;
        uint32_t                 flags;
        uint64_t                 reference_list;
        bool                     kernel_image;      // true if this is the kernel, false if this is a kext

        OSKextLoadedKextSummary() :
            module_sp (),
            load_process_stop_id (UINT32_MAX),
            uuid (),
            so_address (),
            address (LLDB_INVALID_ADDRESS),
            size (0),
            version (0),
            load_tag (0),
            flags (0),
            reference_list (0),
            kernel_image (false)
        {
            name[0] = '\0';
        }
        
        bool
        IsLoaded ()
        {
            return load_process_stop_id != UINT32_MAX;
        }

        void
        Clear (bool load_cmd_data_only)
        {
            if (!load_cmd_data_only)
            {
                so_address.Clear();
                address = LLDB_INVALID_ADDRESS;
                size = 0;
                version = 0;
                load_tag = 0;
                flags = 0;
                reference_list = 0;
                name[0] = '\0';
            }
            module_sp.reset();
            load_process_stop_id = UINT32_MAX;
        }

        bool
        LoadImageAtFileAddress (lldb_private::Process *process);

        bool
        LoadImageUsingMemoryModule (lldb_private::Process *process);
        
//        bool
//        operator == (const OSKextLoadedKextSummary& rhs) const
//        {
//            return  address == rhs.address
//                    && size == rhs.size
//            //&& module_sp.get() == rhs.module_sp.get()
//                    && uuid == rhs.uuid
//                    && version == rhs.version
//                    && load_tag == rhs.load_tag
//                    && flags == rhs.flags
//                    && reference_list == rhs.reference_list
//                    && strncmp (name, rhs.name, KERNEL_MODULE_MAX_NAME) == 0;
//        }
//
        bool
        UUIDValid() const
        {
            return uuid.IsValid();
        }

        uint32_t
        GetAddressByteSize ();

        lldb::ByteOrder
        GetByteOrder();

        lldb_private::ArchSpec
        GetArchitecture () const;

        void
        PutToLog (lldb_private::Log *log) const;

        typedef std::vector<OSKextLoadedKextSummary> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;
    };

    struct OSKextLoadedKextSummaryHeader
    {
        uint32_t version;
        uint32_t entry_size;
        uint32_t entry_count;
        lldb::addr_t image_infos_addr;

        OSKextLoadedKextSummaryHeader() :
            version (0),
            entry_size (0),
            entry_count (0),
            image_infos_addr (LLDB_INVALID_ADDRESS)
        {
        }

        uint32_t
        GetSize()
        {
            switch (version)
            {
                case 0: return 0;   // Can't know the size without a valid version
                case 1: return 8;   // Version 1 only had a version + entry_count
                default: break;
            }
            // Version 2 and above has version, entry_size, entry_count, and reserved
            return 16; 
        }

        void
        Clear()
        {
            version = 0;
            entry_size = 0;
            entry_count = 0;
            image_infos_addr = LLDB_INVALID_ADDRESS;
        }

        bool
        IsValid() const
        {
            return version >= 1 || version <= 2;
        }
    };

    void
    RegisterNotificationCallbacks();

    void
    UnregisterNotificationCallbacks();

    void
    SetNotificationBreakpointIfNeeded ();

    bool
    ReadAllKextSummaries ();

    bool
    ReadKextSummaryHeader ();
    
    bool
    ParseKextSummaries (const lldb_private::Address &kext_summary_addr, 
                        uint32_t count);
    
    bool
    AddModulesUsingImageInfos (OSKextLoadedKextSummary::collection &image_infos);
    
    void
    UpdateImageInfosHeaderAndLoadCommands(OSKextLoadedKextSummary::collection &image_infos, 
                                          uint32_t infos_count, 
                                          bool update_executable);

    uint32_t
    ReadKextSummaries (const lldb_private::Address &kext_summary_addr,
                       uint32_t image_infos_count, 
                       OSKextLoadedKextSummary::collection &image_infos);
    
    OSKextLoadedKextSummary m_kernel; // Info about the current kernel image being used
    lldb_private::Address m_kext_summary_header_ptr_addr;
    lldb_private::Address m_kext_summary_header_addr;
    OSKextLoadedKextSummaryHeader m_kext_summary_header;
    OSKextLoadedKextSummary::collection m_kext_summaries;
    mutable lldb_private::Mutex m_mutex;
    lldb::user_id_t m_break_id;

private:
    DISALLOW_COPY_AND_ASSIGN (DynamicLoaderDarwinKernel);
};

#endif  // liblldb_DynamicLoaderDarwinKernel_h_
