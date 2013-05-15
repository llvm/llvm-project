//===-- DynamicLoaderMacOSXDYLD.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderMacOSXDYLD_h_
#define liblldb_DynamicLoaderMacOSXDYLD_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

class DynamicLoaderMacOSXDYLD : public lldb_private::DynamicLoader
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::DynamicLoader *
    CreateInstance (lldb_private::Process *process, bool force);

    DynamicLoaderMacOSXDYLD (lldb_private::Process *process);

    virtual
    ~DynamicLoaderMacOSXDYLD ();
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

    virtual bool
    ProcessDidExec ();

    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (lldb_private::Thread &thread,
                                  bool stop_others);

    virtual size_t
    FindEquivalentSymbols (lldb_private::Symbol *original_symbol, 
                           lldb_private::ModuleList &module_list, 
                           lldb_private::SymbolContextList &equivalent_symbols);
    
    virtual lldb_private::Error
    CanLoadImage ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual bool
    AlwaysRelyOnEHUnwindInfo (lldb_private::SymbolContext &sym_ctx);

protected:
    void
    PrivateInitialize (lldb_private::Process *process);

    void
    PrivateProcessStateChanged (lldb_private::Process *process,
                                lldb::StateType state);
    bool
    LocateDYLD ();

    bool
    DidSetNotificationBreakpoint () const;

    void
    Clear (bool clear_process);

    void
    PutToLog (lldb_private::Log *log) const;

    bool
    ReadDYLDInfoFromMemoryAndSetNotificationCallback (lldb::addr_t addr);

    static bool
    NotifyBreakpointHit (void *baton,
                         lldb_private::StoppointCallbackContext *context,
                         lldb::user_id_t break_id,
                         lldb::user_id_t break_loc_id);

    uint32_t
    AddrByteSize();

    static lldb::ByteOrder
    GetByteOrderFromMagic (uint32_t magic);

    bool
    ReadMachHeader (lldb::addr_t addr,
                    llvm::MachO::mach_header *header,
                    lldb_private::DataExtractor *load_command_data);
    class Segment
    {
    public:

        Segment() :
            name(),
            vmaddr(LLDB_INVALID_ADDRESS),
            vmsize(0),
            fileoff(0),
            filesize(0),
            maxprot(0),
            initprot(0),
            nsects(0),
            flags(0)
        {
        }

        lldb_private::ConstString name;
        lldb::addr_t vmaddr;
        lldb::addr_t vmsize;
        lldb::addr_t fileoff;
        lldb::addr_t filesize;
        uint32_t maxprot;
        uint32_t initprot;
        uint32_t nsects;
        uint32_t flags;

        bool
        operator==(const Segment& rhs) const
        {
            return name == rhs.name && vmaddr == rhs.vmaddr && vmsize == rhs.vmsize;
        }

        void
        PutToLog (lldb_private::Log *log,
                  lldb::addr_t slide) const;

    };

    struct DYLDImageInfo
    {
        lldb::addr_t address;               // Address of mach header for this dylib
        lldb::addr_t slide;                 // The amount to slide all segments by if there is a global slide.
        lldb::addr_t mod_date;              // Modification date for this dylib
        lldb_private::FileSpec file_spec;   // Resolved path for this dylib
        lldb_private::UUID uuid;            // UUID for this dylib if it has one, else all zeros
        llvm::MachO::mach_header header;    // The mach header for this image
        std::vector<Segment> segments;      // All segment vmaddr and vmsize pairs for this executable (from memory of inferior)
        uint32_t load_stop_id;              // The process stop ID that the sections for this image were loadeded

        DYLDImageInfo() :
            address(LLDB_INVALID_ADDRESS),
            slide(0),
            mod_date(0),
            file_spec(),
            uuid(),
            header(),
            segments(),
            load_stop_id(0)
        {
        }

        void
        Clear(bool load_cmd_data_only)
        {
            if (!load_cmd_data_only)
            {
                address = LLDB_INVALID_ADDRESS;
                slide = 0;
                mod_date = 0;
                file_spec.Clear();
                ::memset (&header, 0, sizeof(header));
            }
            uuid.Clear();
            segments.clear();
            load_stop_id = 0;
        }

        bool
        operator == (const DYLDImageInfo& rhs) const
        {
            return  address == rhs.address
                && slide == rhs.slide
                && mod_date == rhs.mod_date
                && file_spec == rhs.file_spec
                && uuid == rhs.uuid
                && memcmp(&header, &rhs.header, sizeof(header)) == 0
                && segments == rhs.segments;
        }

        bool
        UUIDValid() const
        {
            return uuid.IsValid();
        }

        uint32_t
        GetAddressByteSize ()
        {
            if (header.cputype)
            {
                if (header.cputype & llvm::MachO::CPUArchABI64)
                    return 8;
                else
                    return 4;
            }
            return 0;
        }

        lldb::ByteOrder
        GetByteOrder();

        lldb_private::ArchSpec
        GetArchitecture () const
        {
            return lldb_private::ArchSpec (lldb_private::eArchTypeMachO, header.cputype, header.cpusubtype);
        }

        const Segment *
        FindSegment (const lldb_private::ConstString &name) const;

        void
        PutToLog (lldb_private::Log *log) const;

        typedef std::vector<DYLDImageInfo> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;
    };

    struct DYLDAllImageInfos
    {
        uint32_t version;
        uint32_t dylib_info_count;              // Version >= 1
        lldb::addr_t dylib_info_addr;           // Version >= 1
        lldb::addr_t notification;              // Version >= 1
        bool processDetachedFromSharedRegion;   // Version >= 1
        bool libSystemInitialized;              // Version >= 2
        lldb::addr_t dyldImageLoadAddress;      // Version >= 2

        DYLDAllImageInfos() :
            version (0),
            dylib_info_count (0),
            dylib_info_addr (LLDB_INVALID_ADDRESS),
            notification (LLDB_INVALID_ADDRESS),
            processDetachedFromSharedRegion (false),
            libSystemInitialized (false),
            dyldImageLoadAddress (LLDB_INVALID_ADDRESS)
        {
        }

        void
        Clear()
        {
            version = 0;
            dylib_info_count = 0;
            dylib_info_addr = LLDB_INVALID_ADDRESS;
            notification = LLDB_INVALID_ADDRESS;
            processDetachedFromSharedRegion = false;
            libSystemInitialized = false;
            dyldImageLoadAddress = LLDB_INVALID_ADDRESS;
        }

        bool
        IsValid() const
        {
            return version >= 1 || version <= 6;
        }
    };

    void
    RegisterNotificationCallbacks();

    void
    UnregisterNotificationCallbacks();

    uint32_t
    ParseLoadCommands (const lldb_private::DataExtractor& data,
                       DYLDImageInfo& dylib_info,
                       lldb_private::FileSpec *lc_id_dylinker);

    bool
    UpdateImageLoadAddress(lldb_private::Module *module,
                           DYLDImageInfo& info);

    bool
    UnloadImageLoadAddress (lldb_private::Module *module,
                            DYLDImageInfo& info);

    lldb::ModuleSP
    FindTargetModuleForDYLDImageInfo (DYLDImageInfo &image_info,
                                      bool can_create,
                                      bool *did_create_ptr);

    DYLDImageInfo *
    GetImageInfo (lldb_private::Module *module);

    bool
    NeedToLocateDYLD () const;

    bool
    SetNotificationBreakpoint ();

    // There is a little tricky bit where you might initially attach while dyld is updating
    // the all_image_infos, and you can't read the infos, so you have to continue and pick it
    // up when you hit the update breakpoint.  At that point, you need to run this initialize
    // function, but when you do it that way you DON'T need to do the extra work you would at
    // the breakpoint.
    // So this function will only do actual work if the image infos haven't been read yet.
    // If it does do any work, then it will return true, and false otherwise.  That way you can
    // call it in the breakpoint action, and if it returns true you're done.
    bool
    InitializeFromAllImageInfos ();

    bool
    ReadAllImageInfosStructure ();
    
    bool
    AddModulesUsingImageInfosAddress (lldb::addr_t image_infos_addr, uint32_t image_infos_count);
    
    bool
    AddModulesUsingImageInfos (DYLDImageInfo::collection &image_infos);
    
    bool
    RemoveModulesUsingImageInfosAddress (lldb::addr_t image_infos_addr, uint32_t image_infos_count);

    void
    UpdateImageInfosHeaderAndLoadCommands(DYLDImageInfo::collection &image_infos, 
                                          uint32_t infos_count, 
                                          bool update_executable);

    bool
    ReadImageInfos (lldb::addr_t image_infos_addr, 
                    uint32_t image_infos_count, 
                    DYLDImageInfo::collection &image_infos);


    DYLDImageInfo m_dyld;               // Info about the current dyld being used
    lldb::addr_t m_dyld_all_image_infos_addr;
    DYLDAllImageInfos m_dyld_all_image_infos;
    uint32_t m_dyld_all_image_infos_stop_id;
    lldb::user_id_t m_break_id;
    DYLDImageInfo::collection m_dyld_image_infos;   // Current shared libraries information
    uint32_t m_dyld_image_infos_stop_id;    // The process stop ID that "m_dyld_image_infos" is valid for
    mutable lldb_private::Mutex m_mutex;
    lldb_private::Process::Notifications m_notification_callbacks;
    bool m_process_image_addr_is_all_images_infos;

private:
    DISALLOW_COPY_AND_ASSIGN (DynamicLoaderMacOSXDYLD);
};

#endif  // liblldb_DynamicLoaderMacOSXDYLD_h_
