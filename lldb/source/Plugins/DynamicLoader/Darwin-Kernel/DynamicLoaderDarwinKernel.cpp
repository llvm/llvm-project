//===-- DynamicLoaderDarwinKernel.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"


#include "DynamicLoaderDarwinKernel.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

using namespace lldb;
using namespace lldb_private;

static PropertyDefinition
g_properties[] =
{
    { "load-kexts" , OptionValue::eTypeBoolean, true, true, NULL, NULL, "Automatically loads kext images when attaching to a kernel." },
    {  NULL        , OptionValue::eTypeInvalid, false, 0  , NULL, NULL, NULL  }
};

enum {
    ePropertyLoadKexts
};

class DynamicLoaderDarwinKernelProperties : public Properties
{
public:
    
    static ConstString &
    GetSettingName ()
    {
        static ConstString g_setting_name("darwin-kernel");
        return g_setting_name;
    }

    DynamicLoaderDarwinKernelProperties() :
        Properties ()
    {
        m_collection_sp.reset (new OptionValueProperties(GetSettingName()));
        m_collection_sp->Initialize(g_properties);
    }

    virtual
    ~DynamicLoaderDarwinKernelProperties()
    {
    }
    
    bool
    GetLoadKexts() const
    {
        const uint32_t idx = ePropertyLoadKexts;
        return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
    }
    
};

typedef STD_SHARED_PTR(DynamicLoaderDarwinKernelProperties) DynamicLoaderDarwinKernelPropertiesSP;

static const DynamicLoaderDarwinKernelPropertiesSP &
GetGlobalProperties()
{
    static DynamicLoaderDarwinKernelPropertiesSP g_settings_sp;
    if (!g_settings_sp)
        g_settings_sp.reset (new DynamicLoaderDarwinKernelProperties ());
    return g_settings_sp;
}

//----------------------------------------------------------------------
// Create an instance of this class. This function is filled into
// the plugin info class that gets handed out by the plugin factory and
// allows the lldb to instantiate an instance of this class.
//----------------------------------------------------------------------
DynamicLoader *
DynamicLoaderDarwinKernel::CreateInstance (Process* process, bool force)
{
    bool create = force;
    if (!create)
    {
        Module* exe_module = process->GetTarget().GetExecutableModulePointer();
        if (exe_module)
        {
            ObjectFile *object_file = exe_module->GetObjectFile();
            if (object_file)
            {
                create = (object_file->GetStrata() == ObjectFile::eStrataKernel);
            }
        }
        
        if (create)
        {
            const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
            switch (triple_ref.getOS())
            {
                case llvm::Triple::Darwin:
                case llvm::Triple::MacOSX:
                case llvm::Triple::IOS:
                    create = triple_ref.getVendor() == llvm::Triple::Apple;
                    break;
                default:
                    create = false;
                    break;
            }
        }
    }
    
    if (create)
    {
        process->SetCanJIT(false);
        return new DynamicLoaderDarwinKernel (process);
    }
    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DynamicLoaderDarwinKernel::DynamicLoaderDarwinKernel (Process* process) :
    DynamicLoader(process),
    m_kernel(),
    m_kext_summary_header_ptr_addr (),
    m_kext_summary_header_addr (),
    m_kext_summary_header (),
    m_kext_summaries(),
    m_mutex(Mutex::eMutexTypeRecursive),
    m_break_id (LLDB_INVALID_BREAK_ID)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DynamicLoaderDarwinKernel::~DynamicLoaderDarwinKernel()
{
    Clear(true);
}

void
DynamicLoaderDarwinKernel::UpdateIfNeeded()
{
    LoadKernelModuleIfNeeded();
    SetNotificationBreakpointIfNeeded ();
}
//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::DidAttach ()
{
    PrivateInitialize(m_process);
    UpdateIfNeeded();
}

//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::DidLaunch ()
{
    PrivateInitialize(m_process);
    UpdateIfNeeded();
}


//----------------------------------------------------------------------
// Clear out the state of this class.
//----------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::Clear (bool clear_process)
{
    Mutex::Locker locker(m_mutex);

    if (m_process->IsAlive() && LLDB_BREAK_ID_IS_VALID(m_break_id))
        m_process->ClearBreakpointSiteByID(m_break_id);

    if (clear_process)
        m_process = NULL;
    m_kernel.Clear(false);
    m_kext_summary_header_ptr_addr.Clear();
    m_kext_summary_header_addr.Clear();
    m_kext_summaries.clear();
    m_break_id = LLDB_INVALID_BREAK_ID;
}


bool
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::LoadImageAtFileAddress (Process *process)
{
    if (IsLoaded())
        return true;

    if (module_sp)
    {
        bool changed = false;
        if (module_sp->SetLoadAddress (process->GetTarget(), 0, changed))
            load_process_stop_id = process->GetStopID();
    }
    return false;
}

bool
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::LoadImageUsingMemoryModule (Process *process)
{
    if (IsLoaded())
        return true;

    bool uuid_is_valid = uuid.IsValid();
    bool memory_module_is_kernel = false;

    Target &target = process->GetTarget();
    ModuleSP memory_module_sp;

    // If this is a kext and the user asked us to ignore kexts, don't try to load it.
    if (kernel_image == false && GetGlobalProperties()->GetLoadKexts() == false)
    {
        return false;
    }

    // Use the memory module as the module if we have one
    if (address != LLDB_INVALID_ADDRESS)
    {
        FileSpec file_spec;
        if (module_sp)
            file_spec = module_sp->GetFileSpec();
        else
            file_spec.SetFile (name, false);
        
        memory_module_sp = process->ReadModuleFromMemory (file_spec, address, false, false);
        if (memory_module_sp && !uuid_is_valid)
        {
            uuid = memory_module_sp->GetUUID();
            uuid_is_valid = uuid.IsValid();
        }
        if (memory_module_sp 
            && memory_module_sp->GetObjectFile() 
            && memory_module_sp->GetObjectFile()->GetType() == ObjectFile::eTypeExecutable
            && memory_module_sp->GetObjectFile()->GetStrata() == ObjectFile::eStrataKernel)
        {
            memory_module_is_kernel = true;
            if (memory_module_sp->GetArchitecture().IsValid())
            {
                target.SetArchitecture(memory_module_sp->GetArchitecture());
            }
        }
    }

    if (!module_sp)
    {
        if (uuid_is_valid)
        {
            const ModuleList &target_images = target.GetImages();
            module_sp = target_images.FindModule(uuid);

            if (!module_sp)
            {
                ModuleSpec module_spec;
                module_spec.GetUUID() = uuid;
                module_spec.GetArchitecture() = target.GetArchitecture();

                // For the kernel, we really do need an on-disk file copy of the
                // binary.
                bool force_symbols_search = false;
                if (memory_module_is_kernel)
                {
                    force_symbols_search = true;
                }

                if (Symbols::DownloadObjectAndSymbolFile (module_spec, force_symbols_search))
                {
                    if (module_spec.GetFileSpec().Exists())
                    {
                        module_sp.reset(new Module (module_spec.GetFileSpec(), target.GetArchitecture()));
                        if (module_sp.get() && module_sp->MatchesModuleSpec (module_spec))
                        {
                            ModuleList loaded_module_list;
                            loaded_module_list.Append (module_sp);
                            target.ModulesDidLoad (loaded_module_list);
                        }
                    }
                }
            
                // Ask the Target to find this file on the local system, if possible.
                // This will search in the list of currently-loaded files, look in the 
                // standard search paths on the system, and on a Mac it will try calling
                // the DebugSymbols framework with the UUID to find the binary via its
                // search methods.
                if (!module_sp)
                {
                    module_sp = target.GetSharedModule (module_spec);
                }
            }
        }
    }
    

    if (memory_module_sp && module_sp)
    {
        if (module_sp->GetUUID() == memory_module_sp->GetUUID())
        {
            target.GetImages().Append(module_sp);
            if (memory_module_is_kernel && target.GetExecutableModulePointer() != module_sp.get())
            {
                target.SetExecutableModule (module_sp, false);
            }

            ObjectFile *ondisk_object_file = module_sp->GetObjectFile();
            ObjectFile *memory_object_file = memory_module_sp->GetObjectFile();
            if (memory_object_file && ondisk_object_file)
            {
                SectionList *ondisk_section_list = ondisk_object_file->GetSectionList ();
                SectionList *memory_section_list = memory_object_file->GetSectionList ();
                if (memory_section_list && ondisk_section_list)
                {
                    const uint32_t num_ondisk_sections = ondisk_section_list->GetSize();
                    // There may be CTF sections in the memory image so we can't
                    // always just compare the number of sections (which are actually
                    // segments in mach-o parlance)
                    uint32_t sect_idx = 0;
                    
                    // Use the memory_module's addresses for each section to set the 
                    // file module's load address as appropriate.  We don't want to use
                    // a single slide value for the entire kext - different segments may
                    // be slid different amounts by the kext loader.

                    uint32_t num_sections_loaded = 0;
                    for (sect_idx=0; sect_idx<num_ondisk_sections; ++sect_idx)
                    {
                        SectionSP ondisk_section_sp(ondisk_section_list->GetSectionAtIndex(sect_idx));
                        if (ondisk_section_sp)
                        {
                            const Section *memory_section = memory_section_list->FindSectionByName(ondisk_section_sp->GetName()).get();
                            if (memory_section)
                            {
                                target.GetSectionLoadList().SetSectionLoadAddress (ondisk_section_sp, memory_section->GetFileAddress());
                                ++num_sections_loaded;
                            }
                        }
                    }
                    if (num_sections_loaded > 0)
                        load_process_stop_id = process->GetStopID();
                    else
                        module_sp.reset(); // No sections were loaded
                }
                else
                    module_sp.reset(); // One or both section lists
            }
            else
                module_sp.reset(); // One or both object files missing
        }
        else
            module_sp.reset(); // UUID mismatch
    }

    bool is_loaded = IsLoaded();
    
    if (so_address.IsValid())
    {
        if (is_loaded)
            so_address.SetLoadAddress (address, &target);
        else
            target.GetImages().ResolveFileAddress (address, so_address);

    }

    if (is_loaded && module_sp && memory_module_is_kernel)
    {
        Stream *s = &target.GetDebugger().GetOutputStream();
        if (s)
        {
            char uuidbuf[64];
            s->Printf ("Kernel UUID: %s\n", module_sp->GetUUID().GetAsCString(uuidbuf, sizeof (uuidbuf)));
            s->Printf ("Load Address: 0x%llx\n", address);
            if (module_sp->GetFileSpec().GetDirectory().IsEmpty())
            {
                s->Printf ("Loaded kernel file %s\n", module_sp->GetFileSpec().GetFilename().AsCString());
            }
            else
            {
                s->Printf ("Loaded kernel file %s/%s\n",
                              module_sp->GetFileSpec().GetDirectory().AsCString(),
                              module_sp->GetFileSpec().GetFilename().AsCString());
            }
            s->Flush ();
        }
    }
    return is_loaded;
}

uint32_t
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::GetAddressByteSize ()
{
    if (module_sp)
        return module_sp->GetArchitecture().GetAddressByteSize();
    return 0;
}

lldb::ByteOrder
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::GetByteOrder()
{
    if (module_sp)
        return module_sp->GetArchitecture().GetByteOrder();
    return lldb::endian::InlHostByteOrder();
}

lldb_private::ArchSpec
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::GetArchitecture () const
{
    if (module_sp)
        return module_sp->GetArchitecture();
    return lldb_private::ArchSpec ();
}


//----------------------------------------------------------------------
// Load the kernel module and initialize the "m_kernel" member. Return
// true _only_ if the kernel is loaded the first time through (subsequent
// calls to this function should return false after the kernel has been
// already loaded).
//----------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::LoadKernelModuleIfNeeded()
{
    if (!m_kext_summary_header_ptr_addr.IsValid())
    {
        m_kernel.Clear(false);
        m_kernel.module_sp = m_process->GetTarget().GetExecutableModule();
        m_kernel.kernel_image = true;

        ConstString kernel_name("mach_kernel");
        if (m_kernel.module_sp.get() 
            && m_kernel.module_sp->GetObjectFile()
            && !m_kernel.module_sp->GetObjectFile()->GetFileSpec().GetFilename().IsEmpty())
        {
            kernel_name = m_kernel.module_sp->GetObjectFile()->GetFileSpec().GetFilename();
        }
        strncpy (m_kernel.name, kernel_name.AsCString(), sizeof(m_kernel.name));
        m_kernel.name[sizeof (m_kernel.name) - 1] = '\0';

        if (m_kernel.address == LLDB_INVALID_ADDRESS)
        {
            m_kernel.address = m_process->GetImageInfoAddress ();
            if (m_kernel.address == LLDB_INVALID_ADDRESS && m_kernel.module_sp)
            {
                // We didn't get a hint from the process, so we will
                // try the kernel at the address that it exists at in
                // the file if we have one
                ObjectFile *kernel_object_file = m_kernel.module_sp->GetObjectFile();
                if (kernel_object_file)
                {
                    addr_t load_address = kernel_object_file->GetHeaderAddress().GetLoadAddress(&m_process->GetTarget());
                    addr_t file_address = kernel_object_file->GetHeaderAddress().GetFileAddress();
                    if (load_address != LLDB_INVALID_ADDRESS && load_address != 0)
                    {
                        m_kernel.address = load_address;
                        if (load_address != file_address)
                        {
                            // Don't accidentally relocate the kernel to the File address -- 
                            // the Load address has already been set to its actual in-memory address.  
                            // Mark it as IsLoaded.
                            m_kernel.load_process_stop_id = m_process->GetStopID();
                        }
                    }
                    else
                    {
                        m_kernel.address = file_address;
                    }
                }
            }
        }
        
        if (m_kernel.address != LLDB_INVALID_ADDRESS)
        {
            if (!m_kernel.LoadImageUsingMemoryModule (m_process))
            {
                m_kernel.LoadImageAtFileAddress (m_process);
            }
        }

        if (m_kernel.IsLoaded() && m_kernel.module_sp)
        {
            static ConstString kext_summary_symbol ("gLoadedKextSummaries");
            const Symbol *symbol = m_kernel.module_sp->FindFirstSymbolWithNameAndType (kext_summary_symbol, eSymbolTypeData);
            if (symbol)
            {
                m_kext_summary_header_ptr_addr = symbol->GetAddress();
                // Update all image infos
                ReadAllKextSummaries ();
            }
        }
        else
        {
            m_kernel.Clear(false);
        }
    }
}

//----------------------------------------------------------------------
// Static callback function that gets called when our DYLD notification
// breakpoint gets hit. We update all of our image infos and then
// let our super class DynamicLoader class decide if we should stop
// or not (based on global preference).
//----------------------------------------------------------------------
bool
DynamicLoaderDarwinKernel::BreakpointHitCallback (void *baton, 
                                                  StoppointCallbackContext *context, 
                                                  user_id_t break_id, 
                                                  user_id_t break_loc_id)
{    
    return static_cast<DynamicLoaderDarwinKernel*>(baton)->BreakpointHit (context, break_id, break_loc_id);    
}

bool
DynamicLoaderDarwinKernel::BreakpointHit (StoppointCallbackContext *context, 
                                          user_id_t break_id, 
                                          user_id_t break_loc_id)
{    
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf ("DynamicLoaderDarwinKernel::BreakpointHit (...)\n");

    ReadAllKextSummaries ();
    
    if (log)
        PutToLog(log.get());

    return GetStopWhenImagesChange();
}


bool
DynamicLoaderDarwinKernel::ReadKextSummaryHeader ()
{
    Mutex::Locker locker(m_mutex);

    // the all image infos is already valid for this process stop ID

    m_kext_summaries.clear();
    if (m_kext_summary_header_ptr_addr.IsValid())
    {
        const uint32_t addr_size = m_kernel.GetAddressByteSize ();
        const ByteOrder byte_order = m_kernel.GetByteOrder();
        Error error;
        // Read enough bytes for a "OSKextLoadedKextSummaryHeader" structure
        // which is currenty 4 uint32_t and a pointer.
        uint8_t buf[24];
        DataExtractor data (buf, sizeof(buf), byte_order, addr_size);
        const size_t count = 4 * sizeof(uint32_t) + addr_size;
        const bool prefer_file_cache = false;
        if (m_process->GetTarget().ReadPointerFromMemory (m_kext_summary_header_ptr_addr, 
                                                          prefer_file_cache,
                                                          error,
                                                          m_kext_summary_header_addr))
        {
            // We got a valid address for our kext summary header and make sure it isn't NULL
            if (m_kext_summary_header_addr.IsValid() && 
                m_kext_summary_header_addr.GetFileAddress() != 0)
            {
                const size_t bytes_read = m_process->GetTarget().ReadMemory (m_kext_summary_header_addr, prefer_file_cache, buf, count, error);
                if (bytes_read == count)
                {
                    uint32_t offset = 0;
                    m_kext_summary_header.version = data.GetU32(&offset);
                    if (m_kext_summary_header.version >= 2)
                    {
                        m_kext_summary_header.entry_size = data.GetU32(&offset);
                    }
                    else
                    {
                        // Versions less than 2 didn't have an entry size, it was hard coded
                        m_kext_summary_header.entry_size = KERNEL_MODULE_ENTRY_SIZE_VERSION_1;
                    }
                    m_kext_summary_header.entry_count = data.GetU32(&offset);
                    return true;
                }
            }
        }
    }
    m_kext_summary_header_addr.Clear();
    return false;
}


bool
DynamicLoaderDarwinKernel::ParseKextSummaries (const Address &kext_summary_addr, 
                                               uint32_t count)
{
    OSKextLoadedKextSummary::collection kext_summaries;
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf ("Adding %d modules.\n", count);
        
    Mutex::Locker locker(m_mutex);

    if (!ReadKextSummaries (kext_summary_addr, count, kext_summaries))
        return false;

    Stream *s = &m_process->GetTarget().GetDebugger().GetOutputStream();
    if (s)
        s->Printf ("Loading %d kext modules ", count);
    for (uint32_t i = 0; i < count; i++)
    {
        if (!kext_summaries[i].LoadImageUsingMemoryModule (m_process))
            kext_summaries[i].LoadImageAtFileAddress (m_process);

        if (s)
            s->Printf (".");

        if (log)
            kext_summaries[i].PutToLog (log.get());
    }
    if (s)
    {
        s->Printf (" done.\n");
        s->Flush ();
    }

    bool return_value = AddModulesUsingImageInfos (kext_summaries);
    return return_value;
}

// Adds the modules in image_infos to m_kext_summaries.  
// NB don't call this passing in m_kext_summaries.

bool
DynamicLoaderDarwinKernel::AddModulesUsingImageInfos (OSKextLoadedKextSummary::collection &image_infos)
{
    // Now add these images to the main list.
    ModuleList loaded_module_list;
    
    for (uint32_t idx = 0; idx < image_infos.size(); ++idx)
    {
        OSKextLoadedKextSummary &image_info = image_infos[idx];
        m_kext_summaries.push_back(image_info);
        
        if (image_info.module_sp && m_process->GetStopID() == image_info.load_process_stop_id)
            loaded_module_list.AppendIfNeeded (image_infos[idx].module_sp);
    }
    
    m_process->GetTarget().ModulesDidLoad (loaded_module_list);
    return true;
}


uint32_t
DynamicLoaderDarwinKernel::ReadKextSummaries (const Address &kext_summary_addr,
                                              uint32_t image_infos_count, 
                                              OSKextLoadedKextSummary::collection &image_infos)
{
    const ByteOrder endian = m_kernel.GetByteOrder();
    const uint32_t addr_size = m_kernel.GetAddressByteSize();

    image_infos.resize(image_infos_count);
    const size_t count = image_infos.size() * m_kext_summary_header.entry_size;
    DataBufferHeap data(count, 0);
    Error error;
    
    const bool prefer_file_cache = false;
    const size_t bytes_read = m_process->GetTarget().ReadMemory (kext_summary_addr, 
                                                                 prefer_file_cache,
                                                                 data.GetBytes(), 
                                                                 data.GetByteSize(),
                                                                 error);
    if (bytes_read == count)
    {
        
        DataExtractor extractor (data.GetBytes(), data.GetByteSize(), endian, addr_size);
        uint32_t i=0;
        for (uint32_t kext_summary_offset = 0;
             i < image_infos.size() && extractor.ValidOffsetForDataOfSize(kext_summary_offset, m_kext_summary_header.entry_size); 
             ++i, kext_summary_offset += m_kext_summary_header.entry_size)
        {
            uint32_t offset = kext_summary_offset;
            const void *name_data = extractor.GetData(&offset, KERNEL_MODULE_MAX_NAME);
            if (name_data == NULL)
                break;
            memcpy (image_infos[i].name, name_data, KERNEL_MODULE_MAX_NAME);
            image_infos[i].uuid.SetBytes(extractor.GetData (&offset, 16));
            image_infos[i].address          = extractor.GetU64(&offset);
            if (!image_infos[i].so_address.SetLoadAddress (image_infos[i].address, &m_process->GetTarget()))
                m_process->GetTarget().GetImages().ResolveFileAddress (image_infos[i].address, image_infos[i].so_address);
            image_infos[i].size             = extractor.GetU64(&offset);
            image_infos[i].version          = extractor.GetU64(&offset);
            image_infos[i].load_tag         = extractor.GetU32(&offset);
            image_infos[i].flags            = extractor.GetU32(&offset);
            if ((offset - kext_summary_offset) < m_kext_summary_header.entry_size)
            {
                image_infos[i].reference_list = extractor.GetU64(&offset);
            }
            else
            {
                image_infos[i].reference_list = 0;
            }
//            printf ("[%3u] %*.*s: address=0x%16.16llx, size=0x%16.16llx, version=0x%16.16llx, load_tag=0x%8.8x, flags=0x%8.8x\n", 
//                    i,
//                    KERNEL_MODULE_MAX_NAME, KERNEL_MODULE_MAX_NAME,  (char *)name_data, 
//                    image_infos[i].address, 
//                    image_infos[i].size,
//                    image_infos[i].version,
//                    image_infos[i].load_tag,
//                    image_infos[i].flags);
        }
        if (i < image_infos.size())
            image_infos.resize(i);
    }
    else
    {
        image_infos.clear();
    }
    return image_infos.size();
}

bool
DynamicLoaderDarwinKernel::ReadAllKextSummaries ()
{
    LogSP log(GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER));
    
    Mutex::Locker locker(m_mutex);
    
    if (ReadKextSummaryHeader ())
    {
        if (m_kext_summary_header.entry_count > 0 && m_kext_summary_header_addr.IsValid())
        {
            Address summary_addr (m_kext_summary_header_addr);
            summary_addr.Slide(m_kext_summary_header.GetSize());
            if (!ParseKextSummaries (summary_addr, m_kext_summary_header.entry_count))
            {
                m_kext_summaries.clear();
            }
            return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// Dump an image info structure to the file handle provided.
//----------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::OSKextLoadedKextSummary::PutToLog (Log *log) const
{
    if (log == NULL)
        return;
    const uint8_t *u = (uint8_t *)uuid.GetBytes();

    if (address == LLDB_INVALID_ADDRESS)
    {
        if (u)
        {
            log->Printf("\tuuid=%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X name=\"%s\" (UNLOADED)",
                        u[ 0], u[ 1], u[ 2], u[ 3],
                        u[ 4], u[ 5], u[ 6], u[ 7],
                        u[ 8], u[ 9], u[10], u[11],
                        u[12], u[13], u[14], u[15],
                        name);
        }
        else
            log->Printf("\tname=\"%s\" (UNLOADED)", name);
    }
    else
    {
        if (u)
        {
            log->Printf("\taddr=0x%16.16llx size=0x%16.16llx version=0x%16.16llx load-tag=0x%8.8x flags=0x%8.8x ref-list=0x%16.16llx uuid=%2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X name=\"%s\"",
                        address, size, version, load_tag, flags, reference_list,
                        u[ 0], u[ 1], u[ 2], u[ 3], u[ 4], u[ 5], u[ 6], u[ 7],
                        u[ 8], u[ 9], u[10], u[11], u[12], u[13], u[14], u[15],
                        name);
        }
        else
        {
            log->Printf("\t[0x%16.16llx - 0x%16.16llx) version=0x%16.16llx load-tag=0x%8.8x flags=0x%8.8x ref-list=0x%16.16llx name=\"%s\"",
                        address, address+size, version, load_tag, flags, reference_list,
                        name);
        }
    }
}

//----------------------------------------------------------------------
// Dump the _dyld_all_image_infos members and all current image infos
// that we have parsed to the file handle provided.
//----------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::PutToLog(Log *log) const
{
    if (log == NULL)
        return;

    Mutex::Locker locker(m_mutex);
    log->Printf("gLoadedKextSummaries = 0x%16.16llx { version=%u, entry_size=%u, entry_count=%u }",
                m_kext_summary_header_addr.GetFileAddress(),
                m_kext_summary_header.version,
                m_kext_summary_header.entry_size,
                m_kext_summary_header.entry_count);

    size_t i;
    const size_t count = m_kext_summaries.size();
    if (count > 0)
    {
        log->PutCString("Loaded:");
        for (i = 0; i<count; i++)
            m_kext_summaries[i].PutToLog(log);
    }
}

void
DynamicLoaderDarwinKernel::PrivateInitialize(Process *process)
{
    DEBUG_PRINTF("DynamicLoaderDarwinKernel::%s() process state = %s\n", __FUNCTION__, StateAsCString(m_process->GetState()));
    Clear(true);
    m_process = process;
}

void
DynamicLoaderDarwinKernel::SetNotificationBreakpointIfNeeded ()
{
    if (m_break_id == LLDB_INVALID_BREAK_ID && m_kernel.module_sp)
    {
        DEBUG_PRINTF("DynamicLoaderDarwinKernel::%s() process state = %s\n", __FUNCTION__, StateAsCString(m_process->GetState()));

        
        const bool internal_bp = true;
        const LazyBool skip_prologue = eLazyBoolNo;
        FileSpecList module_spec_list;
        module_spec_list.Append (m_kernel.module_sp->GetFileSpec());
        Breakpoint *bp = m_process->GetTarget().CreateBreakpoint (&module_spec_list,
                                                                  NULL,
                                                                  "OSKextLoadedKextSummariesUpdated",
                                                                  eFunctionNameTypeFull,
                                                                  skip_prologue,
                                                                  internal_bp).get();

        bp->SetCallback (DynamicLoaderDarwinKernel::BreakpointHitCallback, this, true);
        m_break_id = bp->GetID();
    }
}

//----------------------------------------------------------------------
// Member function that gets called when the process state changes.
//----------------------------------------------------------------------
void
DynamicLoaderDarwinKernel::PrivateProcessStateChanged (Process *process, StateType state)
{
    DEBUG_PRINTF("DynamicLoaderDarwinKernel::%s(%s)\n", __FUNCTION__, StateAsCString(state));
    switch (state)
    {
    case eStateConnected:
    case eStateAttaching:
    case eStateLaunching:
    case eStateInvalid:
    case eStateUnloaded:
    case eStateExited:
    case eStateDetached:
        Clear(false);
        break;

    case eStateStopped:
        UpdateIfNeeded();
        break;

    case eStateRunning:
    case eStateStepping:
    case eStateCrashed:
    case eStateSuspended:
        break;

    default:
        break;
    }
}

ThreadPlanSP
DynamicLoaderDarwinKernel::GetStepThroughTrampolinePlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP thread_plan_sp;
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Could not find symbol for step through.");
    return thread_plan_sp;
}

Error
DynamicLoaderDarwinKernel::CanLoadImage ()
{
    Error error;
    error.SetErrorString("always unsafe to load or unload shared libraries in the darwin kernel");
    return error;
}

void
DynamicLoaderDarwinKernel::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance,
                                   DebuggerInitialize);
}

void
DynamicLoaderDarwinKernel::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

void
DynamicLoaderDarwinKernel::DebuggerInitialize (lldb_private::Debugger &debugger)
{
    if (!PluginManager::GetSettingForDynamicLoaderPlugin (debugger, DynamicLoaderDarwinKernelProperties::GetSettingName()))
    {
        const bool is_global_setting = true;
        PluginManager::CreateSettingForDynamicLoaderPlugin (debugger,
                                                            GetGlobalProperties()->GetValueProperties(),
                                                            ConstString ("Properties for the DynamicLoaderDarwinKernel plug-in."),
                                                            is_global_setting);
    }
}

const char *
DynamicLoaderDarwinKernel::GetPluginNameStatic()
{
    return "dynamic-loader.darwin-kernel";
}

const char *
DynamicLoaderDarwinKernel::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library loads/unloads in the MacOSX kernel.";
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
DynamicLoaderDarwinKernel::GetPluginName()
{
    return "DynamicLoaderDarwinKernel";
}

const char *
DynamicLoaderDarwinKernel::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DynamicLoaderDarwinKernel::GetPluginVersion()
{
    return 1;
}

lldb::ByteOrder
DynamicLoaderDarwinKernel::GetByteOrderFromMagic (uint32_t magic)
{
    switch (magic)
    {
        case llvm::MachO::HeaderMagic32:
        case llvm::MachO::HeaderMagic64:
            return lldb::endian::InlHostByteOrder();
            
        case llvm::MachO::HeaderMagic32Swapped:
        case llvm::MachO::HeaderMagic64Swapped:
            if (lldb::endian::InlHostByteOrder() == lldb::eByteOrderBig)
                return lldb::eByteOrderLittle;
            else
                return lldb::eByteOrderBig;
            
        default:
            break;
    }
    return lldb::eByteOrderInvalid;
}

