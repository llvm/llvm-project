//===-- ProcessMachCore.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <stdlib.h>

// C++ Includes
#include "llvm/Support/MathExtras.h"
#include <mutex>

// Other libraries and framework includes
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes
#include "ProcessMachCore.h"
#include "ThreadMachCore.h"
#include "StopInfoMachException.h"

// Needed for the plug-in names for the dynamic loaders.
#include "lldb/Utility/SafeMachO.h"

#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"

using namespace lldb;
using namespace lldb_private;

ConstString
ProcessMachCore::GetPluginNameStatic()
{
    static ConstString g_name("mach-o-core");
    return g_name;
}

const char *
ProcessMachCore::GetPluginDescriptionStatic()
{
    return "Mach-O core file debugging plug-in.";
}

void
ProcessMachCore::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessMachCore::CreateInstance);
}


lldb::ProcessSP
ProcessMachCore::CreateInstance (lldb::TargetSP target_sp, Listener &listener, const FileSpec *crash_file)
{
    lldb::ProcessSP process_sp;
    if (crash_file)
    {
        const size_t header_size = sizeof(llvm::MachO::mach_header);
        lldb::DataBufferSP data_sp (crash_file->ReadFileContents(0, header_size));
        if (data_sp && data_sp->GetByteSize() == header_size)
        {
            DataExtractor data(data_sp, lldb::eByteOrderLittle, 4);
            
            lldb::offset_t data_offset = 0;
            llvm::MachO::mach_header mach_header;
            if (ObjectFileMachO::ParseHeader(data, &data_offset, mach_header))
            {
                if (mach_header.filetype == llvm::MachO::MH_CORE)
                    process_sp.reset(new ProcessMachCore (target_sp, listener, *crash_file));
            }
        }
        
    }
    return process_sp;
}

bool
ProcessMachCore::CanDebug(lldb::TargetSP target_sp, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    if (!m_core_module_sp && m_core_file.Exists())
    {
        // Don't add the Target's architecture to the ModuleSpec - we may be working
        // with a core file that doesn't have the correct cpusubtype in the header
        // but we should still try to use it - ModuleSpecList::FindMatchingModuleSpec
        // enforces a strict arch mach.
        ModuleSpec core_module_spec(m_core_file);
        Error error (ModuleList::GetSharedModule (core_module_spec, 
                                                  m_core_module_sp, 
                                                  NULL,
                                                  NULL, 
                                                  NULL));

        if (m_core_module_sp)
        {
            ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
            if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile)
                return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessMachCore constructor
//----------------------------------------------------------------------
ProcessMachCore::ProcessMachCore(lldb::TargetSP target_sp, Listener &listener, const FileSpec &core_file) :
    Process (target_sp, listener),
    m_core_aranges (),
    m_core_module_sp (),
    m_core_file (core_file),
    m_dyld_addr (LLDB_INVALID_ADDRESS),
    m_mach_kernel_addr (LLDB_INVALID_ADDRESS),
    m_dyld_plugin_name ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessMachCore::~ProcessMachCore()
{
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
ConstString
ProcessMachCore::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessMachCore::GetPluginVersion()
{
    return 1;
}

bool
ProcessMachCore::GetDynamicLoaderAddress (lldb::addr_t addr)
{
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER | LIBLLDB_LOG_PROCESS));
    llvm::MachO::mach_header header;
    Error error;
    if (DoReadMemory (addr, &header, sizeof(header), error) != sizeof(header))
        return false;
    if (header.magic == llvm::MachO::MH_CIGAM ||
        header.magic == llvm::MachO::MH_CIGAM_64)
    {
        header.magic        = llvm::ByteSwap_32(header.magic);
        header.cputype      = llvm::ByteSwap_32(header.cputype);
        header.cpusubtype   = llvm::ByteSwap_32(header.cpusubtype);
        header.filetype     = llvm::ByteSwap_32(header.filetype);
        header.ncmds        = llvm::ByteSwap_32(header.ncmds);
        header.sizeofcmds   = llvm::ByteSwap_32(header.sizeofcmds);
        header.flags        = llvm::ByteSwap_32(header.flags);
    }

    // TODO: swap header if needed...
    //printf("0x%16.16" PRIx64 ": magic = 0x%8.8x, file_type= %u\n", vaddr, header.magic, header.filetype);
    if (header.magic == llvm::MachO::MH_MAGIC ||
        header.magic == llvm::MachO::MH_MAGIC_64)
    {
        // Check MH_EXECUTABLE to see if we can find the mach image
        // that contains the shared library list. The dynamic loader 
        // (dyld) is what contains the list for user applications,
        // and the mach kernel contains a global that has the list 
        // of kexts to load
        switch (header.filetype)
        {
        case llvm::MachO::MH_DYLINKER:
            //printf("0x%16.16" PRIx64 ": file_type = MH_DYLINKER\n", vaddr);
            // Address of dyld "struct mach_header" in the core file
            if (log)
                log->Printf ("ProcessMachCore::GetDynamicLoaderAddress found a user process dyld binary image at 0x%" PRIx64, addr);
            m_dyld_addr = addr;
            return true;

        case llvm::MachO::MH_EXECUTE:
            //printf("0x%16.16" PRIx64 ": file_type = MH_EXECUTE\n", vaddr);
            // Check MH_EXECUTABLE file types to see if the dynamic link object flag
            // is NOT set. If it isn't, then we have a mach_kernel.
            if ((header.flags & llvm::MachO::MH_DYLDLINK) == 0)
            {
                if (log)
                    log->Printf ("ProcessMachCore::GetDynamicLoaderAddress found a mach kernel binary image at 0x%" PRIx64, addr);
                // Address of the mach kernel "struct mach_header" in the core file.
                m_mach_kernel_addr = addr;
                return true;
            }
            break;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessMachCore::DoLoadCore ()
{
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER | LIBLLDB_LOG_PROCESS));
    Error error;
    if (!m_core_module_sp)
    {
        error.SetErrorString ("invalid core module");   
        return error;
    }

    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
    if (core_objfile == NULL)
    {
        error.SetErrorString ("invalid core object file");   
        return error;
    }
    
    if (core_objfile->GetNumThreadContexts() == 0)
    {
        error.SetErrorString ("core file doesn't contain any LC_THREAD load commands, or the LC_THREAD architecture is not supported in this lldb");
        return error;
    }

    SectionList *section_list = core_objfile->GetSectionList();
    if (section_list == NULL)
    {
        error.SetErrorString ("core file has no sections");   
        return error;
    }
        
    const uint32_t num_sections = section_list->GetNumSections(0);
    if (num_sections == 0)
    {
        error.SetErrorString ("core file has no sections");   
        return error;
    }
    
    SetCanJIT(false);

    llvm::MachO::mach_header header;
    DataExtractor data (&header, 
                        sizeof(header), 
                        m_core_module_sp->GetArchitecture().GetByteOrder(),
                        m_core_module_sp->GetArchitecture().GetAddressByteSize());

    bool ranges_are_sorted = true;
    addr_t vm_addr = 0;
    for (uint32_t i=0; i<num_sections; ++i)
    {
        Section *section = section_list->GetSectionAtIndex (i).get();
        if (section)
        {
            lldb::addr_t section_vm_addr = section->GetFileAddress();
            FileRange file_range (section->GetFileOffset(), section->GetFileSize());
            VMRangeToFileOffset::Entry range_entry (section_vm_addr,
                                                    section->GetByteSize(),
                                                    file_range);
            
            if (vm_addr > section_vm_addr)
                ranges_are_sorted = false;
            vm_addr = section->GetFileAddress();
            VMRangeToFileOffset::Entry *last_entry = m_core_aranges.Back();
//            printf ("LC_SEGMENT[%u] arange=[0x%16.16" PRIx64 " - 0x%16.16" PRIx64 "), frange=[0x%8.8x - 0x%8.8x)\n",
//                    i, 
//                    range_entry.GetRangeBase(),
//                    range_entry.GetRangeEnd(),
//                    range_entry.data.GetRangeBase(),
//                    range_entry.data.GetRangeEnd());

            if (last_entry &&
                last_entry->GetRangeEnd() == range_entry.GetRangeBase() &&
                last_entry->data.GetRangeEnd() == range_entry.data.GetRangeBase())
            {
                last_entry->SetRangeEnd (range_entry.GetRangeEnd());
                last_entry->data.SetRangeEnd (range_entry.data.GetRangeEnd());
                //puts("combine");
            }
            else
            {
                m_core_aranges.Append(range_entry);
            }
        }
    }
    if (!ranges_are_sorted)
    {
        m_core_aranges.Sort();
    }

    if (m_dyld_addr == LLDB_INVALID_ADDRESS || m_mach_kernel_addr == LLDB_INVALID_ADDRESS)
    {
        // We need to locate the main executable in the memory ranges
        // we have in the core file.  We need to search for both a user-process dyld binary
        // and a kernel binary in memory; we must look at all the pages in the binary so
        // we don't miss one or the other.  Step through all memory segments searching for
        // a kernel binary and for a user process dyld -- we'll decide which to prefer 
        // later if both are present.

        const size_t num_core_aranges = m_core_aranges.GetSize();
        for (size_t i = 0; i < num_core_aranges; ++i)
        {
            const VMRangeToFileOffset::Entry *entry = m_core_aranges.GetEntryAtIndex(i);
            lldb::addr_t section_vm_addr_start = entry->GetRangeBase();
            lldb::addr_t section_vm_addr_end = entry->GetRangeEnd();
            for (lldb::addr_t section_vm_addr = section_vm_addr_start;
                 section_vm_addr < section_vm_addr_end;
                 section_vm_addr += 0x1000)
            {
                GetDynamicLoaderAddress (section_vm_addr);
            }
        }
    }

    // If we found both a user-process dyld and a kernel binary, we need to decide
    // which to prefer.
    if (GetCorefilePreference() == eKernelCorefile)
    {
        if (m_mach_kernel_addr != LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("ProcessMachCore::DoLoadCore: Using kernel corefile image at 0x%" PRIx64, m_mach_kernel_addr);
            m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
        }
        else if (m_dyld_addr != LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("ProcessMachCore::DoLoadCore: Using user process dyld image at 0x%" PRIx64, m_dyld_addr);
            m_dyld_plugin_name = DynamicLoaderMacOSXDYLD::GetPluginNameStatic();
        }
    }
    else
    {
        if (m_dyld_addr != LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("ProcessMachCore::DoLoadCore: Using user process dyld image at 0x%" PRIx64, m_dyld_addr);
            m_dyld_plugin_name = DynamicLoaderMacOSXDYLD::GetPluginNameStatic();
        }
        else if (m_mach_kernel_addr != LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("ProcessMachCore::DoLoadCore: Using kernel corefile image at 0x%" PRIx64, m_mach_kernel_addr);
            m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
        }
    }

    // Even if the architecture is set in the target, we need to override
    // it to match the core file which is always single arch.
    ArchSpec arch (m_core_module_sp->GetArchitecture());
    if (arch.GetCore() == ArchSpec::eCore_x86_32_i486)
    {
        arch.SetTriple ("i386", GetTarget().GetPlatform().get());
    }
    if (arch.IsValid())
        GetTarget().SetArchitecture(arch);

    return error;
}

lldb_private::DynamicLoader *
ProcessMachCore::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, m_dyld_plugin_name.IsEmpty() ? NULL : m_dyld_plugin_name.GetCString()));
    return m_dyld_ap.get();
}

bool
ProcessMachCore::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    if (old_thread_list.GetSize(false) == 0)
    {
        // Make up the thread the first time this is called so we can setup our one and only
        // core thread state.
        ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();

        if (core_objfile)
        {
            const uint32_t num_threads = core_objfile->GetNumThreadContexts ();
            for (lldb::tid_t tid = 0; tid < num_threads; ++tid)
            {
                ThreadSP thread_sp(new ThreadMachCore (*this, tid));
                new_thread_list.AddThread (thread_sp);
            }
        }
    }
    else
    {
        const uint32_t num_threads = old_thread_list.GetSize(false);
        for (uint32_t i=0; i<num_threads; ++i)
            new_thread_list.AddThread (old_thread_list.GetThreadAtIndex (i, false));
    }
    return new_thread_list.GetSize(false) > 0;
}

void
ProcessMachCore::RefreshStateAfterStop ()
{
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
    //SetThreadStopInfo (m_last_stop_packet);
}

Error
ProcessMachCore::DoDestroy ()
{
    return Error();
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessMachCore::IsAlive ()
{
    return true;
}

bool
ProcessMachCore::WarnBeforeDetach () const
{
    return false;
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessMachCore::ReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    // Don't allow the caching that lldb_private::Process::ReadMemory does
    // since in core files we have it all cached our our core file anyway.
    return DoReadMemory (addr, buf, size, error);
}

size_t
ProcessMachCore::DoReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();

    if (core_objfile)
    {
        const VMRangeToFileOffset::Entry *core_memory_entry = m_core_aranges.FindEntryThatContains (addr);
        if (core_memory_entry)
        {
            const addr_t offset = addr - core_memory_entry->GetRangeBase();
            const addr_t bytes_left = core_memory_entry->GetRangeEnd() - addr;
            size_t bytes_to_read = size;
            if (bytes_to_read > bytes_left)
                bytes_to_read = bytes_left;
            return core_objfile->CopyData (core_memory_entry->data.GetRangeBase() + offset, bytes_to_read, buf);
        }
        else
        {
            error.SetErrorStringWithFormat ("core file does not contain 0x%" PRIx64, addr);
        }
    }
    return 0;
}

void
ProcessMachCore::Clear()
{
    m_thread_list.Clear();
}

void
ProcessMachCore::Initialize()
{
    static std::once_flag g_once_flag;

    std::call_once(g_once_flag, []() {
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);
    });
}

addr_t
ProcessMachCore::GetImageInfoAddress()
{
    // If we found both a user-process dyld and a kernel binary, we need to decide
    // which to prefer.
    if (GetCorefilePreference() == eKernelCorefile)
    {
        if (m_mach_kernel_addr != LLDB_INVALID_ADDRESS)
        {
            return m_mach_kernel_addr;
        }
        return m_dyld_addr;
    }
    else
    {
        if (m_dyld_addr != LLDB_INVALID_ADDRESS)
        {
            return m_dyld_addr;
        }
        return m_mach_kernel_addr;
    }
}


lldb_private::ObjectFile *
ProcessMachCore::GetCoreObjectFile ()
{
    return m_core_module_sp->GetObjectFile();
}
