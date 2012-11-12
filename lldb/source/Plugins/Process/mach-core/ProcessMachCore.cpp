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
#include "llvm/Support/MachO.h"
#include "llvm/Support/MathExtras.h"

// Other libraries and framework includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
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

#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"

using namespace lldb;
using namespace lldb_private;

const char *
ProcessMachCore::GetPluginNameStatic()
{
    return "mach-o-core";
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
ProcessMachCore::CreateInstance (Target &target, Listener &listener, const FileSpec *crash_file)
{
    lldb::ProcessSP process_sp;
    if (crash_file)
        process_sp.reset(new ProcessMachCore (target, listener, *crash_file));
    return process_sp;
}

bool
ProcessMachCore::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    if (!m_core_module_sp && m_core_file.Exists())
    {
        ModuleSpec core_module_spec(m_core_file, target.GetArchitecture());
        Error error (ModuleList::GetSharedModule (core_module_spec, 
                                                  m_core_module_sp, 
                                                  NULL,
                                                  NULL, 
                                                  NULL));

        if (m_core_module_sp)
        {
            const llvm::Triple &triple_ref = m_core_module_sp->GetArchitecture().GetTriple();
            if (triple_ref.getVendor() == llvm::Triple::Apple)
            {
                ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
                if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile)
                    return true;
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessMachCore constructor
//----------------------------------------------------------------------
ProcessMachCore::ProcessMachCore(Target& target, Listener &listener, const FileSpec &core_file) :
    Process (target, listener),
    m_core_aranges (),
    m_core_module_sp (),
    m_core_file (core_file),
    m_dyld_addr (LLDB_INVALID_ADDRESS),
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
const char *
ProcessMachCore::GetPluginName()
{
    return "Process debugging plug-in that loads mach-o core files.";
}

const char *
ProcessMachCore::GetShortPluginName()
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
    llvm::MachO::mach_header header;
    Error error;
    if (DoReadMemory (addr, &header, sizeof(header), error) != sizeof(header))
        return false;
    if (header.magic == llvm::MachO::HeaderMagic32Swapped ||
        header.magic == llvm::MachO::HeaderMagic64Swapped)
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
    //printf("0x%16.16llx: magic = 0x%8.8x, file_type= %u\n", vaddr, header.magic, header.filetype);
    if (header.magic == llvm::MachO::HeaderMagic32 ||
        header.magic == llvm::MachO::HeaderMagic64)
    {
        // Check MH_EXECUTABLE to see if we can find the mach image
        // that contains the shared library list. The dynamic loader 
        // (dyld) is what contains the list for user applications,
        // and the mach kernel contains a global that has the list 
        // of kexts to load
        switch (header.filetype)
        {
        case llvm::MachO::HeaderFileTypeDynamicLinkEditor:
            //printf("0x%16.16llx: file_type = MH_DYLINKER\n", vaddr);
            // Address of dyld "struct mach_header" in the core file
            m_dyld_plugin_name = DynamicLoaderMacOSXDYLD::GetPluginNameStatic();
            m_dyld_addr = addr;
            return true;

        case llvm::MachO::HeaderFileTypeExecutable:
            //printf("0x%16.16llx: file_type = MH_EXECUTE\n", vaddr);
            // Check MH_EXECUTABLE file types to see if the dynamic link object flag
            // is NOT set. If it isn't, then we have a mach_kernel.
            if ((header.flags & llvm::MachO::HeaderFlagBitIsDynamicLinkObject) == 0)
            {
                m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
                // Address of the mach kernel "struct mach_header" in the core file.
                m_dyld_addr = addr;    
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
//            printf ("LC_SEGMENT[%u] arange=[0x%16.16llx - 0x%16.16llx), frange=[0x%8.8x - 0x%8.8x)\n", 
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
            
            // After we have added this section to our m_core_aranges map,
            // we can check the start of the section to see if it might
            // contain dyld for user space apps, or the mach kernel file 
            // for kernel cores.
            if (m_dyld_addr == LLDB_INVALID_ADDRESS)
                GetDynamicLoaderAddress (section_vm_addr);
        }
    }
    if (!ranges_are_sorted)
    {
        m_core_aranges.Sort();
    }

    // Even if the architecture is set in the target, we need to override
    // it to match the core file which is always single arch.
    ArchSpec arch (m_core_module_sp->GetArchitecture());
    if (arch.GetCore() == ArchSpec::eCore_x86_32_i486)
    {
        arch.SetTriple ("i386", m_target.GetPlatform().get());
    }
    if (arch.IsValid())
        m_target.SetArchitecture(arch);            

    if (m_dyld_addr == LLDB_INVALID_ADDRESS)
    {
        // Check the magic kernel address for the mach image header address in case
        // it is there. 
        if (arch.GetAddressByteSize() == 8)
        {
            Error header_addr_error;
            addr_t header_addr = ReadPointerFromMemory (0xffffff8000002010ull, header_addr_error);
            if (header_addr != LLDB_INVALID_ADDRESS)
                GetDynamicLoaderAddress (header_addr);
        }
        else
        {
            Error header_addr_error;
            addr_t header_addr = ReadPointerFromMemory (0xffff0110, header_addr_error);
            if (header_addr != LLDB_INVALID_ADDRESS)
                GetDynamicLoaderAddress (header_addr);
        }
    }

    return error;
}

lldb_private::DynamicLoader *
ProcessMachCore::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, m_dyld_plugin_name.empty() ? NULL : m_dyld_plugin_name.c_str()));
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
            new_thread_list.AddThread (old_thread_list.GetThreadAtIndex (i));
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
            error.SetErrorStringWithFormat ("core file does not contain 0x%llx", addr);
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
    static bool g_initialized = false;
    
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);        
    }
}

addr_t
ProcessMachCore::GetImageInfoAddress()
{
    return m_dyld_addr;
}


lldb_private::ObjectFile *
ProcessMachCore::GetCoreObjectFile ()
{
    return m_core_module_sp->GetObjectFile();
}
