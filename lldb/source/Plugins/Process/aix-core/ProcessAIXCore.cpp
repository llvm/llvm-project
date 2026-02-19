//===-- ProcessAIXCore.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include <memory>
#include <mutex>

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

#include "llvm/Support/Threading.h"
#include "Plugins/DynamicLoader/AIX-DYLD/DynamicLoaderAIXDYLD.h"

#include "ProcessAIXCore.h"
#include "AIXCore.h"
#include "ThreadAIXCore.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ProcessAIXCore)

llvm::StringRef ProcessAIXCore::GetPluginDescriptionStatic() {
  return "AIX core dump plug-in.";
}

void ProcessAIXCore::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessAIXCore::Terminate() {
  PluginManager::UnregisterPlugin(ProcessAIXCore::CreateInstance);
}

lldb::ProcessSP ProcessAIXCore::CreateInstance(lldb::TargetSP target_sp,
        lldb::ListenerSP listener_sp,
        const FileSpec *crash_file,
        bool can_connect) {
    lldb::ProcessSP process_sp;
    if (crash_file && !can_connect) {
        const size_t header_size = 
            std::max(sizeof(AIXCORE::AIXCore64Header), sizeof(AIXCORE::AIXCore32Header));

        auto data_sp = FileSystem::Instance().CreateDataBuffer(
                crash_file->GetPath(), header_size, 0);

        if (data_sp && data_sp->GetByteSize()) {
            DataExtractor data(data_sp, lldb::eByteOrderBig, 4);
            lldb::offset_t offset = 0;
            offset += 4; // Skipping to the coredump version
            uint32_t magic = data.GetU32(&offset);
            if (magic == 0xfeeddb1) {
                AIXCORE::AIXCore32Header aixcore_header;
                if(aixcore_header.ParseCoreHeader(data, &offset)) {
                    process_sp = std::make_shared<ProcessAIXCore>(target_sp, listener_sp,
                            *crash_file);
                }
            }
            else if (magic == 0xfeeddb2) {
                AIXCORE::AIXCore64Header aixcore_header;
                if(aixcore_header.ParseCoreHeader(data, &offset)) {
                    process_sp = std::make_shared<ProcessAIXCore>(target_sp, listener_sp,
                            *crash_file);
                }
            }
        }

    }
    return process_sp;
}

// ProcessAIXCore constructor
ProcessAIXCore::ProcessAIXCore(lldb::TargetSP target_sp,
                               lldb::ListenerSP listener_sp,
                               const FileSpec &core_file)
    : PostMortemProcess(target_sp, listener_sp, core_file) {}

// Destructor
ProcessAIXCore::~ProcessAIXCore() {
  Clear();
  // We need to call finalize on the process before destroying ourselves to
  // make sure all of the broadcaster cleanup goes as planned. If we destruct
  // this class, then Process::~Process() might have problems trying to fully
  // destroy the broadcaster.
  Finalize(true /* destructing */);
}

lldb::addr_t ProcessAIXCore::AddAddressRanges(AIXCORE::AIXCore64Header header) {
  const lldb::addr_t addr = header.StackBaseAddr;
  FileRange file_range(header.StackOffset, header.StackSize);
  VMRangeToFileOffset::Entry range_entry(addr, header.StackSize, file_range);

  if (header.StackSize > 0) {
    VMRangeToFileOffset::Entry *last_entry = m_core_aranges.Back();
    if (last_entry &&
        last_entry->GetRangeEnd() == range_entry.GetRangeBase() &&
        last_entry->data.GetRangeEnd() == range_entry.data.GetRangeBase() &&
        last_entry->GetByteSize() == last_entry->data.GetByteSize()) {
        last_entry->SetRangeEnd(range_entry.GetRangeEnd());
        last_entry->data.SetRangeEnd(range_entry.data.GetRangeEnd());
    } else {
        m_core_aranges.Append(range_entry);
    }
  }

  const uint32_t permissions = lldb::ePermissionsReadable |
      lldb::ePermissionsWritable;

  m_core_range_infos.Append(
      VMRangeToPermissions::Entry(addr, header.StackSize, permissions));

  return addr;
}
lldb::addr_t ProcessAIXCore::AddAddressRanges(AIXCORE::AIXCore32Header header) {
  const lldb::addr_t addr = header.StackBaseAddr;
  FileRange file_range(header.StackOffset, header.StackSize);
  VMRangeToFileOffset::Entry range_entry(addr, header.StackSize, file_range);

  if (header.StackSize > 0) {
    VMRangeToFileOffset::Entry *last_entry = m_core_aranges.Back();
    if (last_entry &&
        last_entry->GetRangeEnd() == range_entry.GetRangeBase() &&
        last_entry->data.GetRangeEnd() == range_entry.data.GetRangeBase() &&
        last_entry->GetByteSize() == last_entry->data.GetByteSize()) {
        last_entry->SetRangeEnd(range_entry.GetRangeEnd());
        last_entry->data.SetRangeEnd(range_entry.data.GetRangeEnd());
    } else {
        m_core_aranges.Append(range_entry);
    }
  }

  const uint32_t permissions = lldb::ePermissionsReadable |
      lldb::ePermissionsWritable;

  m_core_range_infos.Append(
      VMRangeToPermissions::Entry(addr, header.StackSize, permissions));

  return addr;
}

bool ProcessAIXCore::CanDebug(lldb::TargetSP target_sp,
                                bool plugin_specified_by_name) {

    if (!m_core_module_sp && FileSystem::Instance().Exists(m_core_file)) {
        ModuleSpec core_module_spec(m_core_file, target_sp->GetArchitecture());
        core_module_spec.SetTarget(target_sp);
        Status error(ModuleList::GetSharedModule(core_module_spec, m_core_module_sp,
                                                 nullptr, nullptr));
        if (m_core_module_sp) {
            ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
            if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile){
                return true;
            }
        }
    }
    return false;

}

ArchSpec ProcessAIXCore::GetArchitecture() {

  ArchSpec arch = m_core_module_sp->GetObjectFile()->GetArchitecture();

  ArchSpec target_arch = GetTarget().GetArchitecture();
  arch.MergeFrom(target_arch);

  return arch;
}

lldb_private::DynamicLoader *ProcessAIXCore::GetDynamicLoader() {
  if (m_dyld_up.get() == nullptr) {
    m_dyld_up.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderAIXDYLD::GetPluginNameStatic()));
  }
  return m_dyld_up.get();
}

void ProcessAIXCore::ParseAIXCoreFile() {
    
    Log *log = GetLog(LLDBLog::Process);
    
    const lldb_private::UnixSignals &unix_signals = *GetUnixSignals();
    const ArchSpec &arch = GetArchitecture();
    
    const uint32_t num_threads = m_aixcore_header.NumberOfThreads;
    SetID(m_aixcore_header.User.process.pi_pid);
    m_thread_data.clear();
    m_thread_data.reserve(num_threads > 0 ? (num_threads + 1) : 1);

    for (uint32_t i = 0; i <= num_threads; i++) {
    
        AIXSigInfo siginfo;
        ThreadData thread_data;
        size_t regs_size;

        std::string base_name(m_aixcore_header.User.process.pi_comm,
                          strnlen (m_aixcore_header.User.process.pi_comm,
                             sizeof (m_aixcore_header.User.process.pi_comm)));

        regs_size = (i == 0) ? sizeof(m_aixcore_header.Fault.context)
            : sizeof(m_aixcore_header.threads[i-1].context); 
        
        lldb::DataBufferSP regs_buf_sp(new lldb_private::DataBufferHeap(regs_size, 0));
        if (i == 0) { // The crash thread
            thread_data.tid = m_aixcore_header.Fault.thread.ti_tid;
            thread_data.name = base_name;
            memcpy(static_cast<void *>(const_cast<uint8_t *>(regs_buf_sp->GetBytes())),
                   &m_aixcore_header.Fault.context, regs_size);
            thread_data.siginfo.Parse(m_aixcore_header, arch, unix_signals);
        }
        else { // Other threads
            thread_data.tid = m_aixcore_header.threads[i-1].thread.ti_tid;
            thread_data.name = base_name + "-*thread-" + std::to_string(i) + "*";

            memcpy(static_cast<void *>(const_cast<uint8_t *>(regs_buf_sp->GetBytes())),
                   &m_aixcore_header.threads[i-1].context, regs_size);
        }
        lldb_private::DataExtractor regs_data(regs_buf_sp, lldb::eByteOrderBig, 8);
        thread_data.gpregset = DataExtractor(regs_data, 0, regs_size);

        thread_data.prstatus_sig = 0;
    
        m_thread_data.push_back(std::move(thread_data));

        LLDB_LOGF(log, "ProcessAIXCore: Parsing Complete! tid %d\n",i);
    }
}

void ProcessAIXCore::ParseAIXCore32File() {
    
    Log *log = GetLog(LLDBLog::Process);
    AIXSigInfo siginfo;
    ThreadData thread_data;
    
    const lldb_private::UnixSignals &unix_signals = *GetUnixSignals();
    const ArchSpec &arch = GetArchitecture();
   
    const uint32_t num_threads = m_aixcore32_header.NumberOfThreads;              
    SetID(m_aixcore32_header.User.process.pi_pid);                                
    m_thread_data.clear();                                                      
    m_thread_data.reserve(num_threads > 0 ? (num_threads + 1) : 1); 

    for (uint32_t i = 0; i <= num_threads; i++) {                               
                                                                                
        AIXSigInfo siginfo;                                                     
        ThreadData thread_data;                                                 
        size_t regs_size;                                                       
                                                                                
        std::string base_name(m_aixcore32_header.User.process.pi_comm,            
                          strnlen (m_aixcore32_header.User.process.pi_comm,          
                             sizeof (m_aixcore32_header.User.process.pi_comm)));  
                                                                                
        regs_size = (i == 0) ? sizeof(m_aixcore32_header.Fault.context)           
            : sizeof(m_aixcore32_header.threads[i-1].context);                    
                                                                                
        lldb::DataBufferSP regs_buf_sp(new lldb_private::DataBufferHeap(regs_size, 0));
        if (i == 0) { // The crash thread                                       
            thread_data.tid = m_aixcore32_header.Fault.thread.ti_tid;             
            thread_data.name = base_name;                                       
            memcpy(static_cast<void *>(const_cast<uint8_t *>(regs_buf_sp->GetBytes())),
                   &m_aixcore32_header.Fault.context, regs_size);                 
            thread_data.siginfo.Parse(m_aixcore32_header, arch, unix_signals);       
        }                                                                       
        else { // Other threads                                                 
            thread_data.tid = m_aixcore32_header.threads[i-1].thread.ti_tid;         
            thread_data.name = base_name + "-*thread-" + std::to_string(i) + "*";
                                                                                
            memcpy(static_cast<void *>(const_cast<uint8_t *>(regs_buf_sp->GetBytes())),
                   &m_aixcore32_header.threads[i-1].context, regs_size);          
        }
        lldb_private::DataExtractor regs_data(regs_buf_sp, lldb::eByteOrderBig, 8);
        thread_data.gpregset = DataExtractor(regs_data, 0, regs_size);          
                                                                                
        thread_data.prstatus_sig = 0;                                           
                                                                                
        m_thread_data.push_back(std::move(thread_data));                        
                                                                                
        LLDB_LOGF(log, "ProcessAIXCore: Parsing Complete! tid %d\n",i); 
    }

}
// Process Control
Status ProcessAIXCore::DoLoadCore() {
    
    Status error;
    if (!m_core_module_sp) {
        error = Status::FromErrorString("invalid core module");
        return error;
    }

    FileSpec file = m_core_module_sp->GetObjectFile()->GetFileSpec();
    Log *log = GetLog(LLDBLog::Process);
    
    if (file) {
        auto data_sp = FileSystem::Instance().CreateDataBuffer(
                file.GetPath(), -1, 0);
        if (data_sp && data_sp->GetByteSize()) {
            DataExtractor data(data_sp, lldb::eByteOrderBig, 4);
            lldb::offset_t offset = 0;
            offset += 4; // Skipping to the coredump version
            uint32_t magic = data.GetU32(&offset);
            offset = 0;
            if (magic == 0xfeeddb1) {
                m_is64bit = false;
                m_aixcore32_header.ParseCoreHeader(data, &offset);
                lldb::addr_t addr = AddAddressRanges(m_aixcore32_header);
                if (addr == LLDB_INVALID_ADDRESS)
                    LLDB_LOGF(log, "ProcessAIXCore: Invalid base address. Stack information will be limited");
                auto dyld = static_cast<DynamicLoaderAIXDYLD *>(GetDynamicLoader());
                dyld->FillCoreLoader32Data(data, m_aixcore32_header.LoaderOffset,
                        m_aixcore32_header.LoaderSize);
            }
            else if (magic == 0xfeeddb2) {
                m_aixcore_header.ParseCoreHeader(data, &offset);
                lldb::addr_t addr = AddAddressRanges(m_aixcore_header);
                if (addr == LLDB_INVALID_ADDRESS)
                    LLDB_LOGF(log, "ProcessAIXCore: Invalid base address. Stack information will be limited");
                auto dyld = static_cast<DynamicLoaderAIXDYLD *>(GetDynamicLoader());
                dyld->FillCoreLoaderData(data, m_aixcore_header.LoaderOffset,
                        m_aixcore_header.LoaderSize);
            }
        }
        else {
            error = Status::FromErrorString("invalid data");
            return error;
        }
    } else {
        error = Status::FromErrorString("invalid file");
        return error;
    }

    m_thread_data_valid = true;
    if (m_is64bit)
        ParseAIXCoreFile();
    else
        ParseAIXCore32File();
    ArchSpec arch(m_core_module_sp->GetArchitecture());
    ArchSpec target_arch = GetTarget().GetArchitecture();
    ArchSpec core_arch(m_core_module_sp->GetArchitecture());
    target_arch.MergeFrom(core_arch);
    GetTarget().SetArchitecture(target_arch);
    
    lldb::ModuleSP exe_module_sp = GetTarget().GetExecutableModule();
    if (!exe_module_sp) {
        ModuleSpec exe_module_spec;
        exe_module_spec.GetArchitecture() = arch;
        if(m_is64bit)
            exe_module_spec.GetFileSpec().SetFile(m_aixcore_header.User.process.pi_comm,
                    FileSpec::Style::native);
        else
            exe_module_spec.GetFileSpec().SetFile(m_aixcore32_header.User.process.pi_comm,
                    FileSpec::Style::native);
        exe_module_sp = 
            GetTarget().GetOrCreateModule(exe_module_spec, true /* notify */);
        if (exe_module_sp)
            GetTarget().SetExecutableModule(exe_module_sp, eLoadDependentsNo);
    }
    
    return error;
}

bool ProcessAIXCore::DoUpdateThreadList(ThreadList &old_thread_list,
                                        ThreadList &new_thread_list) 
{
    Log *log = GetLog(LLDBLog::Process);
    const uint32_t num_threads = m_is64bit ? m_aixcore_header.NumberOfThreads :
                                            m_aixcore32_header.NumberOfThreads;
    LLDB_LOGF(log,"Number Of Threads %d\n", num_threads);
    for (lldb::tid_t tid = 0; tid <= num_threads; ++tid) {
        const ThreadData &td = m_thread_data[tid];
        lldb::ThreadSP thread_sp = 
            std::make_shared<ThreadAIXCore>(*this, td);
        if(!thread_sp) {
            LLDB_LOGF(log,"Thread not added %d\n", tid);
            continue;
        }
        new_thread_list.AddThread(thread_sp);
    }
    return true;
} 

void ProcessAIXCore::RefreshStateAfterStop() {}

// Process Memory
size_t ProcessAIXCore::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                  Status &error) {
  if(addr == LLDB_INVALID_ADDRESS)
      return 0;

  if (lldb::ABISP abi_sp = GetABI())
      addr = abi_sp->FixAnyAddress(addr);

  // Don't allow the caching that lldb_private::Process::ReadMemory does since
  // in core files we have it all cached our our core file anyway.
  return DoReadMemory(addr, buf, size, error);
}

size_t ProcessAIXCore::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                    Status &error) {
    ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
    if (core_objfile == nullptr)
        return 0;
    // Get the address range
    const VMRangeToFileOffset::Entry *address_range =
        m_core_aranges.FindEntryThatContains(addr);
    if (address_range == nullptr || address_range->GetRangeEnd() < addr) {
        error = Status::FromErrorStringWithFormat(
                "core file does not contain 0x%" PRIx64, addr);
        return 0;
    }

    // Convert the address into core file offset
    const lldb::addr_t offset = addr - address_range->GetRangeBase();
    const lldb::addr_t file_start = address_range->data.GetRangeBase();
    const lldb::addr_t file_end = address_range->data.GetRangeEnd();
    size_t bytes_to_read = size; // Number of bytes to read from the core file
    size_t bytes_copied = 0;   // Number of bytes actually read from the core file
    // Number of bytes available in the core file from the given address
    lldb::addr_t bytes_left = 0;

    // Don't proceed if core file doesn't contain the actual data for this
    // address range.
    if (file_start == file_end)
        return 0;

    // Figure out how many on-disk bytes remain in this segment starting at the
    // given offset
    if (file_end > file_start + offset)
        bytes_left = file_end - (file_start + offset);

    if (bytes_to_read > bytes_left)
        bytes_to_read = bytes_left;

  // If there is data available on the core file read it
  if (bytes_to_read)
    bytes_copied =
        core_objfile->CopyData(offset + file_start, bytes_to_read, buf);

  return bytes_copied;
}

Status ProcessAIXCore::DoGetMemoryRegionInfo(lldb::addr_t load_addr,
                                              MemoryRegionInfo &region_info) {
    return Status();
}

Status ProcessAIXCore::DoDestroy() { return Status(); }
