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
      const size_t header_size = sizeof(AIXCORE::AIXCore64Header);

      auto data_sp = FileSystem::Instance().CreateDataBuffer(
              crash_file->GetPath(), header_size, 0);

      if (data_sp && data_sp->GetByteSize() == header_size) {
          AIXCORE::AIXCore64Header aixcore_header;
          DataExtractor data(data_sp, lldb::eByteOrderBig, 4);
          lldb::offset_t data_offset = 0;
          if(aixcore_header.ParseCoreHeader(data, &data_offset)) {
              process_sp = std::make_shared<ProcessAIXCore>(target_sp, listener_sp,
                      *crash_file);
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

bool ProcessAIXCore::CanDebug(lldb::TargetSP target_sp,
                                bool plugin_specified_by_name) {

    if (!m_core_module_sp && FileSystem::Instance().Exists(m_core_file)) {
        ModuleSpec core_module_spec(m_core_file, target_sp->GetArchitecture());
        Status error(ModuleList::GetSharedModule(core_module_spec, m_core_module_sp,
                                                 nullptr, nullptr, nullptr));
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
    AIXSigInfo siginfo;
    ThreadData thread_data;
    
    const lldb_private::UnixSignals &unix_signals = *GetUnixSignals();
    const ArchSpec &arch = GetArchitecture();
    
    siginfo.Parse(m_aixcore_header, arch, unix_signals);
    thread_data.siginfo = siginfo;
    SetID(m_aixcore_header.User.process.pi_pid);
    
    thread_data.name.assign (m_aixcore_header.User.process.pi_comm,
            strnlen (m_aixcore_header.User.process.pi_comm,
                sizeof (m_aixcore_header.User.process.pi_comm)));
    
    lldb::DataBufferSP data_buffer_sp(new lldb_private::DataBufferHeap(sizeof(m_aixcore_header.Fault.context), 0));
    
    memcpy(static_cast<void *>(const_cast<uint8_t *>(data_buffer_sp->GetBytes())),
            &m_aixcore_header.Fault.context, sizeof(m_aixcore_header.Fault.context));
    
    lldb_private::DataExtractor data(data_buffer_sp, lldb::eByteOrderBig, 8);

    thread_data.gpregset = DataExtractor(data, 0, sizeof(m_aixcore_header.Fault.context));
    m_thread_data.push_back(thread_data);
    LLDB_LOGF(log, "ProcessAIXCore: Parsing Complete!");

}

// Process Control
Status ProcessAIXCore::DoLoadCore() {
    
    Status error;
    if (!m_core_module_sp) {
        error = Status::FromErrorString("invalid core module");
        return error;
    }

    FileSpec file = m_core_module_sp->GetObjectFile()->GetFileSpec();
    
    if (file) {
        const size_t header_size = sizeof(AIXCORE::AIXCore64Header);
        auto data_sp = FileSystem::Instance().CreateDataBuffer(
                file.GetPath(), -1, 0);
        if (data_sp && data_sp->GetByteSize() != 0) {
            
            DataExtractor data(data_sp, lldb::eByteOrderBig, 4);
            lldb::offset_t data_offset = 0;
            m_aixcore_header.ParseCoreHeader(data, &data_offset);
            auto dyld = static_cast<DynamicLoaderAIXDYLD *>(GetDynamicLoader());
            dyld->FillCoreLoaderData(data, m_aixcore_header.LoaderOffset,
                    m_aixcore_header.LoaderSize);

        } else {
            error = Status::FromErrorString("invalid data");
            return error;
        }
    } else {
        error = Status::FromErrorString("invalid file");
        return error;
    }

    m_thread_data_valid = true;
    ParseAIXCoreFile();
    ArchSpec arch(m_core_module_sp->GetArchitecture());

    ArchSpec target_arch = GetTarget().GetArchitecture();
    ArchSpec core_arch(m_core_module_sp->GetArchitecture());
    target_arch.MergeFrom(core_arch);
    GetTarget().SetArchitecture(target_arch);
    
    lldb::ModuleSP exe_module_sp = GetTarget().GetExecutableModule();
    if (!exe_module_sp) {
        ModuleSpec exe_module_spec;
        exe_module_spec.GetArchitecture() = arch;
        exe_module_spec.GetFileSpec().SetFile(m_aixcore_header.User.process.pi_comm,
                FileSpec::Style::native);
        exe_module_sp = GetTarget().GetOrCreateModule(exe_module_spec, true);
        GetTarget().SetExecutableModule(exe_module_sp, eLoadDependentsNo);
    }
    
    return error;
}

bool ProcessAIXCore::DoUpdateThreadList(ThreadList &old_thread_list,
                                        ThreadList &new_thread_list) 
{
    const ThreadData &td = m_thread_data[0];
    
    lldb::ThreadSP thread_sp = 
        std::make_shared<ThreadAIXCore>(*this, td);
    new_thread_list.AddThread(thread_sp);
    
    return true;
} 

void ProcessAIXCore::RefreshStateAfterStop() {}

// Process Memory
size_t ProcessAIXCore::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                  Status &error) {
  if (lldb::ABISP abi_sp = GetABI())
    addr = abi_sp->FixAnyAddress(addr);

  // Don't allow the caching that lldb_private::Process::ReadMemory does since
  // in core files we have it all cached our our core file anyway.
  return DoReadMemory(addr, buf, size, error);
}

size_t ProcessAIXCore::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                    Status &error) { return 0; }

Status ProcessAIXCore::DoGetMemoryRegionInfo(lldb::addr_t load_addr,
                                              MemoryRegionInfo &region_info) {
    return Status();
}

Status ProcessAIXCore::DoDestroy() { return Status(); }
