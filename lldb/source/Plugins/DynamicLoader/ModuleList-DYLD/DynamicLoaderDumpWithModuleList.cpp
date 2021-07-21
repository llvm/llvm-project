//===-- DynamicLoaderDumpWithModuleList.cpp-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Main header include
#include "DynamicLoaderDumpWithModuleList.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/ThreadPool.h"

#include "Plugins/ObjectFile/Placeholder/ObjectFilePlaceholder.h"
#include <mutex>
#include <regex>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(DynamicLoaderDumpWithModuleList,
                       DynamicLoaderDumpWithModuleList)

void DynamicLoaderDumpWithModuleList::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderDumpWithModuleList::Terminate() {}

llvm::StringRef DynamicLoaderDumpWithModuleList::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in for dumps with module list available";
}

DynamicLoader *DynamicLoaderDumpWithModuleList::CreateInstance(Process *process,
                                                               bool force) {
  // This plug-in is only used when it is requested by name from
  // ProcessELFCore. ProcessELFCore will look to see if the core
  // file contains a NT_FILE ELF note, and ask for this plug-in
  // by name if it does.
  if (force)
    return new DynamicLoaderDumpWithModuleList(process);
  return nullptr;
}

DynamicLoaderDumpWithModuleList::DynamicLoaderDumpWithModuleList(
    Process *process)
    : DynamicLoader(process), m_rendezvous(process), m_auxv(),
      m_vdso_base(LLDB_INVALID_ADDRESS) {}

DynamicLoaderDumpWithModuleList::~DynamicLoaderDumpWithModuleList() {}

void DynamicLoaderDumpWithModuleList::SetLoadedModule(const ModuleSP &module_sp,
                                                      addr_t link_map_addr) {
  llvm::sys::ScopedWriter lock(m_loaded_modules_rw_mutex);
  m_loaded_modules[module_sp] = link_map_addr;
}

void DynamicLoaderDumpWithModuleList::UnloadModule(const ModuleSP &module_sp) {
  llvm::sys::ScopedWriter lock(m_loaded_modules_rw_mutex);
  m_loaded_modules.erase(module_sp);
}

std::optional<lldb::addr_t>
DynamicLoaderDumpWithModuleList::GetLoadedModuleLinkAddr(const ModuleSP &module_sp) {
  llvm::sys::ScopedReader lock(m_loaded_modules_rw_mutex);
  auto it = m_loaded_modules.find(module_sp);
  if (it != m_loaded_modules.end())
    return it->second;
  return std::nullopt;
}

std::optional<const LoadedModuleInfoList::LoadedModuleInfo>
DynamicLoaderDumpWithModuleList::GetModuleInfo(lldb::addr_t module_base_addr) {
  if (m_module_addr_to_info_map.empty()) {
    llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
        m_process->GetLoadedModuleList();
    if (!module_info_list_ep)
      return std::nullopt;

    const LoadedModuleInfoList &module_info_list = *module_info_list_ep;
    if (module_info_list.m_list.empty())
      return std::nullopt;

    for (const LoadedModuleInfoList::LoadedModuleInfo &mod_info :
         module_info_list.m_list) {
      lldb::addr_t base_addr;
      if (!mod_info.get_base(base_addr))
        continue;
      m_module_addr_to_info_map.emplace(base_addr, mod_info);
    }
  }

  auto module_match_iter = m_module_addr_to_info_map.find(module_base_addr);
  if (module_match_iter == m_module_addr_to_info_map.end())
    return std::nullopt;

  return module_match_iter->second;
}

void DynamicLoaderDumpWithModuleList::DetectModuleListMismatch() {
  llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
      m_process->GetLoadedModuleList();
  if (!module_info_list_ep || (*module_info_list_ep).m_list.empty())
    return;

  DYLDRendezvous::iterator I;
  DYLDRendezvous::iterator E;
  uint32_t mismatched_module_count = 0;

  Log *log = GetLog(LLDBLog::DynamicLoader);
  assert(m_rendezvous.IsValid() && "m_rendezvous is not resolved yet.");
  for (I = m_rendezvous.begin(), E = m_rendezvous.end(); I != E; ++I) {
    // vdso is an in-memory module which won't be in loaded module list.
    if (I->base_addr == m_vdso_base)
      continue;

    std::optional<const LoadedModuleInfoList::LoadedModuleInfo> mod_info_opt =
        GetModuleInfo(I->base_addr);
    if (!mod_info_opt.has_value()) {
      ++mismatched_module_count;

      LLDB_LOGF(
          log,
          "DynamicLoaderDumpWithModuleList::%s found mismatch module %s at "
          "rendezvous address 0x%lx",
          __FUNCTION__, I->file_spec.GetPath().c_str(), I->base_addr);
    }
  }
  m_process->GetTarget().GetStatistics().SetMismatchedCoredumpModuleCount(
      mismatched_module_count);
}

void DynamicLoaderDumpWithModuleList::LoadAllModules(
    LoadModuleCallback callback) {

  if (m_rendezvous.Resolve()) {
    // The rendezvous class doesn't enumerate the main module, so track that
    // ourselves here.
    ModuleSP executable = GetTargetExecutable();
    if (executable)
      m_loaded_modules[executable] = m_rendezvous.GetLinkMapAddress();

    DYLDRendezvous::iterator I;
    DYLDRendezvous::iterator E;
    for (I = m_rendezvous.begin(), E = m_rendezvous.end(); I != E; ++I) {
      // Module size has to be > 0 to be valid.
      addr_t module_size = 1;
      std::optional<const LoadedModuleInfoList::LoadedModuleInfo> mod_info_opt =
          GetModuleInfo(I->base_addr);
      if (mod_info_opt.has_value())
        (*mod_info_opt).get_size(module_size);
      callback(I->file_spec.GetPath(), I->base_addr, module_size, I->link_addr);
    }

    DetectModuleListMismatch();
  } else {
    Log *log = GetLog(LLDBLog::DynamicLoader);
    LLDB_LOGF(
        log,
        "DynamicLoaderDumpWithModuleList::%s unable to resolve POSIX DYLD "
        "rendezvous address. Fallback to try GetLoadedModuleList().",
        __FUNCTION__);

    llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
        m_process->GetLoadedModuleList();
    if (!module_info_list_ep) {
      LLDB_LOGF(log,
                "DynamicLoaderDumpWithModuleList::%s fail to get module list "
                "from GetLoadedModuleList().",
                __FUNCTION__);
      llvm::consumeError(module_info_list_ep.takeError());
      return;
    }

    const LoadedModuleInfoList &module_info_list = *module_info_list_ep;
    for (const LoadedModuleInfoList::LoadedModuleInfo &mod_info :
         module_info_list.m_list) {
      addr_t base_addr, module_size;
      std::string name;
      if (!mod_info.get_base(base_addr) || !mod_info.get_name(name) ||
          !mod_info.get_size(module_size) || !ShouldLoadModule(name))
        continue;

      callback(SanitizeName(name), base_addr, module_size, /*link_map_addr=*/0);
    }
  }
}

bool DynamicLoaderDumpWithModuleList::ShouldLoadModule(
    const std::string &module_name) {
  // Use a regular expression to match /dev/* path
  static const std::regex pattern("^/dev/(?!shm.*).*$");
  return !std::regex_match(module_name, pattern);
}

std::string
DynamicLoaderDumpWithModuleList::SanitizeName(const std::string &input) {
  // Use a regular expression to match and remove the parenthesized substring
  static const std::regex pattern("\\s*\\(\\S+\\)\\s*$");
  return std::regex_replace(input, pattern, "");
}

void DynamicLoaderDumpWithModuleList::DidAttach() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderDumpWithModuleList::%s() pid %" PRIu64,
            __FUNCTION__,
            m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);
  m_auxv = std::make_unique<AuxVector>(m_process->GetAuxvData());

  ModuleSP executable_sp = GetTargetExecutable();
  if (executable_sp) {
    m_rendezvous.UpdateExecutablePath();
    UpdateLoadedSections(executable_sp, LLDB_INVALID_ADDRESS, /*load_offset=*/0,
                         true);
  }
  EvalSpecialModulesStatus();
  LoadVDSO();

  ModuleList module_list;
  
  // Collect module information first
  std::vector<ModuleInfo> module_info_list;
  LoadAllModules([&](const std::string &name, addr_t base_addr,
                     addr_t module_size, addr_t link_map_addr) {
    // vdso module has already been loaded.
    if (base_addr == m_vdso_base)
      return;
    module_info_list.emplace_back(ModuleInfo{name, base_addr, module_size, link_map_addr});
  });

  // Load modules in parallel or sequentially based on target setting
  auto load_module_fn = [this, &module_list, &log](const ModuleInfo &module_info) {
    const std::string &name = module_info.name;
    addr_t base_addr = module_info.base_addr;
    addr_t module_size = module_info.module_size;
    addr_t link_map_addr = module_info.link_map_addr;

    FileSpec file(name, m_process->GetTarget().GetArchitecture().GetTriple());
    const bool base_addr_is_offset = false;
    ModuleSP module_sp = DynamicLoader::LoadModuleAtAddress(
        file, link_map_addr, base_addr, base_addr_is_offset);
    if (module_sp.get()) {
      LLDB_LOGF(log, "LoadAllCurrentModules loading module at 0x%lX: %s",
                base_addr, name.c_str());
      // Note: in a multi-threaded environment, these module lists may be
      // appended to out-of-order. This is fine, since there's no
      // expectation for `module_list` to be in any particular order, and
      // appending to the module list is thread-safe.
      module_list.Append(module_sp);
    } else {
      LLDB_LOGF(
          log,
          "DynamicLoaderDumpWithModuleList::%s unable to locate the matching "
          "object file %s, creating a placeholder module at 0x%" PRIx64,
          __FUNCTION__, name.c_str(), base_addr);

      ModuleSpec module_spec(file, m_process->GetTarget().GetArchitecture());
      module_sp = Module::CreateModuleFromObjectFile<ObjectFilePlaceholder>(
          module_spec, base_addr, module_size);
      UpdateLoadedSections(module_sp, link_map_addr, base_addr,
                           base_addr_is_offset);
      m_process->GetTarget().GetImages().Append(module_sp, /*notify*/ true);
    }
    SetLoadedModule(module_sp, link_map_addr);
  };

  if (m_process->GetTarget().GetParallelModuleLoad()) {
    llvm::ThreadPoolTaskGroup task_group(Debugger::GetThreadPool());
    for (const auto &module_info : module_info_list)
      task_group.async(load_module_fn, module_info);
    task_group.wait();
  } else {
    for (const auto &module_info : module_info_list)
      load_module_fn(module_info);
  }

  m_process->GetTarget().ModulesDidLoad(module_list);
}

void DynamicLoaderDumpWithModuleList::EvalSpecialModulesStatus() {
  if (std::optional<uint64_t> vdso_base =
          m_auxv->GetAuxValue(AuxVector::AUXV_AT_SYSINFO_EHDR))
    m_vdso_base = *vdso_base;
}

void DynamicLoaderDumpWithModuleList::LoadVDSO() {
  if (m_vdso_base == LLDB_INVALID_ADDRESS)
    return;

  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "Loading vdso at 0x%lx", m_vdso_base);

  MemoryRegionInfo info;
  Status status = m_process->GetMemoryRegionInfo(m_vdso_base, info);
  if (status.Fail()) {
    LLDB_LOG(log, "Failed to get vdso region info: {0}", status);
    return;
  }

  FileSpec file("[vdso]");
  if (ModuleSP module_sp = m_process->ReadModuleFromMemory(
          file, m_vdso_base, info.GetRange().GetByteSize())) {
    UpdateLoadedSections(module_sp, LLDB_INVALID_ADDRESS, m_vdso_base, false);
    m_process->GetTarget().GetImages().AppendIfNeeded(module_sp);
  }
}

lldb_private::Status DynamicLoaderDumpWithModuleList::CanLoadImage() {
  return Status();
}

// TODO: refactor and merge this implementation with
// DynamicLoaderPOSIXDYLD::GetThreadLocalData
lldb::addr_t DynamicLoaderDumpWithModuleList::GetThreadLocalData(
    const lldb::ModuleSP module_sp, const lldb::ThreadSP thread,
    lldb::addr_t tls_file_addr) {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  std::optional<addr_t> link_map_addr_opt = GetLoadedModuleLinkAddr(module_sp);
  if (!link_map_addr_opt.has_value()) {
    LLDB_LOGF(
        log, "GetThreadLocalData error: module(%s) not found in loaded modules",
        module_sp->GetObjectName().AsCString());
    return LLDB_INVALID_ADDRESS;
  }

  addr_t link_map = link_map_addr_opt.value();
  if (link_map == LLDB_INVALID_ADDRESS || link_map == 0) {
    LLDB_LOGF(log,
              "GetThreadLocalData error: invalid link map address=0x%" PRIx64,
              link_map);
    return LLDB_INVALID_ADDRESS;
  }

  const DYLDRendezvous::ThreadInfo &metadata = m_rendezvous.GetThreadInfo();
  if (!metadata.valid) {
    LLDB_LOGF(log,
              "GetThreadLocalData error: fail to read thread info metadata");
    return LLDB_INVALID_ADDRESS;
  }

  LLDB_LOGF(log,
            "GetThreadLocalData info: link_map=0x%" PRIx64
            ", thread info metadata: "
            "modid_offset=0x%" PRIx32 ", dtv_offset=0x%" PRIx32
            ", tls_offset=0x%" PRIx32 ", dtv_slot_size=%" PRIx32 "\n",
            link_map, metadata.modid_offset, metadata.dtv_offset,
            metadata.tls_offset, metadata.dtv_slot_size);

  // Get the thread pointer.
  addr_t tp = thread->GetThreadPointer();
  if (tp == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log, "GetThreadLocalData error: fail to read thread pointer");
    return LLDB_INVALID_ADDRESS;
  }

  // Find the module's modid.
  int modid_size = 4; // FIXME(spucci): This isn't right for big-endian 64-bit
  int64_t modid = ReadUnsignedIntWithSizeInBytes(
      link_map + metadata.modid_offset, modid_size);
  if (modid == -1) {
    LLDB_LOGF(log, "GetThreadLocalData error: fail to read modid");
    return LLDB_INVALID_ADDRESS;
  }

  // Lookup the DTV structure for this thread.
  addr_t dtv_ptr = tp + metadata.dtv_offset;
  addr_t dtv = ReadPointer(dtv_ptr);
  if (dtv == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log, "GetThreadLocalData error: fail to read dtv");
    return LLDB_INVALID_ADDRESS;
  }

  // Find the TLS block for this module.
  addr_t dtv_slot = dtv + metadata.dtv_slot_size * modid;
  addr_t tls_block = ReadPointer(dtv_slot + metadata.tls_offset);

  LLDB_LOGF(log,
            "DynamicLoaderDumpWithModuleList::Performed TLS lookup: "
            "module=%s, link_map=0x%" PRIx64 ", tp=0x%" PRIx64
            ", modid=%" PRId64 ", tls_block=0x%" PRIx64 "\n",
            module_sp->GetObjectName().AsCString(""), link_map, tp,
            (int64_t)modid, tls_block);

  if (tls_block == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log, "GetThreadLocalData error: fail to read tls_block");
    return LLDB_INVALID_ADDRESS;
  } else
    return tls_block + tls_file_addr;
}
