//===-- DynamicLoaderDumpWithModuleList.cpp-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Main header include
#include "DynamicLoaderDumpWithModuleList.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "Plugins/ObjectFile/Placeholder/ObjectFilePlaceholder.h"

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
    : DynamicLoader(process) {}

DynamicLoaderDumpWithModuleList::~DynamicLoaderDumpWithModuleList() {}

void DynamicLoaderDumpWithModuleList::DidAttach() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderDumpWithModuleList::%s() pid %" PRIu64,
            __FUNCTION__,
            m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);

  // Curently only used by ProcessELFCore who will return the results of the
  // NT_FILE list from ProcessELFCore::GetLoadedModuleList.
  llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
      m_process->GetLoadedModuleList();
  if (!module_info_list_ep) {
    // TODO: log failure.
    llvm::consumeError(module_info_list_ep.takeError());
    return;
  }

  ModuleList module_list;
  const LoadedModuleInfoList &module_info_list = *module_info_list_ep;
  for (const LoadedModuleInfoList::LoadedModuleInfo &modInfo :
       module_info_list.m_list) {
    addr_t base_addr, module_size;
    std::string name;
    if (!modInfo.get_base(base_addr) || !modInfo.get_name(name) ||
        !modInfo.get_size(module_size))
      continue;

    addr_t link_map_addr = 0;
    FileSpec file(name, m_process->GetTarget().GetArchitecture().GetTriple());
    const bool base_addr_is_offset = false;
    ModuleSP module_sp = DynamicLoader::LoadModuleAtAddress(
        file, link_map_addr, base_addr, base_addr_is_offset);
    if (module_sp.get()) {
      LLDB_LOG(log, "LoadAllCurrentModules loading module: {0}", name.c_str());
      module_list.Append(module_sp);
    } else {
      Log *log = GetLog(LLDBLog::DynamicLoader);
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
  }
  m_process->GetTarget().ModulesDidLoad(module_list);
}

lldb_private::Status DynamicLoaderDumpWithModuleList::CanLoadImage() {
  return Status();
}
