//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderAcceleratorGDBRemote.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

LLDB_PLUGIN_DEFINE(DynamicLoaderAcceleratorGDBRemote)

DynamicLoader *
DynamicLoaderAcceleratorGDBRemote::CreateInstance(Process *process,
                                                  bool force) {
  // The library list comes from the accelerator's GDB server.
  if (process->GetPluginName() != ProcessGDBRemote::GetPluginNameStatic())
    return nullptr;
  if (force)
    return new DynamicLoaderAcceleratorGDBRemote(process);

  const llvm::Triple &triple =
      process->GetTarget().GetArchitecture().GetTriple();
  if (triple.isAMDGPU() || triple.isNVPTX())
    return new DynamicLoaderAcceleratorGDBRemote(process);
  return nullptr;
}

DynamicLoaderAcceleratorGDBRemote::DynamicLoaderAcceleratorGDBRemote(
    Process *process)
    : DynamicLoader(process) {}

void DynamicLoaderAcceleratorGDBRemote::DidAttach() {
  LoadModulesFromGDBServer(/*full=*/true);
}

void DynamicLoaderAcceleratorGDBRemote::DidLaunch() {
  LoadModulesFromGDBServer(/*full=*/true);
}

bool DynamicLoaderAcceleratorGDBRemote::LoadModulesFromGDBServer(bool full) {
  Log *log = GetLog(LLDBLog::DynamicLoader);

  // CreateInstance ensures that m_process is a ProcessGDBRemote.
  ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(m_process);
  AcceleratorDynamicLoaderArgs args;
  args.full = full;

  Target &target = m_process->GetTarget();
  ModuleList loaded_module_list;
  std::optional<AcceleratorDynamicLoaderResponse> response =
      gdb_process->GetGDBRemote().GetAcceleratorDynamicLoaderLibraryInfos(args);
  if (!response) {
    LLDB_LOG(log, "failed to get dynamic loader info from the GDB server");
    return false;
  }

  for (const AcceleratorDynamicLoaderLibraryInfo &info :
       response->library_infos) {
    UUID uuid;
    if (info.uuid_str)
      uuid.SetFromStringRef(*info.uuid_str);

    // Either a whole file, or a slice of a containing file.
    ModuleSpec module_spec(FileSpec(info.pathname), uuid);
    if (info.file_offset)
      module_spec.SetObjectOffset(*info.file_offset);
    if (info.file_size)
      module_spec.SetObjectSize(*info.file_size);

    if (!info.load) {
      ModuleList matching_module_list;
      target.GetImages().FindModules(module_spec, matching_module_list);
      matching_module_list.ForEach(
          [this](const ModuleSP &module_sp) -> IterationAction {
            UnloadSections(module_sp);
            return IterationAction::Continue;
          });
      continue;
    }

    ModuleSP module_sp = target.GetOrCreateModule(module_spec, /*notify=*/true);
    if (!module_sp)
      continue;

    bool changed = false;
    if (info.load_address) {
      module_sp->SetLoadAddress(target, *info.load_address,
                                /*value_is_offset=*/true, changed);
    } else if (!info.loaded_sections.empty()) {
      for (const AcceleratorSectionInfo &sect : info.loaded_sections) {
        if (sect.names.empty())
          continue;
        SectionSP section_sp;
        for (const std::string &name : sect.names) {
          ConstString section_name(name);
          if (section_sp)
            section_sp =
                section_sp->GetChildren().FindSectionByName(section_name);
          else
            section_sp =
                module_sp->GetSectionList()->FindSectionByName(section_name);
          if (!section_sp)
            break;
        }
        if (section_sp)
          changed |= target.SetSectionLoadAddress(section_sp, sect.load_address,
                                                  /*warn_multiple=*/true);
      }
    } else {
      // No slide: load at the file addresses.
      module_sp->SetLoadAddress(target, 0, /*value_is_offset=*/true, changed);
    }

    if (changed)
      loaded_module_list.AppendIfNeeded(module_sp);
  }

  target.ModulesDidLoad(loaded_module_list);
  return true;
}

ThreadPlanSP DynamicLoaderAcceleratorGDBRemote::GetStepThroughTrampolinePlan(
    Thread &thread, bool stop_others) {
  return ThreadPlanSP();
}

Status DynamicLoaderAcceleratorGDBRemote::CanLoadImage() {
  return Status::FromErrorString("can't load images on accelerator targets");
}

void DynamicLoaderAcceleratorGDBRemote::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderAcceleratorGDBRemote::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef
DynamicLoaderAcceleratorGDBRemote::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that gets shared library loads/unloads from "
         "an lldb-server accelerator plugin.";
}
