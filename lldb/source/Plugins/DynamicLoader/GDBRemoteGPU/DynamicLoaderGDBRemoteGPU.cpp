//===-- DynamicLoaderGDBRemoteGPU.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"

#include "DynamicLoaderGDBRemoteGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

LLDB_PLUGIN_DEFINE(DynamicLoaderGDBRemoteGPU)

// Create an instance of this class. This function is filled into the plugin
// info class that gets handed out by the plugin factory and allows the lldb to
// instantiate an instance of this class.
DynamicLoader *DynamicLoaderGDBRemoteGPU::CreateInstance(Process *process,
                                                         bool force) {
  // Only create an instance if the clients ask for this plugin by name. This
  // plugin will be created by the ProcessGDBRemote class by asking for it by
  // name.
  if (force)
    return new DynamicLoaderGDBRemoteGPU(process);
  return nullptr;
}

// Constructor
DynamicLoaderGDBRemoteGPU::DynamicLoaderGDBRemoteGPU(Process *process)
    : DynamicLoader(process) {}

/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
void DynamicLoaderGDBRemoteGPU::DidAttach() { LoadModules(true); }

/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
void DynamicLoaderGDBRemoteGPU::DidLaunch() { LoadModules(true); }

bool DynamicLoaderGDBRemoteGPU::HandleStopReasonDynammicLoader() { 
  LoadModules(false);
  return GetStopWhenImagesChange();
}

void DynamicLoaderGDBRemoteGPU::LoadModules(bool full) {
  
  ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(m_process);
  ModuleList loaded_module_list;
  GPUDynamicLoaderArgs args;
  args.full = full;
  std::optional<GPUDynamicLoaderResponse> response =
      gdb_process->GetGDBRemote().GetGPUDynamicLoaderLibraryInfos(args);
  if (response) {
    for (const GPUDynamicLoaderLibraryInfo &info : response->library_infos) {
      UUID uuid;
      std::shared_ptr<DataBufferHeap> data_sp;
      if (info.native_memory_address && info.native_memory_size) {
        data_sp = std::make_shared<DataBufferHeap>(*info.native_memory_size, 0);
        Status error;
        // TODO: we are assuming we can read the memory from the GPU process
        // since the memory is shared with the host process.
        const size_t bytes_read = m_process->ReadMemory(
            *info.native_memory_address, data_sp->GetBytes(), 
            data_sp->GetByteSize(), error);
        if (bytes_read != *info.native_memory_size)
          data_sp.reset();
      }
        
      if (info.uuid_str)
        uuid.SetFromStringRef(*info.uuid_str);
      ModuleSpec module_spec(FileSpec(info.pathname), uuid, data_sp);
      if (info.file_offset)
        module_spec.SetObjectOffset(*info.file_offset);
      if (info.file_size)
        module_spec.SetObjectSize(*info.file_size);

      ModuleSP module_sp = m_process->GetTarget().GetOrCreateModule(module_spec, 
                                                    /*notify=*/true);
      if (module_sp) {
      }
    }
  }
#if 0
  ModuleList loaded_module_list;

  Target &target = m_process->GetTarget();
  for (ModuleSP module_sp : module_list.Modules()) {
    if (module_sp) {
      bool changed = false;
      bool no_load_addresses = true;
      // If this module has a section with a load address set in
      // the target, assume all necessary work is already done. There
      // may be sections without a load address set intentionally
      // and we don't want to mutate that.
      // For a module with no load addresses set, set the load addresses
      // to slide == 0, the same as the file addresses, in the target.
      ObjectFile *image_object_file = module_sp->GetObjectFile();
      if (image_object_file) {
        SectionList *section_list = image_object_file->GetSectionList();
        if (section_list) {
          const size_t num_sections = section_list->GetSize();
          for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
            SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
            if (section_sp) {
              if (target.GetSectionLoadAddress(section_sp) !=
                  LLDB_INVALID_ADDRESS) {
                no_load_addresses = false;
                break;
              }
            }
          }
        }
      }
      if (no_load_addresses)
        module_sp->SetLoadAddress(target, 0, true /*value_is_offset*/, changed);

      if (changed)
        loaded_module_list.AppendIfNeeded(module_sp);
    }
  }

  target.ModulesDidLoad(loaded_module_list);
  #endif
}

ThreadPlanSP
DynamicLoaderGDBRemoteGPU::GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others) {
  return ThreadPlanSP();
}

Status DynamicLoaderGDBRemoteGPU::CanLoadImage() {
  return Status::FromErrorString(
      "can't load images on GPU targets");
}

void DynamicLoaderGDBRemoteGPU::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderGDBRemoteGPU::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef DynamicLoaderGDBRemoteGPU::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in for GPU targets that uses GDB remote packets "
         "tailored for GPUs to get the library load and unload information from"
         " the lldb-server GPU plug-in GDB server connection.";
}
