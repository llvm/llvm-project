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
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

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
void DynamicLoaderGDBRemoteGPU::DidAttach() { LoadModulesFromGDBServer(true); }

/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
void DynamicLoaderGDBRemoteGPU::DidLaunch() { LoadModulesFromGDBServer(true); }

bool DynamicLoaderGDBRemoteGPU::HandleStopReasonDynammicLoader() { 
  return LoadModulesFromGDBServer(false);
}

bool DynamicLoaderGDBRemoteGPU::LoadModulesFromGDBServer(bool full) {
  Log *log = GetLog(LLDBLog::DynamicLoader);

  ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(m_process);
  ModuleList loaded_module_list;
  GPUDynamicLoaderArgs args;
  args.full = full;
  Target &target = m_process->GetTarget();
  std::optional<GPUDynamicLoaderResponse> response =
      gdb_process->GetGDBRemote().GetGPUDynamicLoaderLibraryInfos(args);
  if (!response) {
    LLDB_LOG(log, "Failed to get dynamic loading info from GDB server");
    return false;
  }
  for (const GPUDynamicLoaderLibraryInfo &info : response->library_infos) {
    std::shared_ptr<DataBufferHeap> data_sp;
    // Read the object file from memory if requested.
    if (info.native_memory_address && info.native_memory_size) {
      LLDB_LOG(log, "Reading \"{0}\" from memory at {1:x}", info.pathname, 
               *info.native_memory_address);
      data_sp = std::make_shared<DataBufferHeap>(*info.native_memory_size, 0);
      Status error;
      // TODO: we are assuming we can read the memory from the GPU process
      // since the memory is shared with the host process.
      const size_t bytes_read = m_process->ReadMemory(
          *info.native_memory_address, data_sp->GetBytes(), 
          data_sp->GetByteSize(), error);
      if (bytes_read != *info.native_memory_size) {
        LLDB_LOG(log, "Failed to read \"{0}\" from memory at {1:x}: {2}", 
                 info.pathname, *info.native_memory_address, error);
        data_sp.reset();
      }
    }
    // Extract the UUID if available.
    UUID uuid;
    if (info.uuid_str)
      uuid.SetFromStringRef(*info.uuid_str);
    // Create a module specification from the info we got.
    ModuleSpec module_spec(FileSpec(info.pathname), uuid, data_sp);
    if (info.file_offset)
      module_spec.SetObjectOffset(*info.file_offset);
    if (info.file_size)
      module_spec.SetObjectSize(*info.file_size);
    // Get or create the module from the module spec.
    ModuleSP module_sp = target.GetOrCreateModule(module_spec, 
                                                  /*notify=*/true);
    if (module_sp) {
      LLDB_LOG(log, "Created module for \"{0}\": {1:x}", 
               info.pathname, module_sp.get());
      bool changed = false;
      if (info.load_address) {
        LLDB_LOG(log, "Setting load address for module \"{0}\" to {1:x}", 
                 info.pathname, *info.load_address);

        module_sp->SetLoadAddress(target, *info.load_address, 
                                  /*value_is_offset=*/true , changed);
      } else if (!info.loaded_sections.empty()) {    
        
        // Set the load address of the module to the first loaded section.
        bool warn_multiple = true;
        for (const GPUSectionInfo &sect : info.loaded_sections) {
          if (sect.names.empty())
            continue;
          // Find the section by name using the names specified. If there is 
          // only on name, them find it. If there are multiple names, the top
          // most section names comes first and then we find child sections
          // by name within the previous section.
          SectionSP section_sp;
          for (uint32_t i=0; i<sect.names.size(); ++i) {
            ConstString name(sect.names[i]);
            if (section_sp)
              section_sp = section_sp->GetChildren().FindSectionByName(name);
            else
              section_sp = module_sp->GetSectionList()->FindSectionByName(name);
            if (!section_sp)
              break;
          }
          if (section_sp) {
            LLDB_LOG(log, "Loading module \"{0}\" section \"{1} to {2:x}", 
                     info.pathname, section_sp->GetName(), sect.load_address);
            changed = target.SetSectionLoadAddress(section_sp, 
                                                   sect.load_address, 
                                                   warn_multiple);
          } else {
            LLDB_LOG(log, "Failed to find section \"{0}\"", 
                     section_sp->GetName());
          }
        }
      }
      if (changed) {
        LLDB_LOG(log, "Module \"{0}\" was loaded, notifying target", 
                 info.pathname);
        loaded_module_list.AppendIfNeeded(module_sp);            
      }
    }
  }
  target.ModulesDidLoad(loaded_module_list);
  return true; // Handled the request.
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
