//===-- DynamicLoaderWasmDYLD.cpp --------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderWasmDYLD.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "llvm/ADT/Triple.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoaderWasmDYLD::DynamicLoaderWasmDYLD(Process *process)
    : DynamicLoader(process) {}

DynamicLoaderWasmDYLD::~DynamicLoaderWasmDYLD() {}

void DynamicLoaderWasmDYLD::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderWasmDYLD::Terminate() {}

ConstString DynamicLoaderWasmDYLD::GetPluginNameStatic() {
  static ConstString g_plugin_name("wasm-dyld");
  return g_plugin_name;
}

const char *DynamicLoaderWasmDYLD::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that watches for shared library "
         "loads/unloads in WebAssembly engines.";
}

DynamicLoader *DynamicLoaderWasmDYLD::CreateInstance(Process *process,
                                                     bool force) {
  bool should_create = force;
  if (!should_create) {
    const llvm::Triple &triple_ref =
        process->GetTarget().GetArchitecture().GetTriple();
    if (triple_ref.getArch() == llvm::Triple::wasm32)
      should_create = true;
  }

  if (should_create)
    return new DynamicLoaderWasmDYLD(process);

  return nullptr;
}

ModuleSP DynamicLoaderWasmDYLD::LoadModuleAtAddress(const FileSpec &file,
                                                    addr_t link_map_addr,
                                                    addr_t base_addr,
                                                    bool base_addr_is_offset) {
  Target &target = m_process->GetTarget();
  ModuleList &modules = target.GetImages();
  ModuleSpec module_spec(file, target.GetArchitecture());
  ModuleSP module_sp;

  if ((module_sp = modules.FindFirstModule(module_spec))) {
    UpdateLoadedSections(module_sp, link_map_addr, base_addr,
                         base_addr_is_offset);
    return module_sp;
  }

  if ((module_sp = m_process->ReadModuleFromMemory(file, base_addr))) {
    UpdateLoadedSections(module_sp, link_map_addr, base_addr, false);
    target.GetImages().AppendIfNeeded(module_sp);
  }

  return module_sp;
}

void DynamicLoaderWasmDYLD::DidAttach() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  LLDB_LOGF(log, "DynamicLoaderWasmDYLD::%s()", __FUNCTION__);

  // Ask the process for the list of loaded WebAssembly modules.
  auto error = m_process->LoadModules();
  LLDB_LOG_ERROR(log, std::move(error), "Couldn't load modules: {0}");

  ModuleList loaded_module_list;
  const ModuleList &module_list = m_process->GetTarget().GetImages();
  const size_t num_modules = module_list.GetSize();
    for (uint32_t idx = 0; idx < num_modules; ++idx) {
      ModuleSP module_sp(module_list.GetModuleAtIndexUnlocked(idx));
      ObjectFile *image_object_file = module_sp->GetObjectFile();
      lldb::addr_t code_load_address =
          image_object_file->GetBaseAddress().GetOffset();
      lldb::addr_t image_load_address =
          image_object_file->GetBaseAddress().GetOffset() & 0xffffffff00000000;
      if (module_sp) {
        bool changed = false;
        if (image_object_file) {
          SectionList *section_list = image_object_file->GetSectionList();
          if (section_list) {
            // Fixes the section load address for each section.
            const size_t num_sections = section_list->GetSize();
            for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
              SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
              if (section_sp) {
                // Code section load address is offsetted by the code section
                // offset in the Wasm module.
                if (section_sp->GetName() == "code") {
                  if (m_process->GetTarget().SetSectionLoadAddress(
                          section_sp,
                          code_load_address | section_sp->GetFileAddress())) {
                    changed = true;
                  }
                } else {
                  if (m_process->GetTarget().SetSectionLoadAddress(
                          section_sp,
                          image_load_address | section_sp->GetFileAddress())) {
                    changed = true;
                  }
                }
              }
            }
          }
        }

        if (changed)
          loaded_module_list.AppendIfNeeded(module_sp);
      }
    }

    m_process->GetTarget().ModulesDidLoad(loaded_module_list);
}

void DynamicLoaderWasmDYLD::DidLaunch() {}

Status DynamicLoaderWasmDYLD::CanLoadImage() { return Status(); }

ConstString DynamicLoaderWasmDYLD::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t DynamicLoaderWasmDYLD::GetPluginVersion() { return 1; }

ThreadPlanSP DynamicLoaderWasmDYLD::GetStepThroughTrampolinePlan(Thread &thread,
                                                       bool stop) {
  auto arch = m_process->GetTarget().GetArchitecture();
  if (arch.GetMachine() != llvm::Triple::wasm32) {
    return ThreadPlanSP();
  }

  // TODO(paolosev) - What should we do here? 
  return ThreadPlanSP();
}
