//===-- DynamicLoaderWindowsDYLD.cpp --------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderWindowsDYLD.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoaderWindowsDYLD::DynamicLoaderWindowsDYLD(Process *process)
    : DynamicLoader(process) {}

DynamicLoaderWindowsDYLD::~DynamicLoaderWindowsDYLD() {}

void DynamicLoaderWindowsDYLD::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderWindowsDYLD::Terminate() {}

ConstString DynamicLoaderWindowsDYLD::GetPluginNameStatic() {
  static ConstString g_plugin_name("windows-dyld");
  return g_plugin_name;
}

const char *DynamicLoaderWindowsDYLD::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that watches for shared library "
         "loads/unloads in Windows processes.";
}

DynamicLoader *DynamicLoaderWindowsDYLD::CreateInstance(Process *process,
                                                        bool force) {
  bool should_create = force;
  if (!should_create) {
    const llvm::Triple &triple_ref =
        process->GetTarget().GetArchitecture().GetTriple();
    if (triple_ref.getOS() == llvm::Triple::Win32)
      should_create = true;
  }

  if (should_create)
    return new DynamicLoaderWindowsDYLD(process);

  return nullptr;
}

void DynamicLoaderWindowsDYLD::DidAttach() {}

void DynamicLoaderWindowsDYLD::DidLaunch() {}

Error DynamicLoaderWindowsDYLD::CanLoadImage() { return Error(); }

ConstString DynamicLoaderWindowsDYLD::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t DynamicLoaderWindowsDYLD::GetPluginVersion() { return 1; }

ThreadPlanSP
DynamicLoaderWindowsDYLD::GetStepThroughTrampolinePlan(Thread &thread,
                                                       bool stop) {
  return ThreadPlanSP();
}
