//===-- InstrumentationRuntime.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "lldb/Target/InstrumentationRuntime.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

void InstrumentationRuntime::ModulesDidLoad(
    lldb_private::ModuleList &module_list, lldb_private::Process *process,
    InstrumentationRuntimeCollection &runtimes) {
  for (auto &cbs : PluginManager::GetInstrumentationRuntimeCallbacks()) {
    InstrumentationRuntimeType type = cbs.get_type_callback();
    if (runtimes.find(type) == runtimes.end())
      runtimes[type] = cbs.create_callback(process->shared_from_this());
  }
}

void InstrumentationRuntime::ModulesDidLoad(
    lldb_private::ModuleList &module_list) {
  if (!IsEnabled())
    return;

  if (IsActive())
    return;

  if (GetRuntimeModuleSP()) {
    Activate();
    return;
  }

  module_list.ForEach([this](const lldb::ModuleSP module_sp) {
    const FileSpec &file_spec = module_sp->GetFileSpec();
    if (!file_spec)
      return IterationAction::Continue;

    const RegularExpression &runtime_regex = GetPatternForRuntimeLibrary();
    if (MatchAllModules() ||
        runtime_regex.Execute(file_spec.GetFilename().GetCString()) ||
        module_sp->IsExecutable()) {
      if (CheckIfRuntimeIsValid(module_sp)) {
        SetRuntimeModuleSP(module_sp);
        Activate();
        if (!IsActive())
          SetRuntimeModuleSP({}); // Don't cache module if activation failed.
        return IterationAction::Stop;
      }
    }

    return IterationAction::Continue;
  });
}

llvm::Error InstrumentationRuntime::Enable() {
  SetEnabled(true);

  if (IsActive())
    return llvm::Error::success();

  // Fast path. During a previous time when the plugin was active the relevant
  // runtime module was found so we can just activate immediately.
  // FIXME: What if the module was unloaded via dlclose()?
  if (GetRuntimeModuleSP()) {
    Activate();
    return llvm::Error::success();
  }

  // Slow path. The plugin has never found the relevant runtime module in the
  // past so pretend the current list of modules in the target were just loaded
  // to give the plugin a chance to activate.
  if (ProcessSP process_sp = GetProcessSP()) {
    ModuleList module_list;
    for (const auto &module_sp :
         process_sp->GetTarget().GetImages().Modules()) {
      module_list.Append(module_sp);
    }
    // Give the plugin a chance to activate.
    ModulesDidLoad(module_list);
  }
  return llvm::Error::success();
}

llvm::Error InstrumentationRuntime::Disable() {
  if (IsActive())
    Deactivate();

  if (IsActive())
    return llvm::createStringError(
        "failed to deactivate instrumentation runtime");

  SetEnabled(false);
  return llvm::Error::success();
}

lldb::ThreadCollectionSP
InstrumentationRuntime::GetBacktracesFromExtendedStopInfo(
    StructuredData::ObjectSP info) {
  return std::make_shared<ThreadCollection>();
}
