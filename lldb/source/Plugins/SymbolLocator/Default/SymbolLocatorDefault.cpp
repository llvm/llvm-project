//===-- SymbolLocatorDefault.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorDefault.h"

#include <cstring>
#include <optional>

#include "Plugins/ObjectFile/wasm/ObjectFileWasm.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/LocateSymbolFile.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"

// From MacOSX system header "mach/machine.h"
typedef int cpu_type_t;
typedef int cpu_subtype_t;

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorDefault)

SymbolLocatorDefault::SymbolLocatorDefault() : SymbolLocator() {}

void SymbolLocatorDefault::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                LocateExecutableObjectFile);
}

void SymbolLocatorDefault::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolLocatorDefault::GetPluginDescriptionStatic() {
  return "Default symbol locator.";
}

SymbolLocator *SymbolLocatorDefault::CreateInstance() {
  return new SymbolLocatorDefault();
}

std::optional<ModuleSpec> SymbolLocatorDefault::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  const FileSpec &exec_fspec = module_spec.GetFileSpec();
  const ArchSpec *arch = module_spec.GetArchitecturePtr();
  const UUID *uuid = module_spec.GetUUIDPtr();
  LLDB_SCOPED_TIMERF(
      "LocateExecutableObjectFile (file = %s, arch = %s, uuid = %p)",
      exec_fspec ? exec_fspec.GetFilename().AsCString("<NULL>") : "<NULL>",
      arch ? arch->GetArchitectureName() : "<NULL>", (const void *)uuid);

  ModuleSpecList module_specs;
  ModuleSpec matched_module_spec;
  if (exec_fspec &&
      ObjectFile::GetModuleSpecifications(exec_fspec, 0, 0, module_specs) &&
      module_specs.FindMatchingModuleSpec(module_spec, matched_module_spec)) {
    ModuleSpec result;
    result.GetFileSpec() = exec_fspec;
    return result;
  }

  return {};
}
