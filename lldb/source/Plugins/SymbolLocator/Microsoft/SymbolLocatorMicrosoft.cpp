//===-- SymbolLocatorMicrosoft.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorMicrosoft.h"

#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/UUID.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorMicrosoft)

namespace {

#define LLDB_PROPERTIES_symbollocatormicrosoft
#include "SymbolLocatorMicrosoftProperties.inc"

enum {
#define LLDB_PROPERTIES_symbollocatormicrosoft
#include "SymbolLocatorMicrosoftPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() {
    return SymbolLocatorMicrosoft::GetPluginNameStatic();
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbollocatormicrosoft_properties_def);
  }

  Args GetURLs() const {
    Args urls;
    m_collection_sp->GetPropertyAtIndexAsArgs(ePropertySymStoreURLs, urls);
    return urls;
  }
};

} // namespace

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

SymbolLocatorMicrosoft::SymbolLocatorMicrosoft() : SymbolLocator() {}

void SymbolLocatorMicrosoft::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(
        GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
        nullptr, LocateExecutableSymbolFile, nullptr,
        nullptr, SymbolLocatorMicrosoft::DebuggerInitialize);
  });
}

void SymbolLocatorMicrosoft::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolLocatorPlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolLocatorPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the Microsoft Symbol Locator plug-in.",
        is_global_setting);
  }
}

void SymbolLocatorMicrosoft::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolLocatorMicrosoft::GetPluginDescriptionStatic() {
  return "Symbol locator for PDB in Microsoft SymStore";
}

SymbolLocator *SymbolLocatorMicrosoft::CreateInstance() {
  return new SymbolLocatorMicrosoft();
}

static llvm::StringRef getFileName(const ModuleSpec &module_spec,
                                   std::string url_path) {
  // Check if the URL path requests an executable file or a symbol file
  bool is_executable = url_path.find("debuginfo") == std::string::npos;
  if (is_executable)
    return module_spec.GetFileSpec().GetFilename().GetStringRef();
  llvm::StringRef symbol_file =
      module_spec.GetSymbolFileSpec().GetFilename().GetStringRef();
  // Remove llvmcache- prefix and hash, keep origin file name
  if (symbol_file.starts_with("llvmcache-")) {
    size_t pos = symbol_file.rfind('-');
    if (pos != llvm::StringRef::npos) {
      symbol_file = symbol_file.substr(pos + 1);
    }
  }
  return symbol_file;
}

// LLDB stores PDB identity as a 20-byte UUID:
//   bytes  0-15  GUID in big-endian canonical form
//   bytes 16-19  Age as big-endian uint32
//
// The symsrv key is: <GUID-uppercase-hex><decimal-age>
// e.g. "A0586BA32F284960B536A424603C76891" (age 1)
static std::string formatSymStoreKey(const UUID &uuid) {
  llvm::ArrayRef<uint8_t> bytes = uuid.GetBytes();
  uint32_t age = llvm::support::endian::read32be(bytes.data() + 16);
  constexpr bool LowerCase = false;
  return llvm::toHex(bytes.slice(0, 16), LowerCase) + std::to_string(age);
}

std::optional<FileSpec> SymbolLocatorMicrosoft::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  // Bail out if we don't have a valid UUID for PDB or
  // 'symbols.enable-external-lookup' is disabled
  const UUID &module_uuid = module_spec.GetUUID();
  if (!module_uuid.IsValid() || module_uuid.GetBytes().size() != 20 ||
      !ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return {};

  Log *log = GetLog(LLDBLog::Symbols);

  std::string key = formatSymStoreKey(module_uuid);
  llvm::StringRef pdb_name =
      module_spec.GetSymbolFileSpec().GetFilename().GetStringRef();
  if (pdb_name.empty()) {
    LLDB_LOGV(log, "Failed to resolve symbol PDB module: PDB name empty");
    return {};
  }

  llvm::StringRef src_dir = GetGlobalPluginProperties().GetURLs().entries().front().ref();
  Args SymStoreURLs = GetGlobalPluginProperties().GetURLs();
  for (const Args::ArgEntry &URL : SymStoreURLs) {
    llvm::SmallString<256> src;
    llvm::sys::path::append(src, src_dir, pdb_name, URL.ref(), pdb_name);
    FileSpec src_spec(src);
    if (!FileSystem::Instance().Exists(src_spec)) {
      LLDB_LOGV(log, "SymbolLocatorMicrosoft: {0} not found in symstore", src);
      continue;
    }
    return src_spec;
  }

  return {};
}
