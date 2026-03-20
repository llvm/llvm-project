//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorSymStore.h"

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

LLDB_PLUGIN_DEFINE(SymbolLocatorSymStore)

namespace {

#define LLDB_PROPERTIES_symbollocatorsymstore
#include "SymbolLocatorSymStoreProperties.inc"

enum {
#define LLDB_PROPERTIES_symbollocatorsymstore
#include "SymbolLocatorSymStorePropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() {
    return SymbolLocatorSymStore::GetPluginNameStatic();
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbollocatorsymstore_properties_def);
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

SymbolLocatorSymStore::SymbolLocatorSymStore() : SymbolLocator() {}

void SymbolLocatorSymStore::Initialize() {
  // First version can only locate PDB in local SymStore (no download yet).
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      nullptr, LocateExecutableSymbolFile, nullptr, nullptr,
      SymbolLocatorSymStore::DebuggerInitialize);
}

void SymbolLocatorSymStore::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolLocatorPlugin(
          debugger, PluginProperties::GetSettingName())) {
    constexpr bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolLocatorPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the SymStore Symbol Locator plug-in.",
        is_global_setting);
  }
}

void SymbolLocatorSymStore::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolLocatorSymStore::GetPluginDescriptionStatic() {
  return "Symbol locator for PDB in SymStore";
}

SymbolLocator *SymbolLocatorSymStore::CreateInstance() {
  return new SymbolLocatorSymStore();
}

// RSDS entries store identity as a 20-byte UUID composed of 16-byte GUID and
// 4-byte age:
//   12345678-1234-5678-9ABC-DEF012345678-00000001
//
// SymStore key is a string with no separators and age as decimal:
//   12345678123456789ABCDEF0123456781
//
static std::string formatSymStoreKey(const UUID &uuid) {
  llvm::ArrayRef<uint8_t> bytes = uuid.GetBytes();
  uint32_t age = llvm::support::endian::read32be(bytes.data() + 16);
  constexpr bool LowerCase = false;
  return llvm::toHex(bytes.slice(0, 16), LowerCase) + std::to_string(age);
}

std::optional<FileSpec> SymbolLocatorSymStore::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  const UUID &uuid = module_spec.GetUUID();
  if (!uuid.IsValid() ||
      !ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return {};

  Log *log = GetLog(LLDBLog::Symbols);
  std::string pdb_name =
      module_spec.GetSymbolFileSpec().GetFilename().GetStringRef().str();
  if (pdb_name.empty()) {
    LLDB_LOG_VERBOSE(log,
                     "Failed to resolve symbol PDB module: PDB name empty");
    return {};
  }

  LLDB_LOG_VERBOSE(log, "LocateExecutableSymbolFile {0} with UUID {1}",
                   pdb_name, uuid.GetAsString());
  if (uuid.GetBytes().size() != 20) {
    LLDB_LOG_VERBOSE(log, "Failed to resolve symbol PDB module: UUID invalid");
    return {};
  }

  std::string key = formatSymStoreKey(uuid);
  Args sym_store_urls = GetGlobalPluginProperties().GetURLs();
  for (const Args::ArgEntry &url : sym_store_urls) {
    llvm::SmallString<256> path;
    llvm::sys::path::append(path, url.ref(), pdb_name, key, pdb_name);
    FileSpec spec(path);
    if (FileSystem::Instance().Exists(spec)) {
      LLDB_LOG_VERBOSE(log, "Found {0} in SymStore {1}", pdb_name, url.ref());
      return spec;
    }
  }

  return {};
}
