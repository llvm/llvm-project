//===-- DLanguage.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DLanguage.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"

#include <cstdlib>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(DLanguage)

/// The mangling scheme that this plugin registered.
static Mangled::ManglingScheme g_d_scheme = Mangled::eManglingSchemeNone;

/// Returns true if \p name, which starts with "_D", is a D mangled name.
static bool IsDMangledName(llvm::StringRef name) {
  // A D mangled name begins with `_D`, followed by a numeric length. One known
  // exception is the symbol `_Dmain`. See `SymbolName` and `LName` in
  // https://dlang.org/spec/abi.html#name_mangling
  llvm::StringRef rest = name.drop_front(2);
  return (!rest.empty() && llvm::isDigit(rest.front())) || name == "_Dmain";
}

void DLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "D Language",
                                CreateInstance);
  g_d_scheme =
      Mangled::RegisterManglingScheme({"_D"}, eLanguageTypeD, IsDMangledName);
}

void DLanguage::Terminate() {
  Mangled::UnregisterManglingScheme(g_d_scheme);
  PluginManager::UnregisterPlugin(CreateInstance);
}

// Static Functions
Language *DLanguage::CreateInstance(lldb::LanguageType language) {
  if (language == eLanguageTypeD)
    return new DLanguage();
  return nullptr;
}

bool DLanguage::IsSourceFile(llvm::StringRef file_path) const {
  return file_path.ends_with_insensitive(".d") ||
         file_path.ends_with_insensitive(".di");
}

Language::DemangledName
DLanguage::DemangleSymbol(ConstString mangled, Mangled::ManglingScheme scheme,
                          const DemangleOptions &options) const {
  llvm::StringRef M = mangled.GetStringRef();
  char *demangled_cstr = llvm::dlangDemangle(M);

  if (Log *log = GetLog(LLDBLog::Demangle)) {
    if (demangled_cstr && demangled_cstr[0])
      LLDB_LOG(log, "demangled dlang: {0} -> \"{1}\"", M, demangled_cstr);
    else
      LLDB_LOG(log, "demangled dlang: {0} -> error: failed to demangle", M);
  }

  DemangledName demangled;
  if (demangled_cstr) {
    demangled.name = demangled_cstr;
    std::free(demangled_cstr);
  }
  return demangled;
}
