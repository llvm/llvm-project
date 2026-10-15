//===-- RustLanguage.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RustLanguage.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/Demangle/Demangle.h"

#include <cstdlib>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(RustLanguage)

/// The mangling scheme that this plugin registered.
static Mangled::ManglingScheme g_rust_v0_scheme = Mangled::eManglingSchemeNone;

void RustLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Rust Language",
                                CreateInstance);
  g_rust_v0_scheme = Mangled::RegisterManglingScheme({"_R"}, eLanguageTypeRust);
}

void RustLanguage::Terminate() {
  Mangled::UnregisterManglingScheme(g_rust_v0_scheme);
  PluginManager::UnregisterPlugin(CreateInstance);
}

// Static Functions
Language *RustLanguage::CreateInstance(lldb::LanguageType language) {
  if (language == eLanguageTypeRust)
    return new RustLanguage();
  return nullptr;
}

bool RustLanguage::IsSourceFile(llvm::StringRef file_path) const {
  return file_path.ends_with_insensitive(".rs");
}

Language::DemangledName
RustLanguage::DemangleSymbol(ConstString mangled,
                             Mangled::ManglingScheme scheme,
                             const DemangleOptions &options) const {
  llvm::StringRef M = mangled.GetStringRef();
  char *demangled_cstr = llvm::rustDemangle(M);

  if (Log *log = GetLog(LLDBLog::Demangle)) {
    if (demangled_cstr && demangled_cstr[0])
      LLDB_LOG(log, "demangled rustv0: {0} -> \"{1}\"", M, demangled_cstr);
    else
      LLDB_LOG(log, "demangled rustv0: {0} -> error: failed to demangle", M);
  }

  DemangledName demangled;
  if (demangled_cstr) {
    demangled.name = demangled_cstr;
    std::free(demangled_cstr);
  }
  return demangled;
}
