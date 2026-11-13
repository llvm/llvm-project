//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RustLanguage.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/ValueObject/ValueObject.h"

#include "llvm/Demangle/Demangle.h"

#include <cassert>
#include <cstdlib>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

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
Language *RustLanguage::CreateInstance(LanguageType language) {
  if (Language::LanguageIsRust(language))
    return new RustLanguage();
  return nullptr;
}

bool RustLanguage::IsSourceFile(StringRef file_path) const {
  return file_path.ends_with_insensitive(".rs");
}

std::unique_ptr<Language::TypeScavenger> RustLanguage::GetTypeScavenger() {
  class RustTypeScavenger : public Language::ImageListTypeScavenger {
  public:
    CompilerType AdjustForInclusion(CompilerType &candidate) override {
      LanguageType lang_type(candidate.GetMinimumLanguage());
      if (!Language::LanguageIsC(lang_type) &&
          !Language::LanguageIsRust(lang_type))
        return CompilerType();
      if (candidate.IsTypedefType())
        return candidate.GetTypedefedType();
      return candidate;
    }
  };

  return std::unique_ptr<TypeScavenger>(new RustTypeScavenger());
}

bool RustLanguage::IsNilReference(ValueObject &valobj) {
  if (!Language::LanguageIsRust(valobj.GetObjectRuntimeLanguage()) ||
      !valobj.IsPointerType())
    return false;
  bool canReadValue = true;
  bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
  return canReadValue && isZero;
}

Language::DemangledName
RustLanguage::DemangleSymbol(ConstString mangled,
                             Mangled::ManglingScheme scheme,
                             const DemangleOptions &options) const {
  assert(scheme == g_rust_v0_scheme);
  llvm::StringRef mangled_str = mangled.GetStringRef();
  char *demangled_cstr = llvm::rustDemangle(mangled_str);

  if (Log *log = GetLog(LLDBLog::Demangle)) {
    if (demangled_cstr && demangled_cstr[0])
      LLDB_LOG(log, "demangled rustv0: {0} -> \"{1}\"", mangled_str,
               demangled_cstr);
    else
      LLDB_LOG(log, "demangled rustv0: {0} -> error: failed to demangle",
               mangled_str);
  }

  DemangledName demangled;
  if (demangled_cstr) {
    demangled.name = demangled_cstr;
    std::free(demangled_cstr);
  }
  return demangled;
}

static void LoadRustFormatters(lldb::TypeCategoryImplSP rust_category_sp) {
  if (!rust_category_sp)
    return;

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(false)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  AddCXXSummary(rust_category_sp,
                lldb_private::formatters::ContainerSizeSummaryProvider,
                "Rust vector summary provider", "^alloc::vec::Vec<.+>(( )?&)?$",
                summary_flags, true);

  SyntheticChildren::Flags synth_flags;
  synth_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetFrontEndWantsDereference();

  AddCXXSynthetic(rust_category_sp,
                  lldb_private::formatters::RustVectorSyntheticFrontEndCreator,
                  "Rust vector synthetic children",
                  "^alloc::vec::Vec<.+>(( )?&)?$", synth_flags, true);
}

lldb::TypeCategoryImplSP RustLanguage::GetFormatters() {
  static llvm::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  llvm::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(GetPluginName(), g_category);
    if (g_category)
      LoadRustFormatters(g_category);
  });
  return g_category;
}
