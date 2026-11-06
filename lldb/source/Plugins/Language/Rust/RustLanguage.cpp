//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"

#include "RustLanguage.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(RustLanguage)

void RustLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Rust Language",
                                CreateInstance);
}

void RustLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

StringRef RustLanguage::GetPluginNameStatic() {
  static llvm::StringRef g_name("rust");
  return g_name;
}

StringRef RustLanguage::GetPluginName() { return GetPluginNameStatic(); }

uint32_t RustLanguage::GetPluginVersion() { return 1; }

Language *RustLanguage::CreateInstance(LanguageType language) {
  if (Language::LanguageIsRust(language))
    return new RustLanguage();

  return nullptr;
}

bool RustLanguage::IsSourceFile(StringRef file_path) const {
  const auto suffixes = {".rs"};
  for (auto suffix : suffixes) {
    if (file_path.ends_with_insensitive(suffix))
      return true;
  }
  return false;
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
  synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);

  AddCXXSynthetic(rust_category_sp,
                  lldb_private::formatters::RustVectorSyntheticFrontEndCreator,
                  "Rust vector synthetic children",
                  "^alloc::vec::Vec<.+>(( )?&)?$", synth_flags, true);
}

lldb::TypeCategoryImplSP RustLanguage::GetFormatters() {
  static llvm::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  llvm::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               g_category);
    if (g_category)
      LoadRustFormatters(g_category);
  });
  return g_category;
}
