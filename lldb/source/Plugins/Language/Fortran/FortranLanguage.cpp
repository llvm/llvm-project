//===-- FortranLanguage.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"

#include "FortranLanguage.h"

#include "DynamicArray.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(FortranLanguage)

void FortranLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Fortran Language",
                                CreateInstance);
}

void FortranLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

StringRef FortranLanguage::GetPluginNameStatic() {
  static llvm::StringRef g_name("fortran");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
StringRef FortranLanguage::GetPluginName() { return GetPluginNameStatic(); }

uint32_t FortranLanguage::GetPluginVersion() { return 1; }

Language *FortranLanguage::CreateInstance(LanguageType language) {
  if (Language::LanguageIsFortran(language)) {
    return new FortranLanguage();
  }
  return nullptr;
}

bool FortranLanguage::IsSourceFile(StringRef file_path) const {
  const auto suffixes = {".f90", ".f"};
  for (auto suffix : suffixes) {
    if (file_path.ends_with_insensitive(suffix))
      return true;
  }
  return false;
}

static lldb::SyntheticChildrenSP
FortranDynamicArrayFinder(ValueObject &valobj,
                          lldb::DynamicValueType use_dynamic) {
  CompilerType type = valobj.GetCompilerType();

  // 1. Query if it is actually an array type according to DWARF
  if (type.IsValid() && type.IsArrayType(nullptr, nullptr, nullptr)) {

    // 2. Setup the flags
    SyntheticChildren::Flags flags;
    flags.SetCascades(true)
        .SetSkipPointers(false)
        .SetSkipReferences(false)
        .SetFrontEndWantsDereference();

    // 3. Bind and return your custom frontend creator
    return lldb::SyntheticChildrenSP(new CXXSyntheticChildren(
        flags, "fortran array synthetic children",
        lldb_private::formatters::FortranDynamicArraySyntheticFrontEndCreator));
  }

  // Return null if it's not an array, so LLDB can try other formatters
  return nullptr;
}

// Implement the Language plugin override
HardcodedFormatters::HardcodedSyntheticFinder
FortranLanguage::GetHardcodedSynthetics() {
  HardcodedFormatters::HardcodedSyntheticFinder formatters;

  // Push our structural finder into the formatters list
  formatters.push_back([](lldb_private::ValueObject &valobj,
                          lldb::DynamicValueType use_dynamic,
                          FormatManager &fmt_mgr) -> lldb::SyntheticChildrenSP {
    return FortranDynamicArrayFinder(valobj, use_dynamic);
  });

  return formatters;
}
