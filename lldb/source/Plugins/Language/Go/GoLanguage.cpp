//===-- GoLanguage.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GoLanguage.h"

#include "GoFormatterFunctions.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "llvm/Support/Threading.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(GoLanguage)

void GoLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Go Language",
                                CreateInstance);
}

void GoLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

Language *GoLanguage::CreateInstance(lldb::LanguageType language) {
  if (language == eLanguageTypeGo)
    return new GoLanguage();
  return nullptr;
}

HardcodedFormatters::HardcodedSummaryFinder
GoLanguage::GetHardcodedSummaries() {
  static llvm::once_flag g_initialize;
  static HardcodedFormatters::HardcodedSummaryFinder g_formatters;

  llvm::call_once(g_initialize, [] {
    g_formatters.push_back(
        [](ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &) -> TypeSummaryImpl::SharedPointer {
          static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
              new CXXFunctionSummaryFormat(
                  TypeSummaryImpl::Flags().SetDontShowChildren(true),
                  formatters::GoStringSummaryProvider,
                  "Go string summary provider"));
          if (formatters::IsGoString(valobj))
            return formatter_sp;
          return nullptr;
        });
  });
  return g_formatters;
}

bool GoLanguage::IsSourceFile(llvm::StringRef file_path) const {
  return file_path.ends_with_insensitive(".go");
}
