//===-- JavaLanguage.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <string.h>
// C++ Includes
#include <functional>
#include <mutex>

// Other libraries and framework includes
#include "llvm/ADT/StringRef.h"

// Project includes
#include "JavaFormatterFunctions.h"
#include "JavaLanguage.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/JavaASTContext.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

void JavaLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Java Language",
                                CreateInstance);
}

void JavaLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString JavaLanguage::GetPluginNameStatic() {
  static ConstString g_name("Java");
  return g_name;
}

lldb_private::ConstString JavaLanguage::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t JavaLanguage::GetPluginVersion() { return 1; }

Language *JavaLanguage::CreateInstance(lldb::LanguageType language) {
  if (language == eLanguageTypeJava)
    return new JavaLanguage();
  return nullptr;
}

bool JavaLanguage::IsNilReference(ValueObject &valobj) {
  if (!valobj.GetCompilerType().IsReferenceType())
    return false;

  // If we failed to read the value then it is not a nil reference.
  return valobj.GetValueAsUnsigned(UINT64_MAX) == 0;
}

lldb::TypeCategoryImplSP JavaLanguage::GetFormatters() {
  static std::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  std::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(GetPluginName(), g_category);
    if (g_category) {
      const char *array_regexp = "^.*\\[\\]&?$";

      lldb::TypeSummaryImplSP string_summary_sp(new CXXFunctionSummaryFormat(
          TypeSummaryImpl::Flags().SetDontShowChildren(true),
          lldb_private::formatters::JavaStringSummaryProvider,
          "java.lang.String summary provider"));
      g_category->GetTypeSummariesContainer()->Add(
          ConstString("java::lang::String"), string_summary_sp);

      lldb::TypeSummaryImplSP array_summary_sp(new CXXFunctionSummaryFormat(
          TypeSummaryImpl::Flags().SetDontShowChildren(true),
          lldb_private::formatters::JavaArraySummaryProvider,
          "Java array summary provider"));
      g_category->GetRegexTypeSummariesContainer()->Add(
          RegularExpressionSP(new RegularExpression(array_regexp)),
          array_summary_sp);

#ifndef LLDB_DISABLE_PYTHON
      AddCXXSynthetic(
          g_category,
          lldb_private::formatters::JavaArraySyntheticFrontEndCreator,
          "Java array synthetic children", ConstString(array_regexp),
          SyntheticChildren::Flags().SetCascades(true), true);
#endif
    }
  });
  return g_category;
}
