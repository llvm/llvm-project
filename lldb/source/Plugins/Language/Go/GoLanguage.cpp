//===-- GoLanguage.cpp ------------------------------------------*- C++ -*-===//
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
#include "GoLanguage.h"
#include "Plugins/Language/Go/GoFormatterFunctions.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/GoASTContext.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

void GoLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Go Language",
                                CreateInstance);
}

void GoLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString GoLanguage::GetPluginNameStatic() {
  static ConstString g_name("Go");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString GoLanguage::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t GoLanguage::GetPluginVersion() { return 1; }

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
Language *GoLanguage::CreateInstance(lldb::LanguageType language) {
  if (language == eLanguageTypeGo)
    return new GoLanguage();
  return nullptr;
}

HardcodedFormatters::HardcodedSummaryFinder
GoLanguage::GetHardcodedSummaries() {
  static std::once_flag g_initialize;
  static HardcodedFormatters::HardcodedSummaryFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType, FormatManager &)
                               -> TypeSummaryImpl::SharedPointer {
      static CXXFunctionSummaryFormat::SharedPointer formatter_sp(
          new CXXFunctionSummaryFormat(
              TypeSummaryImpl::Flags().SetDontShowChildren(true),
              lldb_private::formatters::GoStringSummaryProvider,
              "Go string summary provider"));
      if (GoASTContext::IsGoString(valobj.GetCompilerType())) {
        return formatter_sp;
      }
      if (GoASTContext::IsGoString(valobj.GetCompilerType().GetPointeeType())) {
        return formatter_sp;
      }
      return nullptr;
    });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType, FormatManager &)
                               -> TypeSummaryImpl::SharedPointer {
      static lldb::TypeSummaryImplSP formatter_sp(new StringSummaryFormat(
          TypeSummaryImpl::Flags().SetHideItemNames(true),
          "(len ${var.len}, cap ${var.cap})"));
      if (GoASTContext::IsGoSlice(valobj.GetCompilerType())) {
        return formatter_sp;
      }
      if (GoASTContext::IsGoSlice(valobj.GetCompilerType().GetPointeeType())) {
        return formatter_sp;
      }
      return nullptr;
    });
  });
  return g_formatters;
}

HardcodedFormatters::HardcodedSyntheticFinder
GoLanguage::GetHardcodedSynthetics() {
  static std::once_flag g_initialize;
  static HardcodedFormatters::HardcodedSyntheticFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType,
           FormatManager &fmt_mgr) -> SyntheticChildren::SharedPointer {
          static CXXSyntheticChildren::SharedPointer formatter_sp(
              new CXXSyntheticChildren(
                  SyntheticChildren::Flags(), "slice synthetic children",
                  lldb_private::formatters::GoSliceSyntheticFrontEndCreator));
          if (GoASTContext::IsGoSlice(valobj.GetCompilerType())) {
            return formatter_sp;
          }
          return nullptr;
        });
  });

  return g_formatters;
}
