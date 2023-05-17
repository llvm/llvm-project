//===-- ObjCLanguage.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>

#include "ObjCLanguage.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/Threading.h"

#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

#include "CF.h"
#include "Cocoa.h"
#include "CoreMedia.h"
#include "NSDictionary.h"
#include "NSSet.h"
#include "NSString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(ObjCLanguage)

void ObjCLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Objective-C Language",
                                CreateInstance);
}

void ObjCLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

// Static Functions

Language *ObjCLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeObjC:
    return new ObjCLanguage();
  default:
    return nullptr;
  }
}

void ObjCLanguage::MethodName::Clear() {
  m_full.Clear();
  m_class.Clear();
  m_category.Clear();
  m_selector.Clear();
  m_type = eTypeUnspecified;
  m_category_is_valid = false;
}

bool ObjCLanguage::MethodName::SetName(llvm::StringRef name, bool strict) {
  Clear();
  if (name.empty())
    return IsValid(strict);

  // If "strict" is true. then the method must be specified with a '+' or '-'
  // at the beginning. If "strict" is false, then the '+' or '-' can be omitted
  bool valid_prefix = false;

  if (name.size() > 1 && (name[0] == '+' || name[0] == '-')) {
    valid_prefix = name[1] == '[';
    if (name[0] == '+')
      m_type = eTypeClassMethod;
    else
      m_type = eTypeInstanceMethod;
  } else if (!strict) {
    // "strict" is false, the name just needs to start with '['
    valid_prefix = name[0] == '[';
  }

  if (valid_prefix) {
    int name_len = name.size();
    // Objective-C methods must have at least:
    //      "-[" or "+[" prefix
    //      One character for a class name
    //      One character for the space between the class name
    //      One character for the method name
    //      "]" suffix
    if (name_len >= (5 + (strict ? 1 : 0)) && name.back() == ']') {
      m_full.SetString(name);
    }
  }
  return IsValid(strict);
}

bool ObjCLanguage::MethodName::SetName(const char *name, bool strict) {
  return SetName(llvm::StringRef(name), strict);
}

ConstString ObjCLanguage::MethodName::GetClassName() {
  if (!m_class) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *paren_pos = strchr(class_start, '(');
      if (paren_pos) {
        m_class.SetCStringWithLength(class_start, paren_pos - class_start);
      } else {
        // No '(' was found in the full name, we can definitively say that our
        // category was valid (and empty).
        m_category_is_valid = true;
        const char *space_pos = strchr(full, ' ');
        if (space_pos) {
          m_class.SetCStringWithLength(class_start, space_pos - class_start);
          if (!m_class_category) {
            // No category in name, so we can also fill in the m_class_category
            m_class_category = m_class;
          }
        }
      }
    }
  }
  return m_class;
}

ConstString ObjCLanguage::MethodName::GetClassNameWithCategory() {
  if (!m_class_category) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *space_pos = strchr(full, ' ');
      if (space_pos) {
        m_class_category.SetCStringWithLength(class_start,
                                              space_pos - class_start);
        // If m_class hasn't been filled in and the class with category doesn't
        // contain a '(', then we can also fill in the m_class
        if (!m_class && strchr(m_class_category.GetCString(), '(') == nullptr) {
          m_class = m_class_category;
          // No '(' was found in the full name, we can definitively say that
          // our category was valid (and empty).
          m_category_is_valid = true;
        }
      }
    }
  }
  return m_class_category;
}

ConstString ObjCLanguage::MethodName::GetSelector() {
  if (!m_selector) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *space_pos = strchr(full, ' ');
      if (space_pos) {
        ++space_pos; // skip the space
        m_selector.SetCStringWithLength(space_pos, m_full.GetLength() -
                                                       (space_pos - full) - 1);
      }
    }
  }
  return m_selector;
}

ConstString ObjCLanguage::MethodName::GetCategory() {
  if (!m_category_is_valid && !m_category) {
    if (IsValid(false)) {
      m_category_is_valid = true;
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *open_paren_pos = strchr(class_start, '(');
      if (open_paren_pos) {
        ++open_paren_pos; // Skip the open paren
        const char *close_paren_pos = strchr(open_paren_pos, ')');
        if (close_paren_pos)
          m_category.SetCStringWithLength(open_paren_pos,
                                          close_paren_pos - open_paren_pos);
      }
    }
  }
  return m_category;
}

ConstString ObjCLanguage::MethodName::GetFullNameWithoutCategory(
    bool empty_if_no_category) {
  if (IsValid(false)) {
    if (HasCategory()) {
      StreamString strm;
      if (m_type == eTypeClassMethod)
        strm.PutChar('+');
      else if (m_type == eTypeInstanceMethod)
        strm.PutChar('-');
      strm.Printf("[%s %s]", GetClassName().GetCString(),
                  GetSelector().GetCString());
      return ConstString(strm.GetString());
    }

    if (!empty_if_no_category) {
      // Just return the full name since it doesn't have a category
      return GetFullName();
    }
  }
  return ConstString();
}

std::vector<Language::MethodNameVariant>
ObjCLanguage::GetMethodNameVariants(ConstString method_name) const {
  std::vector<Language::MethodNameVariant> variant_names;
  ObjCLanguage::MethodName objc_method(method_name.GetCString(), false);
  if (!objc_method.IsValid(false)) {
    return variant_names;
  }

  variant_names.emplace_back(objc_method.GetSelector(),
                             lldb::eFunctionNameTypeSelector);

  const bool is_class_method =
      objc_method.GetType() == MethodName::eTypeClassMethod;
  const bool is_instance_method =
      objc_method.GetType() == MethodName::eTypeInstanceMethod;
  ConstString name_sans_category =
      objc_method.GetFullNameWithoutCategory(/*empty_if_no_category*/ true);

  if (is_class_method || is_instance_method) {
    if (name_sans_category)
      variant_names.emplace_back(name_sans_category,
                                 lldb::eFunctionNameTypeFull);
  } else {
    StreamString strm;

    strm.Printf("+%s", objc_method.GetFullName().GetCString());
    variant_names.emplace_back(ConstString(strm.GetString()),
                               lldb::eFunctionNameTypeFull);
    strm.Clear();

    strm.Printf("-%s", objc_method.GetFullName().GetCString());
    variant_names.emplace_back(ConstString(strm.GetString()),
                               lldb::eFunctionNameTypeFull);
    strm.Clear();

    if (name_sans_category) {
      strm.Printf("+%s", name_sans_category.GetCString());
      variant_names.emplace_back(ConstString(strm.GetString()),
                                 lldb::eFunctionNameTypeFull);
      strm.Clear();

      strm.Printf("-%s", name_sans_category.GetCString());
      variant_names.emplace_back(ConstString(strm.GetString()),
                                 lldb::eFunctionNameTypeFull);
    }
  }

  return variant_names;
}

bool ObjCLanguage::SymbolNameFitsToLanguage(Mangled mangled) const {
  ConstString demangled_name = mangled.GetDemangledName();
  if (!demangled_name)
    return false;
  return ObjCLanguage::IsPossibleObjCMethodName(demangled_name.GetCString());
}

static void LoadObjCFormatters(TypeCategoryImplSP objc_category_sp) {
  if (!objc_category_sp)
    return;

  TypeSummaryImpl::Flags objc_flags;
  objc_flags.SetCascades(false)
      .SetSkipPointers(true)
      .SetSkipReferences(true)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  lldb::TypeSummaryImplSP ObjC_BOOL_summary(new CXXFunctionSummaryFormat(
      objc_flags, lldb_private::formatters::ObjCBOOLSummaryProvider, ""));
  objc_category_sp->AddTypeSummary("BOOL", eFormatterMatchExact,
                                   ObjC_BOOL_summary);
  objc_category_sp->AddTypeSummary("BOOL &", eFormatterMatchExact,
                                   ObjC_BOOL_summary);
  objc_category_sp->AddTypeSummary("BOOL *", eFormatterMatchExact,
                                   ObjC_BOOL_summary);

  // we need to skip pointers here since we are special casing a SEL* when
  // retrieving its value
  objc_flags.SetSkipPointers(true);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<false>,
                "SEL summary provider", "SEL", objc_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<false>,
                "SEL summary provider", "struct objc_selector", objc_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<false>,
                "SEL summary provider", "objc_selector", objc_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<true>,
                "SEL summary provider", "objc_selector *", objc_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<true>,
                "SEL summary provider", "SEL *", objc_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCClassSummaryProvider,
                "Class summary provider", "Class", objc_flags);

  SyntheticChildren::Flags class_synth_flags;
  class_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::ObjCClassSyntheticFrontEndCreator,
                  "Class synthetic children", "Class", class_synth_flags);

  objc_flags.SetSkipPointers(false);
  objc_flags.SetCascades(true);
  objc_flags.SetSkipReferences(false);

  AddStringSummary(objc_category_sp, "${var.__FuncPtr%A}",
                   "__block_literal_generic", objc_flags);

  AddStringSummary(objc_category_sp,
                   "${var.years} years, ${var.months} "
                   "months, ${var.days} days, ${var.hours} "
                   "hours, ${var.minutes} minutes "
                   "${var.seconds} seconds",
                   "CFGregorianUnits", objc_flags);
  AddStringSummary(objc_category_sp,
                   "location=${var.location} length=${var.length}", "CFRange",
                   objc_flags);

  AddStringSummary(objc_category_sp,
                   "location=${var.location}, length=${var.length}", "NSRange",
                   objc_flags);
  AddStringSummary(objc_category_sp, "(${var.origin}, ${var.size}), ...",
                   "NSRectArray", objc_flags);

  AddOneLineSummary(objc_category_sp, "NSPoint", objc_flags);
  AddOneLineSummary(objc_category_sp, "NSSize", objc_flags);
  AddOneLineSummary(objc_category_sp, "NSRect", objc_flags);

  AddOneLineSummary(objc_category_sp, "CGSize", objc_flags);
  AddOneLineSummary(objc_category_sp, "CGPoint", objc_flags);
  AddOneLineSummary(objc_category_sp, "CGRect", objc_flags);

  AddStringSummary(objc_category_sp,
                   "red=${var.red} green=${var.green} blue=${var.blue}",
                   "RGBColor", objc_flags);
  AddStringSummary(
      objc_category_sp,
      "(t=${var.top}, l=${var.left}, b=${var.bottom}, r=${var.right})", "Rect",
      objc_flags);
  AddStringSummary(objc_category_sp, "{(v=${var.v}, h=${var.h})}", "Point",
                   objc_flags);
  AddStringSummary(objc_category_sp,
                   "${var.month}/${var.day}/${var.year}  ${var.hour} "
                   ":${var.minute} :${var.second} dayOfWeek:${var.dayOfWeek}",
                   "DateTimeRect *", objc_flags);
  AddStringSummary(objc_category_sp,
                   "${var.ld.month}/${var.ld.day}/"
                   "${var.ld.year} ${var.ld.hour} "
                   ":${var.ld.minute} :${var.ld.second} "
                   "dayOfWeek:${var.ld.dayOfWeek}",
                   "LongDateRect", objc_flags);
  AddStringSummary(objc_category_sp, "(x=${var.x}, y=${var.y})", "HIPoint",
                   objc_flags);
  AddStringSummary(objc_category_sp, "origin=${var.origin} size=${var.size}",
                   "HIRect", objc_flags);

  TypeSummaryImpl::Flags appkit_flags;
  appkit_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  appkit_flags.SetDontShowChildren(false);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "NSArray", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "NSConstantArray", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "NSMutableArray", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "__NSArrayI", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "__NSArray0", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", "__NSSingleObjectArrayI", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "__NSArrayM", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "__NSCFArray", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "_NSCallStackArray", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "CFArrayRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", "CFMutableArrayRef", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "NSDictionary", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "NSConstantDictionary",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "NSMutableDictionary",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "__NSCFDictionary",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "__NSDictionaryI",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "__NSSingleEntryDictionaryI",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", "__NSDictionaryM",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider", "CFDictionaryRef",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider", "__CFDictionary",
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider", "CFMutableDictionaryRef",
                appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSSet summary", "NSSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSMutableSet summary", "NSMutableSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<true>,
                "CFSetRef summary", "CFSetRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<true>,
                "CFMutableSetRef summary", "CFMutableSetRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSCFSet summary", "__NSCFSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__CFSet summary", "__CFSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSSetI summary", "__NSSetI", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSSetM summary", "__NSSetM", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSCountedSet summary", "NSCountedSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSMutableSet summary", "NSMutableSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSOrderedSet summary", "NSOrderedSet", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSOrderedSetI summary", "__NSOrderedSetI", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSOrderedSetM summary", "__NSOrderedSetM", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSError_SummaryProvider,
                "NSError summary provider", "NSError", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSException_SummaryProvider,
                "NSException summary provider", "NSException", appkit_flags);

  // AddSummary(appkit_category_sp, "${var.key%@} -> ${var.value%@}",
  // ConstString("$_lldb_typegen_nspair"), appkit_flags);

  appkit_flags.SetDontShowChildren(true);

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "__NSArrayM",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "__NSArrayI",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "__NSArray0",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "__NSSingleObjectArrayI",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "NSArray",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "NSConstantArray",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "NSMutableArray",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "__NSCFArray",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "_NSCallStackArray",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "CFMutableArrayRef",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", "CFArrayRef",
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "__NSDictionaryM",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "NSConstantDictionary",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "__NSDictionaryI",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "__NSSingleEntryDictionaryI",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "__NSCFDictionary",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "NSDictionary",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "NSMutableDictionary",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "CFDictionaryRef",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "CFMutableDictionaryRef",
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", "__CFDictionary",
      ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSErrorSyntheticFrontEndCreator,
                  "NSError synthetic children", "NSError",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSExceptionSyntheticFrontEndCreator,
                  "NSException synthetic children", "NSException",
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(
      objc_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator,
      "NSSet synthetic children", "NSSet", ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSSetI synthetic children", "__NSSetI",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSSetM synthetic children", "__NSSetM",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSCFSet synthetic children", "__NSCFSet",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "CFSetRef synthetic children", "CFSetRef",
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "NSMutableSet synthetic children", "NSMutableSet",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "NSOrderedSet synthetic children", "NSOrderedSet",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSOrderedSetI synthetic children", "__NSOrderedSetI",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSOrderedSetM synthetic children", "__NSOrderedSetM",
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__CFSet synthetic children", "__CFSet",
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSIndexPathSyntheticFrontEndCreator,
                  "NSIndexPath synthetic children", "NSIndexPath",
                  ScriptedSyntheticChildren::Flags());

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", "CFBagRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", "__CFBag", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", "const struct __CFBag", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", "CFMutableBagRef", appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBinaryHeapSummaryProvider,
      "CFBinaryHeap summary provider", "CFBinaryHeapRef", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBinaryHeapSummaryProvider,
      "CFBinaryHeap summary provider", "__CFBinaryHeap", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "NSString", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "CFStringRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "__CFString", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", "CFMutableStringRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "NSMutableString", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", "__NSCFConstantString", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "__NSCFString", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", "NSCFConstantString", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "NSCFString", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", "NSPathStore2", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", "NSTaggedPointerString", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSAttributedStringSummaryProvider,
                "NSAttributedString summary provider", "NSAttributedString",
                appkit_flags);
  AddCXXSummary(
      objc_category_sp,
      lldb_private::formatters::NSMutableAttributedStringSummaryProvider,
      "NSMutableAttributedString summary provider", "NSMutableAttributedString",
      appkit_flags);
  AddCXXSummary(
      objc_category_sp,
      lldb_private::formatters::NSMutableAttributedStringSummaryProvider,
      "NSMutableAttributedString summary provider",
      "NSConcreteMutableAttributedString", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSBundleSummaryProvider,
                "NSBundle summary provider", "NSBundle", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", "NSData", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", "_NSInlineData", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", "NSConcreteData", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>,
      "NSData summary provider", "NSConcreteMutableData", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", "NSMutableData", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", "__NSCFData", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<true>,
                "NSData summary provider", "CFDataRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<true>,
                "NSData summary provider", "CFMutableDataRef", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSMachPortSummaryProvider,
                "NSMachPort summary provider", "NSMachPort", appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNotificationSummaryProvider,
      "NSNotification summary provider", "NSNotification", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNotificationSummaryProvider,
                "NSNotification summary provider", "NSConcreteNotification",
                appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider", "NSNumber", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", "NSConstantIntegerNumber", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", "NSConstantDoubleNumber", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", "NSConstantFloatNumber", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "CFNumberRef summary provider", "CFNumberRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider", "__NSCFBoolean", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider", "__NSCFNumber", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider", "NSCFBoolean", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider", "NSCFNumber", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSDecimalNumber summary provider", "NSDecimalNumber", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSURLSummaryProvider,
                "NSURL summary provider", "NSURL", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSURLSummaryProvider,
                "NSURL summary provider", "CFURLRef", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDateSummaryProvider,
                "NSDate summary provider", "NSDate", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDateSummaryProvider,
                "NSDate summary provider", "__NSDate", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDateSummaryProvider,
                "NSDate summary provider", "__NSTaggedDate", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDateSummaryProvider,
                "NSDate summary provider", "NSCalendarDate", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSTimeZoneSummaryProvider,
                "NSTimeZone summary provider", "NSTimeZone", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSTimeZoneSummaryProvider,
                "NSTimeZone summary provider", "CFTimeZoneRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSTimeZoneSummaryProvider,
                "NSTimeZone summary provider", "__NSTimeZone", appkit_flags);

  // CFAbsoluteTime is actually a double rather than a pointer to an object we
  // do not care about the numeric value, since it is probably meaningless to
  // users
  appkit_flags.SetDontShowValue(true);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFAbsoluteTimeSummaryProvider,
      "CFAbsoluteTime summary provider", "CFAbsoluteTime", appkit_flags);
  appkit_flags.SetDontShowValue(false);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSIndexSetSummaryProvider,
                "NSIndexSet summary provider", "NSIndexSet", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSIndexSetSummaryProvider,
      "NSIndexSet summary provider", "NSMutableIndexSet", appkit_flags);

  AddStringSummary(objc_category_sp,
                   "@\"${var.month%d}/${var.day%d}/${var.year%d} "
                   "${var.hour%d}:${var.minute%d}:${var.second}\"",
                   "CFGregorianDate", appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider", "CFBitVectorRef", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider,
      "CFBitVector summary provider", "CFMutableBitVectorRef", appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider", "__CFBitVector", appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider,
      "CFBitVector summary provider", "__CFMutableBitVector", appkit_flags);
}

static void LoadCoreMediaFormatters(TypeCategoryImplSP objc_category_sp) {
  if (!objc_category_sp)
    return;

  TypeSummaryImpl::Flags cm_flags;
  cm_flags.SetCascades(true)
      .SetDontShowChildren(false)
      .SetDontShowValue(false)
      .SetHideItemNames(false)
      .SetShowMembersOneLiner(false)
      .SetSkipPointers(false)
      .SetSkipReferences(false);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CMTimeSummaryProvider,
                "CMTime summary provider", "CMTime", cm_flags);
}

lldb::TypeCategoryImplSP ObjCLanguage::GetFormatters() {
  static llvm::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  llvm::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               g_category);
    if (g_category) {
      LoadCoreMediaFormatters(g_category);
      LoadObjCFormatters(g_category);
    }
  });
  return g_category;
}

std::vector<FormattersMatchCandidate>
ObjCLanguage::GetPossibleFormattersMatches(ValueObject &valobj,
                                           lldb::DynamicValueType use_dynamic) {
  std::vector<FormattersMatchCandidate> result;

  if (use_dynamic == lldb::eNoDynamicValues)
    return result;

  CompilerType compiler_type(valobj.GetCompilerType());

  const bool check_cpp = false;
  const bool check_objc = true;
  bool canBeObjCDynamic =
      compiler_type.IsPossibleDynamicType(nullptr, check_cpp, check_objc);

  if (canBeObjCDynamic && ClangUtil::IsClangType(compiler_type)) {
    do {
      lldb::ProcessSP process_sp = valobj.GetProcessSP();
      if (!process_sp)
        break;
      ObjCLanguageRuntime *runtime = ObjCLanguageRuntime::Get(*process_sp);
      if (runtime == nullptr)
        break;
      ObjCLanguageRuntime::ClassDescriptorSP objc_class_sp(
          runtime->GetClassDescriptor(valobj));
      if (!objc_class_sp)
        break;
      if (ConstString name = objc_class_sp->GetClassName())
        result.push_back(
            {name, valobj.GetTargetSP()->GetDebugger().GetScriptInterpreter(),
             TypeImpl(objc_class_sp->GetType()),
             FormattersMatchCandidate::Flags{}});
    } while (false);
  }

  return result;
}

std::unique_ptr<Language::TypeScavenger> ObjCLanguage::GetTypeScavenger() {
  class ObjCScavengerResult : public Language::TypeScavenger::Result {
  public:
    ObjCScavengerResult(CompilerType type)
        : Language::TypeScavenger::Result(), m_compiler_type(type) {}

    bool IsValid() override { return m_compiler_type.IsValid(); }

    bool DumpToStream(Stream &stream, bool print_help_if_available) override {
      if (IsValid()) {
        m_compiler_type.DumpTypeDescription(&stream);
        stream.EOL();
        return true;
      }
      return false;
    }

  private:
    CompilerType m_compiler_type;
  };

  class ObjCRuntimeScavenger : public Language::TypeScavenger {
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      bool result = false;

      if (auto *process = exe_scope->CalculateProcess().get()) {
        if (auto *objc_runtime = ObjCLanguageRuntime::Get(*process)) {
          if (auto *decl_vendor = objc_runtime->GetDeclVendor()) {
            ConstString name(key);
            for (const CompilerType &type :
                 decl_vendor->FindTypes(name, /*max_matches*/ UINT32_MAX)) {
              result = true;
              std::unique_ptr<Language::TypeScavenger::Result> result(
                  new ObjCScavengerResult(type));
              results.insert(std::move(result));
            }
          }
        }
      }

      return result;
    }

    friend class lldb_private::ObjCLanguage;
  };

  class ObjCModulesScavenger : public Language::TypeScavenger {
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      bool result = false;

      if (auto *target = exe_scope->CalculateTarget().get()) {
        auto *persistent_vars = llvm::cast<ClangPersistentVariables>(
            target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeC));
        if (std::shared_ptr<ClangModulesDeclVendor> clang_modules_decl_vendor =
                persistent_vars->GetClangModulesDeclVendor()) {
          ConstString key_cs(key);
          auto types = clang_modules_decl_vendor->FindTypes(
              key_cs, /*max_matches*/ UINT32_MAX);
          if (!types.empty()) {
            result = true;
            std::unique_ptr<Language::TypeScavenger::Result> result(
                new ObjCScavengerResult(types.front()));
            results.insert(std::move(result));
          }
        }
      }

      return result;
    }

    friend class lldb_private::ObjCLanguage;
  };
  
  class ObjCDebugInfoScavenger : public Language::ImageListTypeScavenger {
  public:
    CompilerType AdjustForInclusion(CompilerType &candidate) override {
      LanguageType lang_type(candidate.GetMinimumLanguage());
      if (!Language::LanguageIsObjC(lang_type))
        return CompilerType();
      if (candidate.IsTypedefType())
        return candidate.GetTypedefedType();
      return candidate;
    }
  };

  return std::unique_ptr<TypeScavenger>(
      new Language::EitherTypeScavenger<ObjCModulesScavenger,
                                        ObjCRuntimeScavenger,
                                        ObjCDebugInfoScavenger>());
}

bool ObjCLanguage::GetFormatterPrefixSuffix(ValueObject &valobj,
                                            ConstString type_hint,
                                            std::string &prefix,
                                            std::string &suffix) {
  static ConstString g_CFBag("CFBag");
  static ConstString g_CFBinaryHeap("CFBinaryHeap");

  static ConstString g_NSNumberChar("NSNumber:char");
  static ConstString g_NSNumberShort("NSNumber:short");
  static ConstString g_NSNumberInt("NSNumber:int");
  static ConstString g_NSNumberLong("NSNumber:long");
  static ConstString g_NSNumberInt128("NSNumber:int128_t");
  static ConstString g_NSNumberFloat("NSNumber:float");
  static ConstString g_NSNumberDouble("NSNumber:double");

  static ConstString g_NSData("NSData");
  static ConstString g_NSArray("NSArray");
  static ConstString g_NSString("NSString");
  static ConstString g_NSStringStar("NSString*");

  if (type_hint.IsEmpty())
    return false;

  prefix.clear();
  suffix.clear();

  if (type_hint == g_CFBag || type_hint == g_CFBinaryHeap) {
    prefix = "@";
    return true;
  }

  if (type_hint == g_NSNumberChar) {
    prefix = "(char)";
    return true;
  }
  if (type_hint == g_NSNumberShort) {
    prefix = "(short)";
    return true;
  }
  if (type_hint == g_NSNumberInt) {
    prefix = "(int)";
    return true;
  }
  if (type_hint == g_NSNumberLong) {
    prefix = "(long)";
    return true;
  }
  if (type_hint == g_NSNumberInt128) {
    prefix = "(int128_t)";
    return true;
  }
  if (type_hint == g_NSNumberFloat) {
    prefix = "(float)";
    return true;
  }
  if (type_hint == g_NSNumberDouble) {
    prefix = "(double)";
    return true;
  }

  if (type_hint == g_NSData || type_hint == g_NSArray) {
    prefix = "@\"";
    suffix = "\"";
    return true;
  }

  if (type_hint == g_NSString || type_hint == g_NSStringStar) {
    prefix = "@";
    return true;
  }

  return false;
}

bool ObjCLanguage::IsNilReference(ValueObject &valobj) {
  const uint32_t mask = eTypeIsObjC | eTypeIsPointer;
  bool isObjCpointer =
      (((valobj.GetCompilerType().GetTypeInfo(nullptr)) & mask) == mask);
  if (!isObjCpointer)
    return false;
  bool canReadValue = true;
  bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
  return canReadValue && isZero;
}

bool ObjCLanguage::IsSourceFile(llvm::StringRef file_path) const {
  const auto suffixes = {".h", ".m", ".M"};
  for (auto suffix : suffixes) {
    if (file_path.endswith_insensitive(suffix))
      return true;
  }
  return false;
}
