//===-- SwiftLanguage.cpp ---------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftLanguage.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"

#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"

#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Target/SwiftLanguageRuntime.h"

#include "ObjCRuntimeSyntheticProvider.cpp"
#include "SwiftFormatters.h"

#include <functional>
#include <mutex>

#include "swift/AST/Module.h"
#include "swift/AST/Type.h"
#include "swift/AST/Types.h"
#include "llvm/Support/ConvertUTF.h"

#include "Plugins/Language/ObjC/Cocoa.h"
#include "Plugins/Language/ObjC/NSDictionary.h"
#include "Plugins/Language/ObjC/NSSet.h"
#include "Plugins/Language/ObjC/NSString.h"

using namespace lldb;
using namespace lldb_private;

#ifndef LLDB_DISABLE_PYTHON
using lldb_private::formatters::AddCXXSummary;
using lldb_private::formatters::AddCXXSynthetic;
#endif
using lldb_private::formatters::AddFormat;
using lldb_private::formatters::AddStringSummary;
using lldb_private::formatters::AddSummary;

void SwiftLanguage::Initialize() {
  static ConstString g_NSDictionaryClass1(
      "_TtCSs29_NativeDictionaryStorageOwner");
  static ConstString g_NSDictionaryClass2(
      "_TtCs29_NativeDictionaryStorageOwner");
  static ConstString g_NSDictionaryClass3(
      "_TtGCs29_NativeDictionaryStorageOwner");
  static ConstString g_NSSetClass1("_TtCSs22_NativeSetStorageOwner");
  static ConstString g_NSSetClass2("_TtCs22_NativeSetStorageOwner");
  static ConstString g_NSStringClass1("_NSContiguousString");
  static ConstString g_NSStringClass2("_TtCSs19_NSContiguousString");
  static ConstString g_NSStringClass3("_TtCs19_NSContiguousString");
  static ConstString g_NSArrayClass1("_TtCs21_SwiftDeferredNSArray");

  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Swift Language",
                                CreateInstance);

  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSummaries()
      .push_back({lldb_private::formatters::NSDictionary_Additionals::
                      AdditionalFormatterMatching()
                          .GetFullMatch(g_NSDictionaryClass1),
                  lldb_private::formatters::swift::Dictionary_SummaryProvider});
  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSummaries()
      .push_back({lldb_private::formatters::NSDictionary_Additionals::
                      AdditionalFormatterMatching()
                          .GetFullMatch(g_NSDictionaryClass2),
                  lldb_private::formatters::swift::Dictionary_SummaryProvider});
  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSummaries()
  .push_back({lldb_private::formatters::NSDictionary_Additionals::
    AdditionalFormatterMatching()
    .GetPrefixMatch(g_NSDictionaryClass3),
    lldb_private::formatters::swift::Dictionary_SummaryProvider});
  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSynthetics()
      .push_back({lldb_private::formatters::NSDictionary_Additionals::
                      AdditionalFormatterMatching()
                          .GetFullMatch(g_NSDictionaryClass1),
                  lldb_private::formatters::swift::
                      DictionarySyntheticFrontEndCreator});
  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSynthetics()
      .push_back({lldb_private::formatters::NSDictionary_Additionals::
                      AdditionalFormatterMatching()
                          .GetFullMatch(g_NSDictionaryClass2),
                  lldb_private::formatters::swift::
                      DictionarySyntheticFrontEndCreator});
  lldb_private::formatters::NSDictionary_Additionals::GetAdditionalSynthetics()
  .push_back({lldb_private::formatters::NSDictionary_Additionals::
    AdditionalFormatterMatching()
    .GetPrefixMatch(g_NSDictionaryClass3),
    lldb_private::formatters::swift::
    DictionarySyntheticFrontEndCreator});

  lldb_private::formatters::NSSet_Additionals::GetAdditionalSummaries().emplace(
      g_NSSetClass1, lldb_private::formatters::swift::Set_SummaryProvider);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSynthetics()
      .emplace(g_NSSetClass1,
               lldb_private::formatters::swift::SetSyntheticFrontEndCreator);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSummaries().emplace(
      g_NSSetClass2, lldb_private::formatters::swift::Set_SummaryProvider);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSynthetics()
      .emplace(g_NSSetClass2,
               lldb_private::formatters::swift::SetSyntheticFrontEndCreator);

  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .emplace(
          g_NSStringClass1,
          lldb_private::formatters::swift::NSContiguousString_SummaryProvider);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .emplace(
          g_NSStringClass2,
          lldb_private::formatters::swift::NSContiguousString_SummaryProvider);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .emplace(
          g_NSStringClass3,
          lldb_private::formatters::swift::NSContiguousString_SummaryProvider);

  lldb_private::formatters::NSArray_Additionals::GetAdditionalSummaries()
      .emplace(g_NSArrayClass1,
               lldb_private::formatters::swift::Array_SummaryProvider);
  lldb_private::formatters::NSArray_Additionals::GetAdditionalSynthetics()
      .emplace(g_NSArrayClass1,
               lldb_private::formatters::swift::ArraySyntheticFrontEndCreator);
}

void SwiftLanguage::Terminate() {
  static ConstString g_NSDictionaryClass1(
      "_TtCSs29_NativeDictionaryStorageOwner");
  static ConstString g_NSDictionaryClass2(
      "_TtCs29_NativeDictionaryStorageOwner");
  static ConstString g_NSSetClass1("_TtCSs22_NativeSetStorageOwner");
  static ConstString g_NSSetClass2("_TtCs22_NativeSetStorageOwner");
  static ConstString g_NSStringClass1("_NSContiguousString");
  static ConstString g_NSStringClass2("_TtCSs19_NSContiguousString");
  static ConstString g_NSStringClass3("_TtCs19_NSContiguousString");
  static ConstString g_NSArrayClass1("_TtCs21_SwiftDeferredNSArray");

  lldb_private::formatters::NSSet_Additionals::GetAdditionalSummaries().erase(
      g_NSSetClass1);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSynthetics().erase(
      g_NSSetClass1);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSummaries().erase(
      g_NSSetClass2);
  lldb_private::formatters::NSSet_Additionals::GetAdditionalSynthetics().erase(
      g_NSSetClass2);

  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .erase(g_NSStringClass1);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .erase(g_NSStringClass2);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .erase(g_NSStringClass3);

  lldb_private::formatters::NSArray_Additionals::GetAdditionalSummaries().erase(
      g_NSArrayClass1);
  lldb_private::formatters::NSArray_Additionals::GetAdditionalSynthetics()
      .erase(g_NSArrayClass1);

  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString SwiftLanguage::GetPluginNameStatic() {
  static ConstString g_name("swift");
  return g_name;
}

bool SwiftLanguage::IsTopLevelFunction(Function &function) {
  static ConstString g_main("main");

  if (CompileUnit *comp_unit = function.GetCompileUnit()) {
    if (comp_unit->GetLanguage() == lldb::eLanguageTypeSwift) {
      if (function.GetMangled().GetMangledName() == g_main)
        return true;
    }
  }

  return false;
}

static void LoadSwiftFormatters(lldb::TypeCategoryImplSP swift_category_sp) {
  if (!swift_category_sp)
    return;

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetCascades(true)
      .SetSkipPointers(true)
      .SetSkipReferences(true)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  TypeFormatImpl::Flags format_flags;
  format_flags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(true);

  SyntheticChildren::Flags basic_synth_flags;
  basic_synth_flags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(
      true);

#ifndef LLDB_DISABLE_PYTHON
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::ObjC_Selector_SummaryProvider,
                "ObjectiveC.Selector", ConstString("ObjectiveC.Selector"),
                summary_flags);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Int64", ConstString("Swift.Int64"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Int32", ConstString("Swift.Int32"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Int16", ConstString("Swift.Int16"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Int8", ConstString("Swift.Int8"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Int", ConstString("Swift.Int"), basic_synth_flags);

  AddFormat(swift_category_sp, lldb::eFormatDecimal, ConstString("Swift.Int64"),
            format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatDecimal, ConstString("Swift.Int32"),
            format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatDecimal, ConstString("Swift.Int16"),
            format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatDecimal, ConstString("Swift.Int8"),
            format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatDecimal, ConstString("Swift.Int"),
            format_flags, false);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UInt64", ConstString("Swift.UInt64"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UInt32", ConstString("Swift.UInt32"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UInt16", ConstString("Swift.UInt16"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UInt8", ConstString("Swift.UInt8"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UInt", ConstString("Swift.UInt"), basic_synth_flags);

  AddFormat(swift_category_sp, lldb::eFormatUnsigned,
            ConstString("Swift.UInt64"), format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatUnsigned,
            ConstString("Swift.UInt32"), format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatUnsigned,
            ConstString("Swift.UInt16"), format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatUnsigned,
            ConstString("Swift.UInt8"), format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatUnsigned, ConstString("Swift.UInt"),
            format_flags, false);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Float32", ConstString("Swift.Float32"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Float64", ConstString("Swift.Float64"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Float80", ConstString("Swift.Float80"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Float", ConstString("Swift.Float"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.Double", ConstString("Swift.Double"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.CDouble", ConstString("Swift.CDouble"), basic_synth_flags);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Darwin.size_t", ConstString("Darwin.size_t"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.OpaquePointer", ConstString("Swift.OpaquePointer"),
      basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UWord", ConstString("Swift.UWord"), basic_synth_flags);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UnsafePointer", ConstString("^Swift.UnsafePointer<.+>$"),
      basic_synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator,
      "Swift.UnsafeMutablePointer",
      ConstString("^Swift.UnsafeMutablePointer<.+>$"), basic_synth_flags, true);

  AddFormat(swift_category_sp, lldb::eFormatPointer,
            ConstString("Swift.OpaquePointer"), format_flags, false);
  AddFormat(swift_category_sp, lldb::eFormatPointer,
            ConstString("^Swift.UnsafePointer<.+>$"), format_flags, true);
  AddFormat(swift_category_sp, lldb::eFormatPointer,
            ConstString("^Swift.UnsafeMutablePointer<.+>$"), format_flags,
            true);
  AddFormat(swift_category_sp, lldb::eFormatUnsigned,
            ConstString("Swift.UWord"), format_flags, false);

  SyntheticChildren::Flags synth_flags;
  synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);

  // Arrays and Dictionaries want their children shown.. that's the whole point
  summary_flags.SetDontShowChildren(false);
  summary_flags.SetSkipPointers(false); // the ContiguousArrayStorage ugliness
                                        // will come back to us as pointers
  synth_flags.SetSkipPointers(false);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Array_SummaryProvider,
                "Swift.Array summary provider",
                ConstString("^Swift.Array<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Array_SummaryProvider,
                "Swift.Array summary provider",
                ConstString("Swift._NSSwiftArray"), summary_flags, false);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Array_SummaryProvider,
                "Swift.Array summary provider",
                ConstString("^Swift.NativeArray<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Array_SummaryProvider,
                "Swift.ArraySlice summary provider",
                ConstString("^Swift.ArraySlice<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Array_SummaryProvider,
                "Swift.Array summary provider",
                ConstString("^_TtCs23_ContiguousArrayStorage[A-Fa-f0-9]+$"),
                summary_flags, true);
  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Array_SummaryProvider,
      "Swift.Array summary provider",
      ConstString("^Swift._ContiguousArrayStorage"), summary_flags, true);
  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "Swift.Array summary provider",
      ConstString("_TtCs21_SwiftDeferredNSArray"), summary_flags, false);

  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Dictionary_SummaryProvider,
                "Swift.Dictionary summary provider",
                ConstString("^Swift.Dictionary<.+,.+>$"), summary_flags, true);
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::NSDictionarySummaryProvider<false>,
      "Swift.Dictionary synthetic children",
      ConstString("^_TtCs29_NativeDictionaryStorageOwner[A-Fa-f0-9]+$"),
      summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "Swift.Dictionary synthetic children",
                ConstString("^_TtGCs29_NativeDictionaryStorageOwner.*_$"),
                summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "Swift.Dictionary synthetic children",
                ConstString("^Swift._NativeDictionaryStorageOwner.*$"),
                summary_flags, true);

  summary_flags.SetDontShowChildren(true);
  summary_flags.SetSkipPointers(true);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children", ConstString("^Swift.Array<.+>$"),
      synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children", ConstString("Swift._NSSwiftArray"),
      synth_flags, false);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children", ConstString("^Swift.NativeArray<.+>$"),
      synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children", ConstString("^Swift.ArraySlice<.+>$"),
      synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children",
      ConstString("^_TtCs23_ContiguousArrayStorage[A-Fa-f0-9]+$"), synth_flags,
      true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children",
      ConstString("^Swift._ContiguousArrayStorage"), synth_flags, true);
  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "Swift.Array synthetic children",
                  ConstString("_TtCs21_SwiftDeferredNSArray"), synth_flags,
                  false);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::DictionarySyntheticFrontEndCreator,
      "Swift.Dictionary synthetic children",
      ConstString("^Swift.Dictionary<.+,.+>$"), synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "Swift.Dictionary synthetic children",
      ConstString("^_TtCs29_NativeDictionaryStorageOwner[A-Fa-f0-9]+$"),
      synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "Swift.Dictionary synthetic children",
      ConstString("^_TtGCs29_NativeDictionaryStorageOwner.*_$"), synth_flags,
      true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "Swift.Dictionary synthetic children",
      ConstString("^Swift._NativeDictionaryStorageOwner.*$"), synth_flags,
      true);

  // FIXME: _Set and Set seem to be coexisting on different trains - support
  // both for a while
  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::swift::SetSyntheticFrontEndCreator,
                  "Swift.Set synthetic children",
                  ConstString("^Swift.Set<.+>$"), synth_flags, true);
  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::swift::SetSyntheticFrontEndCreator,
                  "Swift.Set synthetic children",
                  ConstString("^Swift._Set<.+>$"), synth_flags, true);
  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "Swift.Set synthetic children",
                  ConstString("^_TtCs22_NativeSetStorageOwner[A-Fa-f0-9]+$"),
                  synth_flags, true);

  synth_flags.SetSkipPointers(true);

  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Bool_SummaryProvider,
      "Swift.Bool summary provider", ConstString("Swift.Bool"), summary_flags);
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::TypePreservingNSNumber_SummaryProvider,
      "_SwiftTypePreservingNSNumber summary provider",
      ConstString("_SwiftTypePreservingNSNumber"), summary_flags);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::DarwinBoolean_SummaryProvider,
                "DarwinBoolean summary provider", ConstString("DarwinBoolean"),
                summary_flags);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::UnicodeScalar_SummaryProvider,
                "Swift.UnicodeScalar summary provider",
                ConstString("Swift.UnicodeScalar"), summary_flags);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Character_SummaryProvider,
                "Swift.Character summary provider",
                ConstString("Swift.Character"), summary_flags);
  bool (*string_summary_provider)(ValueObject &, Stream &,
                                  const TypeSummaryOptions &) =
      lldb_private::formatters::swift::String_SummaryProvider;
  AddCXXSummary(swift_category_sp, string_summary_provider,
                "Swift.String summary provider", ConstString("Swift.String"),
                summary_flags);
  bool (*staticstring_summary_provider)(ValueObject &, Stream &,
                                        const TypeSummaryOptions &) =
      lldb_private::formatters::swift::StaticString_SummaryProvider;
  AddCXXSummary(swift_category_sp, staticstring_summary_provider,
                "Swift.StaticString summary provider",
                ConstString("Swift.StaticString"), summary_flags);
  summary_flags.SetSkipPointers(false);
  // this is an ObjC dynamic type - as such it comes in pointer form
  // NSContiguousString* - do not skip pointers here
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::NSContiguousString_SummaryProvider,
      "NSContiguousString summary provider", ConstString("_NSContiguousString"),
      summary_flags);
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::NSContiguousString_SummaryProvider,
      "NSContiguousString summary provider",
      ConstString("_TtCs19_NSContiguousString"), summary_flags);
  summary_flags.SetSkipPointers(true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::BuiltinObjC_SummaryProvider,
                "ObjC object pointer summary provider",
                ConstString("Builtin.ObjCPointer"), summary_flags);

  TypeSummaryImpl::Flags optional_summary_flags;
  optional_summary_flags.SetCascades(true)
      .SetDontShowChildren(false) // this one will actually be calculated at
                                  // runtime, what you pass here doesn't matter
      .SetDontShowValue(true)
      .SetHideItemNames(false)
      .SetShowMembersOneLiner(false)
      .SetSkipPointers(true)
      .SetSkipReferences(false);

  SyntheticChildren::Flags optional_synth_flags;
  optional_synth_flags.SetCascades(true)
      .SetSkipPointers(true)
      .SetSkipReferences(false);

  TypeSummaryImplSP swift_optional_summary_sp(
      new lldb_private::formatters::swift::SwiftOptionalSummaryProvider(
          optional_summary_flags));
  TypeSummaryImplSP swift_unchecked_optional_summary_sp(
      new lldb_private::formatters::swift::SwiftOptionalSummaryProvider(
          optional_summary_flags));

  // do not move the relative order of these - @unchecked needs to come first or
  // else pain will ensue
  AddSummary(swift_category_sp, swift_unchecked_optional_summary_sp,
             ConstString("^Swift.ImplicitlyUnwrappedOptional<.+>$"), true);
  AddSummary(swift_category_sp, swift_optional_summary_sp,
             ConstString("^Swift.Optional<.+>$"), true);

  AddSummary(swift_category_sp, swift_unchecked_optional_summary_sp,
             ConstString("AnyObject!"), false);
  AddSummary(swift_category_sp, swift_optional_summary_sp,
             ConstString("AnyObject?"), false);

  AddSummary(swift_category_sp, swift_unchecked_optional_summary_sp,
             ConstString("()!"), false);
  AddSummary(swift_category_sp, swift_optional_summary_sp, ConstString("()?"),
             false);

  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::swift::
                      SwiftUncheckedOptionalSyntheticFrontEndCreator,
                  "Swift.Optional synthetic children",
                  ConstString("^Swift.ImplicitlyUnwrappedOptional<.+>$"),
                  optional_synth_flags, true);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEndCreator,
      "Swift.Optional synthetic children", ConstString("^Swift.Optional<.+>$"),
      optional_synth_flags, true);

  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::swift::
                      SwiftUncheckedOptionalSyntheticFrontEndCreator,
                  "Swift.Optional synthetic children",
                  ConstString("AnyObject!"), optional_synth_flags, false);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEndCreator,
      "Swift.Optional synthetic children", ConstString("AnyObject?"),
      optional_synth_flags, false);

  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::swift::
                      SwiftUncheckedOptionalSyntheticFrontEndCreator,
                  "Swift.Optional synthetic children", ConstString("()!"),
                  optional_synth_flags, false);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEndCreator,
      "Swift.Optional synthetic children", ConstString("()?"),
      optional_synth_flags, false);

  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::Range_SummaryProvider,
                "Swift.Range summary provider", ConstString("Swift.Range<.+>$"),
                summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::CountableRange_SummaryProvider,
                "Swift.CountableRange summary provider",
                ConstString("Swift.CountableRange<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::ClosedRange_SummaryProvider,
                "Swift.ClosedRange summary provider",
                ConstString("Swift.ClosedRange<.+>$"), summary_flags, true);
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::CountableClosedRange_SummaryProvider,
      "Swift.CountableClosedRange summary provider",
      ConstString("Swift.CountableClosedRange<.+>$"), summary_flags, true);

  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::StridedRangeGenerator_SummaryProvider,
      "Swift.StridedRangeGenerator summary provider",
      ConstString("Swift.StridedRangeGenerator<.+>$"), summary_flags, true);

  TypeSummaryImpl::Flags nil_summary_flags;
  nil_summary_flags.SetCascades(true)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetHideItemNames(false)
      .SetShowMembersOneLiner(false)
      .SetSkipPointers(true)
      .SetSkipReferences(false);

  AddStringSummary(swift_category_sp, "nil", ConstString("Swift._Nil"),
                   nil_summary_flags);

  AddStringSummary(swift_category_sp, "${var.native}",
                   ConstString("CoreGraphics.CGFloat"), summary_flags);
#endif // LLDB_DISABLE_PYTHON
}

static void
LoadFoundationValueTypesFormatters(lldb::TypeCategoryImplSP swift_category_sp) {
  if (!swift_category_sp)
    return;

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetCascades(true)
      .SetDontShowChildren(false)
      .SetSkipPointers(true)
      .SetSkipReferences(false)
      .SetHideItemNames(false)
      .SetShowMembersOneLiner(false);

#ifndef LLDB_DISABLE_PYTHON
  lldb_private::formatters::AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Date_SummaryProvider,
      "Foundation.Date summary provider", ConstString("Foundation.Date"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::NotificationName_SummaryProvider,
      "Notification.Name summary provider",
      ConstString("Foundation.Notification.Type.Name"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));
  lldb_private::formatters::AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::NotificationName_SummaryProvider,
      "Notification.Name summary provider",
      ConstString("Foundation.Notification.Name"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::URL_SummaryProvider,
      "URL summary provider", ConstString("Foundation.URL"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::IndexPath_SummaryProvider,
      "IndexPath summary provider", ConstString("Foundation.IndexPath"),
      summary_flags);

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::Measurement_SummaryProvider,
      "Measurement summary provider",
      ConstString("Foundation.Measurement<Foundation.Unit>"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::UUID_SummaryProvider,
      "UUID summary provider", ConstString("Foundation.UUID"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Data_SummaryProvider,
      "Data summary provider", ConstString("Foundation.Data"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::URLComponentsSyntheticFrontEndCreator,
      "URLComponents synthetic children",
      ConstString("Foundation.URLComponents"), SyntheticChildren::Flags()
                                                   .SetSkipPointers(true)
                                                   .SetCascades(true)
                                                   .SetSkipReferences(false)
                                                   .SetNonCacheable(false));
#endif
}

lldb::TypeCategoryImplSP SwiftLanguage::GetFormatters() {
  static std::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  std::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(GetPluginName(), g_category);
    if (g_category) {
      LoadSwiftFormatters(g_category);
      LoadFoundationValueTypesFormatters(g_category);
    }
  });
  return g_category;
}

HardcodedFormatters::HardcodedSummaryFinder
SwiftLanguage::GetHardcodedSummaries() {
  static std::once_flag g_initialize;
  static HardcodedFormatters::HardcodedSummaryFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType,
                              FormatManager &) -> lldb::TypeSummaryImplSP {
      static lldb::TypeSummaryImplSP swift_metatype_summary_sp(nullptr);
      static ConstString g_RawPointerType("Builtin.RawPointer");
      CompilerType type(valobj.GetCompilerType());
      Flags type_flags(type.GetTypeInfo());
      if (type_flags.AllSet(eTypeIsMetatype | eTypeIsSwift)) {
        if (!swift_metatype_summary_sp.get()) {
          TypeSummaryImpl::Flags flags;
          flags.SetCascades(true)
              .SetDontShowChildren(true)
              .SetDontShowValue(true)
              .SetHideItemNames(false)
              .SetShowMembersOneLiner(false)
              .SetSkipPointers(false)
              .SetSkipReferences(false);
          swift_metatype_summary_sp.reset(new CXXFunctionSummaryFormat(
              flags,
              lldb_private::formatters::swift::SwiftMetatype_SummaryProvider,
              "Swift Metatype Summary"));
        }
        return swift_metatype_summary_sp;
      }
      if (valobj.GetName().GetLength() > 12 &&
          valobj.GetName().GetStringRef().startswith("$swift.type.") &&
          type.GetTypeName() == g_RawPointerType) {
        if (!swift_metatype_summary_sp.get()) {
          TypeSummaryImpl::Flags flags;
          flags.SetCascades(true)
              .SetDontShowChildren(true)
              .SetDontShowValue(true)
              .SetHideItemNames(false)
              .SetShowMembersOneLiner(false)
              .SetSkipPointers(false)
              .SetSkipReferences(false);
          swift_metatype_summary_sp.reset(new CXXFunctionSummaryFormat(
              flags,
              lldb_private::formatters::swift::SwiftMetatype_SummaryProvider,
              "Swift Metatype Summary"));
        }
        return swift_metatype_summary_sp;
      }
      return nullptr;
    });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType, FormatManager &)
                               -> TypeSummaryImpl::SharedPointer {
      CompilerType clang_type(valobj.GetCompilerType());
      if (lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
              WouldEvenConsiderFormatting(clang_type)) {
        TypeSummaryImpl::SharedPointer formatter_sp(
            new lldb_private::formatters::swift::SwiftOptionSetSummaryProvider(
                clang_type));
        return formatter_sp;
      }
      return nullptr;
    });
  });

  return g_formatters;
}

HardcodedFormatters::HardcodedSyntheticFinder
SwiftLanguage::GetHardcodedSynthetics() {
  static std::once_flag g_initialize;
  static ConstString g_runtime_synths_category_name("runtime-synthetics");
  static HardcodedFormatters::HardcodedSyntheticFinder g_formatters;

  std::call_once(g_initialize, []() -> void {
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType,
                              FormatManager &) -> lldb::SyntheticChildrenSP {
      static lldb::SyntheticChildrenSP swift_enum_synth(nullptr);
      CompilerType type(valobj.GetCompilerType());
      Flags type_flags(type.GetTypeInfo());
      if (type_flags.AllSet(eTypeIsSwift | eTypeIsEnumeration)) {
        if (!swift_enum_synth)
          swift_enum_synth = lldb::SyntheticChildrenSP(new CXXSyntheticChildren(
              SyntheticChildren::Flags()
                  .SetCascades(true)
                  .SetSkipPointers(true)
                  .SetSkipReferences(true),
              "swift Enum synthetic children provider",
              lldb_private::formatters::swift::EnumSyntheticFrontEndCreator));
        return swift_enum_synth;
      }
      return nullptr;
    });
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType dyn_type,
           FormatManager &format_manager) -> lldb::SyntheticChildrenSP {
          struct IsEligible {
            static bool Check(ValueObject &valobj, const CompilerType &type) {
              bool is_imported = false;

              if (type.IsValid()) {
                SwiftASTContext *swift_ast_ctx =
                    llvm::dyn_cast_or_null<SwiftASTContext>(
                        type.GetTypeSystem());
                if (swift_ast_ctx &&
                    swift_ast_ctx->IsImportedType(type, nullptr))
                  is_imported = true;
              }

              if (is_imported && type.GetNumFields() == 0)
                return true;
              if (valobj.IsBaseClass() && type.IsRuntimeGeneratedType()) {
                auto parent(valobj.GetParent());
                if (!parent)
                  return false;
                return IsEligible::Check(*parent, parent->GetCompilerType());
              }
              return false;
            }
          };

          if (dyn_type == lldb::eNoDynamicValues)
            return nullptr;
          CompilerType type(valobj.GetCompilerType());
          const bool can_create = false;
          auto category_sp(format_manager.GetCategory(
              g_runtime_synths_category_name, can_create));
          if (!category_sp || category_sp->IsEnabled() == false)
            return nullptr;
          if (IsEligible::Check(valobj, type)) {
            ProcessSP process_sp(valobj.GetProcessSP());
            if (!process_sp)
              return nullptr;
            ObjCLanguageRuntime *objc_runtime =
                process_sp->GetObjCLanguageRuntime();
            ObjCLanguageRuntime::ClassDescriptorSP valobj_descriptor_sp =
                objc_runtime->GetClassDescriptor(valobj);
            if (valobj_descriptor_sp) {
              return SyntheticChildrenSP(
                  new ObjCRuntimeSyntheticProvider(SyntheticChildren::Flags()
                                                       .SetCascades(true)
                                                       .SetSkipPointers(true)
                                                       .SetSkipReferences(true)
                                                       .SetNonCacheable(true),
                                                   valobj_descriptor_sp));
            }
          }
          return nullptr;
        });
    g_formatters.push_back(
        [](lldb_private::ValueObject &valobj, lldb::DynamicValueType dyn_type,
           FormatManager &format_manager) -> lldb::SyntheticChildrenSP {
          struct IsEligible {
            static bool Check(const CompilerType &type) {
              if ((ClangASTContext::IsObjCObjectPointerType(type) ||
                   ClangASTContext::IsObjCObjectOrInterfaceType(type)) &&
                  type.GetTypeName().GetStringRef().startswith("_TtC"))
                return true;

              return false;
            }
          };

          if (dyn_type == lldb::eNoDynamicValues)
            return nullptr;
          CompilerType type(valobj.GetCompilerType());
          if (IsEligible::Check(type)) {
            ProcessSP process_sp(valobj.GetProcessSP());
            if (!process_sp)
              return nullptr;
            SwiftLanguageRuntime *swift_runtime =
                process_sp->GetSwiftLanguageRuntime();
            return swift_runtime->GetBridgedSyntheticChildProvider(valobj);
          }
          return nullptr;
        });
  });

  return g_formatters;
}

std::vector<ConstString> SwiftLanguage::GetPossibleFormattersMatches(
    ValueObject &valobj, lldb::DynamicValueType use_dynamic) {
  std::vector<ConstString> result;

  if (use_dynamic == lldb::eNoDynamicValues)
    return result;

  CompilerType compiler_type(valobj.GetCompilerType());

  const bool check_cpp = false;
  const bool check_objc = false;
  const bool check_swift = true;
  bool canBeSwiftDynamic = compiler_type.IsPossibleDynamicType(
      nullptr, check_cpp, check_objc, check_swift);

  if (canBeSwiftDynamic) {
    do {
      lldb::ProcessSP process_sp = valobj.GetProcessSP();
      if (!process_sp)
        break;
      SwiftLanguageRuntime *runtime = process_sp->GetSwiftLanguageRuntime();
      if (runtime == nullptr)
        break;
      TypeAndOrName type_and_or_name;
      Address address;
      Value::ValueType value_type;
      if (!runtime->GetDynamicTypeAndAddress(
              valobj, use_dynamic, type_and_or_name, address, value_type))
        break;
      if (ConstString name = type_and_or_name.GetName())
        result.push_back(name);
    } while (false);
  }

#if 0
    if (llvm::isa<SwiftASTContext>(compiler_type.GetTypeSystem()))
        result.push_back(compiler_type.GetMangledTypeName());
#endif

  return result;
}

static char32_t ConvertUTF8ToCodePoint(unsigned char c0, unsigned char c1) {
  return (c0 - 192) * 64 + (c1 - 128);
}
static char32_t ConvertUTF8ToCodePoint(unsigned char c0, unsigned char c1,
                                       unsigned char c2) {
  return (c0 - 224) * 4096 + (c1 - 128) * 64 + (c2 - 128);
}
static char32_t ConvertUTF8ToCodePoint(unsigned char c0, unsigned char c1,
                                       unsigned char c2, unsigned char c3) {
  return (c0 - 240) * 262144 + (c2 - 128) * 4096 + (c2 - 128) * 64 + (c3 - 128);
}

lldb_private::formatters::StringPrinter::EscapingHelper
SwiftLanguage::GetStringPrinterEscapingHelper(
    lldb_private::formatters::StringPrinter::GetPrintableElementType
        elem_type) {
  switch (elem_type) {
  case lldb_private::formatters::StringPrinter::GetPrintableElementType::UTF8:
    return
        [this](uint8_t *buffer, uint8_t *buffer_end,
               uint8_t *&next) -> lldb_private::formatters::StringPrinter::
            StringPrinterBufferPointer<> {
              lldb_private::formatters::StringPrinter::
                  StringPrinterBufferPointer<>
                      retval{nullptr};

              auto isprint32 = [](char32_t codepoint) -> bool {
                if (codepoint <= 0x1F || codepoint == 0x7F) // C0
                {
                  return false;
                }
                if (codepoint >= 0x80 && codepoint <= 0x9F) // C1
                {
                  return false;
                }
                if (codepoint == 0x2028 ||
                    codepoint == 0x2029) // line/paragraph separators
                {
                  return false;
                }
                if (codepoint == 0x200E || codepoint == 0x200F ||
                    (codepoint >= 0x202A &&
                     codepoint <= 0x202E)) // bidirectional text control
                {
                  return false;
                }
                if (codepoint >= 0xFFF9 &&
                    codepoint <= 0xFFFF) // interlinears and generally specials
                {
                  return false;
                }
                return true;
              };

              unsigned utf8_encoded_len = getNumBytesForUTF8(*buffer);

              if (1 + buffer_end - buffer < utf8_encoded_len) {
                // I don't have enough bytes - print whatever I have left
                retval = {buffer, static_cast<size_t>(1 + buffer_end - buffer)};
                next = buffer_end + 1;
                return retval;
              }

              char32_t codepoint = 0;
              switch (utf8_encoded_len) {
              case 1:
                return GetStringPrinterEscapingHelper(
                    lldb_private::formatters::StringPrinter::
                        GetPrintableElementType::ASCII)(buffer, buffer_end,
                                                        next);
              case 2:
                codepoint = ConvertUTF8ToCodePoint(
                    (unsigned char)*buffer, (unsigned char)*(buffer + 1));
                break;
              case 3:
                codepoint = ConvertUTF8ToCodePoint(
                    (unsigned char)*buffer, (unsigned char)*(buffer + 1),
                    (unsigned char)*(buffer + 2));
                break;
              case 4:
                codepoint = ConvertUTF8ToCodePoint(
                    (unsigned char)*buffer, (unsigned char)*(buffer + 1),
                    (unsigned char)*(buffer + 2), (unsigned char)*(buffer + 3));
                break;
              default:
                // this is probably some bogus non-character thing
                // just print it as-is and hope to sync up again soon
                retval = {buffer, 1};
                next = buffer + 1;
                return retval;
              }

              if (codepoint) {
                switch (codepoint) {
                case 0:
                  retval = {"\\0", 2};
                  break;
                case '\a':
                  retval = {"\\a", 2};
                  break;
                case '\n':
                  retval = {"\\n", 2};
                  break;
                case '\r':
                  retval = {"\\r", 2};
                  break;
                case '\t':
                  retval = {"\\t", 2};
                  break;
                case '"':
                  retval = {"\\\"", 2};
                  break;
                case '\'':
                  retval = {"\\'", 2};
                  break;
                case '\\':
                  retval = {"\\\\", 2};
                  break;
                default:
                  if (isprint32(codepoint))
                    retval = {buffer, utf8_encoded_len};
                  else {
                    uint8_t *data = new uint8_t[13];
                    unsigned long data_len =
                        sprintf((char *)data, "\\u{%x}", *buffer);
                    retval = {data, data_len,
                              [](const uint8_t *c) { delete[] c; }};
                    break;
                  }
                }

                next = buffer + utf8_encoded_len;
                return retval;
              }

              // this should not happen - but just in case.. try to resync at
              // some point
              retval = {buffer, 1};
              next = buffer + 1;
              return retval;
            };
  case lldb_private::formatters::StringPrinter::GetPrintableElementType::ASCII:
    return [](uint8_t *buffer, uint8_t *buffer_end,
              uint8_t *&next) -> lldb_private::formatters::StringPrinter::
               StringPrinterBufferPointer<> {
                 lldb_private::formatters::StringPrinter::
                     StringPrinterBufferPointer<>
                         retval = {nullptr};

                 switch (*buffer) {
                 case 0:
                   retval = {"\\0", 2};
                   break;
                 case '\a':
                   retval = {"\\a", 2};
                   break;
                 case '\n':
                   retval = {"\\n", 2};
                   break;
                 case '\r':
                   retval = {"\\r", 2};
                   break;
                 case '\t':
                   retval = {"\\t", 2};
                   break;
                 case '"':
                   retval = {"\\\"", 2};
                   break;
                 case '\'':
                   retval = {"\\'", 2};
                   break;
                 case '\\':
                   retval = {"\\\\", 2};
                   break;
                 default:
                   if (isprint(*buffer))
                     retval = {buffer, 1};
                   else {
                     uint8_t *data = new uint8_t[7];
                     unsigned long data_len =
                         sprintf((char *)data, "\\u{%x}", *buffer);
                     retval = {data, data_len,
                               [](const uint8_t *c) { delete[] c; }};
                     break;
                   }
                 }

                 next = buffer + 1;
                 return retval;
               };
  }
}

static void SplitDottedName(llvm::StringRef name,
                            std::vector<llvm::StringRef> &parts,
                            char sep = '.') {
  if (name.empty())
    return;
  auto pair = name.split(sep);
  if (false == pair.first.empty())
    parts.push_back(pair.first);
  if (false == pair.second.empty())
    SplitDottedName(pair.second, parts, sep);
}

std::unique_ptr<Language::TypeScavenger> SwiftLanguage::GetTypeScavenger() {
  class SwiftTypeScavenger : public Language::TypeScavenger {
  private:
    typedef SwiftASTContext::TypeOrDecl TypeOrDecl;
    typedef SwiftASTContext::TypesOrDecls TypesOrDecls;

    class SwiftScavengerResult : public Language::TypeScavenger::Result {
    public:
      typedef SwiftASTContext::TypeOrDecl TypeOrDecl;

      SwiftScavengerResult(TypeOrDecl type)
          : Language::TypeScavenger::Result(), m_result(type) {}

      bool IsValid() override { return m_result.operator bool(); }

      bool DumpToStream(Stream &stream, bool print_help_if_available) override {
        if (IsValid()) {
          auto as_type = m_result.GetAs<CompilerType>();
          auto as_decl = m_result.GetAs<swift::Decl *>();

          if (as_type.hasValue() && as_type.getValue()) {
            TypeSystem *type_system = as_type->GetTypeSystem();
            if (SwiftASTContext *swift_ast_ctx =
                    llvm::dyn_cast_or_null<SwiftASTContext>(type_system))
              swift_ast_ctx->DumpTypeDescription(as_type->GetOpaqueQualType(),
                                                 &stream,
                                                 print_help_if_available, true);
            else
              as_type->DumpTypeDescription(
                  &stream); // we should always have a swift type here..
          } else if (as_decl.hasValue() && as_decl.getValue()) {
            std::string buffer;
            llvm::raw_string_ostream str_stream(buffer);
            swift::Decl *decl = as_decl.getValue();
            decl->print(str_stream,
                        SwiftASTContext::GetUserVisibleTypePrintingOptions(
                            print_help_if_available));
            str_stream.flush();
            stream.Printf("%s", buffer.c_str());
          }

          stream.EOL();
          return true;
        }
        return false;
      }

      virtual ~SwiftScavengerResult() = default;

    private:
      TypeOrDecl m_result;
    };

  protected:
    SwiftTypeScavenger() = default;

    virtual ~SwiftTypeScavenger() = default;

    typedef std::function<size_t(const char *, ExecutionContextScope *,
                                 TypesOrDecls &)>
        Hoarder;
    typedef std::vector<Hoarder> Hoarders;

    static Hoarders &GetHoarders() {
      static Hoarders g_hoarders;
      static std::once_flag g_init;
      std::call_once(g_init, []() -> void {
        g_hoarders.push_back([](const char *input,
                                ExecutionContextScope *exe_scope,
                                TypesOrDecls &results) -> size_t {
          size_t before = results.size();

          if (exe_scope) {
            Target *target = exe_scope->CalculateTarget().get();
            if (target) {
              const bool create_on_demand = false;
              Error error;
              SwiftASTContext *ast_ctx(
                  target->GetScratchSwiftASTContext(error, create_on_demand));
              if (ast_ctx) {
                const bool is_mangled = true;
                Mangled mangled(ConstString(input), is_mangled);
                if (mangled.GuessLanguage() == eLanguageTypeSwift) {
                  Error error;
                  auto candidate =
                      ast_ctx->GetTypeFromMangledTypename(input, error);
                  if (candidate.IsValid() && error.Success())
                    results.insert(candidate);
                }
              }
            }
          }

          return (results.size() - before);
        });
        g_hoarders.push_back([](const char *input,
                                ExecutionContextScope *exe_scope,
                                TypesOrDecls &results) -> size_t {
          size_t before = results.size();

          if (exe_scope) {
            Target *target = exe_scope->CalculateTarget().get();
            StackFrame *frame = exe_scope->CalculateStackFrame().get();
            if (target && frame) {
              lldb::ValueObjectSP result_sp;
              EvaluateExpressionOptions options;
              options.SetLanguage(eLanguageTypeSwift);
              options.SetGenerateDebugInfo(false);
              StreamString stream;
              stream.Printf("typealias __lldb__typelookup_typealias = %s; "
                            "__lldb__typelookup_typealias.self",
                            input);
              if (target->EvaluateExpression(stream.GetData(), frame, result_sp,
                                             options) == eExpressionCompleted) {
                if (result_sp && result_sp->GetCompilerType().IsValid()) {
                  CompilerType result_type(result_sp->GetCompilerType());
                  if (Flags(result_type.GetTypeInfo())
                          .AllSet(eTypeIsSwift | eTypeIsMetatype))
                    result_type = result_type.GetInstanceType();
                  results.insert(TypeOrDecl(result_type));
                }
              }
            }
          }

          return (results.size() - before);
        });
        g_hoarders.push_back([](const char *input,
                                ExecutionContextScope *exe_scope,
                                TypesOrDecls &results) -> size_t {
          size_t before = results.size();

          if (exe_scope) {
            Target *target = exe_scope->CalculateTarget().get();
            const bool create_on_demand = false;
            Error error;
            SwiftASTContext *ast_ctx(
                target->GetScratchSwiftASTContext(error, create_on_demand));
            if (ast_ctx) {
              auto iter = ast_ctx->GetModuleCache().begin(),
                   end = ast_ctx->GetModuleCache().end();

              std::vector<llvm::StringRef> name_parts;
              SplitDottedName(input, name_parts);

              std::function<void(swift::ModuleDecl *)> lookup_func =
                  [ast_ctx, input, name_parts,
                   &results](swift::ModuleDecl *module) -> void {

                swift::Module::AccessPathTy access_path;

                module->forAllVisibleModules(
                    access_path, true,
                    [ast_ctx, input, name_parts, &results](
                        swift::Module::ImportedModule imported_module) -> bool {
                      auto module = imported_module.second;
                      TypesOrDecls local_results;
                      ast_ctx->FindTypesOrDecls(input, module, local_results,
                                                false);
                      llvm::Optional<TypeOrDecl> candidate;
                      if (local_results.empty() && name_parts.size() > 1) {
                        size_t idx_of_deeper = 1;
                        // if you're looking for Swift.Int in module Swift, try
                        // looking for Int
                        if (name_parts.front() == module->getName().str()) {
                          candidate = ast_ctx->FindTypeOrDecl(
                              name_parts[1].str().c_str(), module);
                          idx_of_deeper = 2;
                        }
                        // this is probably the top-level name of a nested type
                        // String.UTF8View
                        else {
                          candidate = ast_ctx->FindTypeOrDecl(
                              name_parts[0].str().c_str(), module);
                        }
                        if (candidate.hasValue()) {
                          TypesOrDecls candidates{candidate.getValue()};
                          for (; idx_of_deeper < name_parts.size();
                               idx_of_deeper++) {
                            TypesOrDecls new_candidates;
                            for (auto candidate : candidates) {
                              ast_ctx->FindContainedTypeOrDecl(
                                  name_parts[idx_of_deeper], candidate,
                                  new_candidates);
                            }
                            candidates = new_candidates;
                          }
                          for (auto candidate : candidates) {
                            if (candidate)
                              results.insert(candidate);
                          }
                        }
                      } else if (local_results.size() > 0) {
                        for (const auto &result : local_results)
                          results.insert(result);
                      } else if (local_results.empty() && module &&
                                 name_parts.size() == 1 &&
                                 name_parts.front() == module->getName().str())
                        results.insert(
                            CompilerType(ast_ctx->GetASTContext(),
                                         swift::ModuleType::get(module)));
                      return true;
                    });

              };

              for (; iter != end; iter++)
                lookup_func(iter->second);
            }
          }

          return (results.size() - before);
        });
      });

      return g_hoarders;
    }

    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &result_set) override {
      Hoarders &hoarders(GetHoarders());
      TypesOrDecls types_or_decls;

      for (auto &hoarder : hoarders) {
        hoarder(key, exe_scope, types_or_decls);
        if (types_or_decls.size())
          break;
      }

      bool any_found = false;

      for (TypeOrDecl type_or_decl : types_or_decls) {
        any_found = true;
        std::unique_ptr<Language::TypeScavenger::Result> result(
            new SwiftScavengerResult(type_or_decl));
        result_set.insert(std::move(result));
      }

      return any_found;
    }

    friend class SwiftLanguage;
  };

  return std::unique_ptr<TypeScavenger>(new SwiftTypeScavenger());
}

const char *SwiftLanguage::GetLanguageSpecificTypeLookupHelp() {
  return "\nFor Swift, in addition to a simple type name (such as String, Int, "
         "NSObject, ..), one can also provide:\n"
         "- a mangled type name (e.g. _TtSi)\n"
         "- the name of a function, even if multiple overloads of it exist\n"
         "- the name of an operator\n"
         "- the name of a module available in the current target, which will "
         "print all types and declarations available in that module";
}

bool SwiftLanguage::GetFormatterPrefixSuffix(ValueObject &valobj,
                                             ConstString type_hint,
                                             std::string &prefix,
                                             std::string &suffix) {
  static ConstString g_NSNumberChar("NSNumber:char");
  static ConstString g_NSNumberShort("NSNumber:short");
  static ConstString g_NSNumberInt("NSNumber:int");
  static ConstString g_NSNumberLong("NSNumber:long");
  static ConstString g_NSNumberFloat("NSNumber:float");
  static ConstString g_NSNumberDouble("NSNumber:double");

  if (type_hint.IsEmpty())
    return false;

  prefix.clear();
  suffix.clear();

  if (type_hint == g_NSNumberChar) {
    prefix = "UInt8(";
    suffix = ")";
    return true;
  }
  if (type_hint == g_NSNumberShort) {
    prefix = "Int16(";
    suffix = ")";
    return true;
  }
  if (type_hint == g_NSNumberInt) {
    prefix = "Int32(";
    suffix = ")";
    return true;
  }
  if (type_hint == g_NSNumberLong) {
    prefix = "Int64(";
    suffix = ")";
    return true;
  }
  if (type_hint == g_NSNumberFloat) {
    prefix = "Float(";
    suffix = ")";
    return true;
  }
  if (type_hint == g_NSNumberDouble) {
    prefix = "Double(";
    suffix = ")";
    return true;
  }

  return false;
}

DumpValueObjectOptions::DeclPrintingHelper
SwiftLanguage::GetDeclPrintingHelper() {
  return [](ConstString type_name, ConstString var_name,
            const DumpValueObjectOptions &options, Stream &stream) -> bool {
    std::string type_name_str(type_name ? type_name.GetCString() : "");
    if (type_name) {
      for (auto iter = type_name_str.find(" *"); iter != std::string::npos;
           iter = type_name_str.find(" *")) {
        type_name_str.erase(iter, 2);
      }
      if (!type_name_str.empty()) {
        if (type_name_str.front() != '(' || type_name_str.back() != ')') {
          type_name_str = "(" + type_name_str + ")";
        }
      }
    }

    if (!type_name_str.empty())
      stream.Printf("%s ", type_name_str.c_str());
    if (var_name)
      stream.Printf("%s =", var_name.GetCString());
    else if (!options.m_hide_name)
      stream.Printf(" =");

    return true;
  };
}

LazyBool SwiftLanguage::IsLogicalTrue(ValueObject &valobj, Error &error) {
  static ConstString g_SwiftBool("Swift.Bool");
  static ConstString g_value("_value");

  Scalar scalar_value;

  CompilerType valobj_type(valobj.GetCompilerType());
  Flags type_flags(valobj_type.GetTypeInfo());
  if (llvm::isa<SwiftASTContext>(valobj_type.GetTypeSystem())) {
    if (type_flags.AllSet(eTypeIsStructUnion) &&
        valobj_type.GetTypeName() == g_SwiftBool) {
      ValueObjectSP your_value_sp(valobj.GetChildMemberWithName(g_value, true));
      if (!your_value_sp) {
        error.SetErrorString("unexpected data layout");
        return eLazyBoolNo;
      } else {
        if (!your_value_sp->ResolveValue(scalar_value)) {
          error.SetErrorString("unexpected data layout");
          return eLazyBoolNo;
        } else {
          error.Clear();
          if (scalar_value.ULongLong(1) == 0)
            return eLazyBoolNo;
          else
            return eLazyBoolYes;
        }
      }
    }
  }

  error.SetErrorString("not a Swift boolean type");
  return eLazyBoolNo;
}

bool SwiftLanguage::IsUninitializedReference(ValueObject &valobj) {
  const uint32_t mask = eTypeIsSwift | eTypeIsClass;
  bool isSwiftClass =
      (((valobj.GetCompilerType().GetTypeInfo(nullptr)) & mask) == mask);
  if (!isSwiftClass)
    return false;
  bool canReadValue = true;
  bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
  return canReadValue && isZero;
}

bool SwiftLanguage::GetFunctionDisplayName(
    const SymbolContext *sc, const ExecutionContext *exe_ctx,
    FunctionNameRepresentation representation, Stream &s) {
  switch (representation) {
  case Language::FunctionNameRepresentation::eName:
    break; // no need to customize this
  case Language::FunctionNameRepresentation::eNameWithNoArgs: {
    if (sc->function) {
      if (sc->function->GetLanguage() == eLanguageTypeSwift) {
        if (ConstString cs = sc->function->GetDisplayName()) {
          s.Printf("%s", cs.AsCString());
          return true;
        }
      }
    }
  }
  case Language::FunctionNameRepresentation::eNameWithArgs: {
    if (sc->function) {
      if (sc->function->GetLanguage() == eLanguageTypeSwift) {
        if (const char *cstr = sc->function->GetDisplayName().AsCString()) {
          ExecutionContextScope *exe_scope =
              exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL;

          const InlineFunctionInfo *inline_info = NULL;
          VariableListSP variable_list_sp;
          bool get_function_vars = true;
          if (sc->block) {
            Block *inline_block = sc->block->GetContainingInlinedBlock();

            if (inline_block) {
              get_function_vars = false;
              inline_info = sc->block->GetInlinedFunctionInfo();
              if (inline_info)
                variable_list_sp = inline_block->GetBlockVariableList(true);
            }
          }

          if (get_function_vars) {
            variable_list_sp =
                sc->function->GetBlock(true).GetBlockVariableList(true);
          }

          if (inline_info) {
            s.PutCString(cstr);
            s.PutCString(" [inlined] ");
            cstr =
                inline_info->GetName(sc->function->GetLanguage()).GetCString();
          }

          VariableList args;
          if (variable_list_sp)
            variable_list_sp->AppendVariablesWithScope(
                eValueTypeVariableArgument, args);
          if (args.GetSize() > 0) {
            const char *open_paren = strchr(cstr, '(');
            const char *close_paren = nullptr;
            const char *generic = strchr(cstr, '<');
            // if before the arguments list begins there is a template sign
            // then scan to the end of the generic args before you try to find
            // the arguments list
            if (generic && open_paren && generic < open_paren) {
              int generic_depth = 1;
              ++generic;
              for (; *generic && generic_depth > 0; generic++) {
                if (*generic == '<')
                  generic_depth++;
                if (*generic == '>')
                  generic_depth--;
              }
              if (*generic)
                open_paren = strchr(generic, '(');
              else
                open_paren = nullptr;
            }
            if (open_paren) {
              close_paren = strchr(open_paren, ')');
            }

            if (open_paren)
              s.Write(cstr, open_paren - cstr + 1);
            else {
              s.PutCString(cstr);
              s.PutChar('(');
            }
            const size_t num_args = args.GetSize();
            for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
              std::string buffer;

              VariableSP var_sp(args.GetVariableAtIndex(arg_idx));
              ValueObjectSP var_value_sp(
                  ValueObjectVariable::Create(exe_scope, var_sp));
              StreamString ss;
              const char *var_representation = nullptr;
              const char *var_name = var_value_sp->GetName().GetCString();
              if (var_value_sp->GetCompilerType().IsValid()) {
                if (var_value_sp && exe_scope->CalculateTarget())
                  var_value_sp =
                      var_value_sp->GetQualifiedRepresentationIfAvailable(
                          exe_scope->CalculateTarget()
                              ->TargetProperties::GetPreferDynamicValue(),
                          exe_scope->CalculateTarget()
                              ->TargetProperties::GetEnableSyntheticValue());
                if (var_value_sp->GetCompilerType().IsAggregateType() &&
                    DataVisualization::ShouldPrintAsOneLiner(
                        *var_value_sp.get())) {
                  static StringSummaryFormat format(
                      TypeSummaryImpl::Flags()
                          .SetHideItemNames(false)
                          .SetShowMembersOneLiner(true),
                      "");
                  format.FormatObject(var_value_sp.get(), buffer,
                                      TypeSummaryOptions());
                  var_representation = buffer.c_str();
                } else
                  var_value_sp->DumpPrintableRepresentation(
                      ss, ValueObject::ValueObjectRepresentationStyle::
                              eValueObjectRepresentationStyleSummary,
                      eFormatDefault,
                      ValueObject::PrintableRepresentationSpecialCases::
                          ePrintableRepresentationSpecialCasesAllow,
                      false);
              }
              if (ss.GetData() && ss.GetSize())
                var_representation = ss.GetData();
              if (arg_idx > 0)
                s.PutCString(", ");
              if (var_value_sp->GetError().Success()) {
                if (var_representation)
                  s.Printf("%s=%s", var_name, var_representation);
                else
                  s.Printf("%s=%s at %s", var_name,
                           var_value_sp->GetTypeName().GetCString(),
                           var_value_sp->GetLocationAsCString());
              } else
                s.Printf("%s=<unavailable>", var_name);
            }

            if (close_paren)
              s.PutCString(close_paren);
            else
              s.PutChar(')');

          } else {
            s.PutCString(cstr);
          }
          return true;
        }
      }
    }
  }
  }

  return false;
}

void SwiftLanguage::GetExceptionResolverDescription(bool catch_on,
                                                    bool throw_on, Stream &s) {
  s.Printf("Swift Error breakpoint");
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString SwiftLanguage::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t SwiftLanguage::GetPluginVersion() { return 1; }

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
Language *SwiftLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeSwift:
    return new SwiftLanguage();
  default:
    return nullptr;
  }
}
