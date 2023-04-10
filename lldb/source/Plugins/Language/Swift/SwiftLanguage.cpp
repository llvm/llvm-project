//===-- SwiftLanguage.cpp ---------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftLanguage.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Utility/ConstString.h"

#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/LLDBLog.h"

#include "LogChannelSwift.h"
#include "ObjCRuntimeSyntheticProvider.h"
#include "SwiftFormatters.h"

#include <functional>
#include <mutex>

#include "swift/AST/ImportCache.h"
#include "swift/AST/Module.h"
#include "swift/AST/Type.h"
#include "swift/AST/Types.h"
#include "swift/Basic/InitializeSwiftModules.h"
#include "swift/Demangling/ManglingMacros.h"
#include "llvm/Support/ConvertUTF.h"

#include "Plugins/Language/ObjC/Cocoa.h"
#include "Plugins/Language/ObjC/NSString.h"

using namespace lldb;
using namespace lldb_private;

using lldb_private::formatters::AddCXXSummary;
using lldb_private::formatters::AddCXXSynthetic;
using lldb_private::formatters::AddFormat;
using lldb_private::formatters::AddStringSummary;
using lldb_private::formatters::AddSummary;
using lldb_private::formatters::swift::DictionaryConfig;
using lldb_private::formatters::swift::SetConfig;

LLDB_PLUGIN_DEFINE(SwiftLanguage)

void SwiftLanguage::Initialize() {
  LogChannelSwift::Initialize();
  static ConstString g_SwiftSharedStringClass("_TtCs21__SharedStringStorage");
  static ConstString g_SwiftStringStorageClass("_TtCs15__StringStorage");
  static ConstString g_NSArrayClass1("_TtCs22__SwiftDeferredNSArray");
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Swift Language",
                                CreateInstance);

  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .emplace(
          g_SwiftSharedStringClass,
          lldb_private::formatters::swift::SwiftSharedString_SummaryProvider);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .emplace(
          g_SwiftStringStorageClass,
          lldb_private::formatters::swift::SwiftStringStorage_SummaryProvider);
  lldb_private::formatters::NSArray_Additionals::GetAdditionalSummaries()
      .emplace(g_NSArrayClass1,
               lldb_private::formatters::swift::Array_SummaryProvider);
  lldb_private::formatters::NSArray_Additionals::GetAdditionalSynthetics()
      .emplace(g_NSArrayClass1,
               lldb_private::formatters::swift::ArraySyntheticFrontEndCreator);

  initializeSwiftModules();
}

void SwiftLanguage::Terminate() {
  // FIXME: Duplicating this list from Initialize seems error-prone.
  static ConstString g_SwiftSharedStringClass("_TtCs21__SharedStringStorage");
  static ConstString g_SwiftStringStorageClass("_TtCs15__StringStorage");
  static ConstString g_NSArrayClass1("_TtCs22__SwiftDeferredNSArray");

  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .erase(g_SwiftSharedStringClass);
  lldb_private::formatters::NSString_Additionals::GetAdditionalSummaries()
      .erase(g_SwiftStringStorageClass);

  lldb_private::formatters::NSArray_Additionals::GetAdditionalSummaries().erase(
      g_NSArrayClass1);
  lldb_private::formatters::NSArray_Additionals::GetAdditionalSynthetics()
      .erase(g_NSArrayClass1);

  PluginManager::UnregisterPlugin(CreateInstance);
}

bool SwiftLanguage::SymbolNameFitsToLanguage(Mangled mangled) const {
  return SwiftLanguageRuntime::IsSwiftMangledName(
      mangled.GetMangledName().GetStringRef());
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

std::vector<Language::MethodNameVariant>
SwiftLanguage::GetMethodNameVariants(ConstString method_name) const {
  std::vector<Language::MethodNameVariant> variant_names;

  // NOTE:  We need to do this because we don't have a proper parser for Swift
  // function name syntax so we try to ensure that if we autocomplete to
  // something, we'll look for its mangled equivalent and use the mangled
  // version as a lookup as well.

  ConstString counterpart;
  if (method_name.GetMangledCounterpart(counterpart))
    if (SwiftLanguageRuntime::IsSwiftMangledName(counterpart.GetStringRef()))
      variant_names.emplace_back(counterpart, eFunctionNameTypeFull);
  return variant_names;
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
  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Array_SummaryProvider,
      "Swift.ContiguousArray summary provider",
      ConstString("^Swift.ContiguousArray<.+>$"), summary_flags, true);
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
      ConstString("_TtCs22__SwiftDeferredNSArray"), summary_flags, false);
  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::swift::Array_SummaryProvider,
      "Swift.Array summary provider",
      ConstString("Swift.__SwiftDeferredNSArray"), summary_flags, false);
  AddCXXSummary(
      swift_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "Swift.Array summary provider",
      ConstString("_TtCs22__SwiftDeferredNSArray"), summary_flags, false);

  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::UnsafeTypeSummaryProvider,
      "Swift.Unsafe[Mutable][Raw][Buffer]Pointer",
      ConstString("^Swift.Unsafe(Mutable)?(Raw)?(Buffer)?Pointer(<.+>)?$"),
      summary_flags, true);

  DictionaryConfig::Get().RegisterSummaryProviders(swift_category_sp,
                                                   summary_flags);
  SetConfig::Get().RegisterSummaryProviders(swift_category_sp, summary_flags);

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
      "Swift.ContiguousArray synthetic children",
      ConstString("^Swift.ContiguousArray<.+>$"), synth_flags, true);
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
                  ConstString("_TtCs22__SwiftDeferredNSArray"), synth_flags,
                  false);
  AddCXXSynthetic(swift_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "Swift.Array synthetic children",
                  ConstString("_TtCs22__SwiftDeferredNSArray"), synth_flags,
                  false);
  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::ArraySyntheticFrontEndCreator,
      "Swift.Array synthetic children",
      ConstString("Swift.__SwiftDeferredNSArray"), synth_flags, false);

  AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::UnsafeTypeSyntheticFrontEndCreator,
      "Swift.Unsafe[Mutable][Raw][Buffer]Pointer",
      ConstString("^Swift.Unsafe(Mutable)?(Raw)?(Buffer)?Pointer(<.+>)?$"),
      synth_flags, true);

  DictionaryConfig::Get().RegisterSyntheticChildrenCreators(swift_category_sp,
                                                            synth_flags);
  SetConfig::Get().RegisterSyntheticChildrenCreators(swift_category_sp,
                                                     synth_flags);

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
                lldb_private::formatters::swift::UnicodeScalar_SummaryProvider,
                "Swift.Unicode.Scalar summary provider",
                ConstString("Swift.Unicode.Scalar"), summary_flags);
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
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::StringIndex_SummaryProvider,
                "Swift String.Index summary provider",
                ConstString("Swift.String.Index"), summary_flags);
  bool (*staticstring_summary_provider)(ValueObject &, Stream &,
                                        const TypeSummaryOptions &) =
      lldb_private::formatters::swift::StaticString_SummaryProvider;
  {
    TypeSummaryImpl::Flags substring_summary_flags = summary_flags;
    substring_summary_flags.SetDontShowChildren(false);
    AddCXXSummary(swift_category_sp,
                  lldb_private::formatters::swift::Substring_SummaryProvider,
                  "Swift.Substring summary provider",
                  ConstString("Swift.Substring"), substring_summary_flags);
  }
  AddCXXSummary(swift_category_sp, staticstring_summary_provider,
                "Swift.StaticString summary provider",
                ConstString("Swift.StaticString"), summary_flags);
  summary_flags.SetSkipPointers(false);
  // this is an ObjC dynamic type - as such it comes in pointer form
  // NSContiguousString* - do not skip pointers here
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::SwiftSharedString_SummaryProvider,
      "SharedStringStorage summary provider",
      ConstString("_TtCs21__SharedStringStorage"), summary_flags);
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
                "Swift.Range summary provider",
                ConstString("^Swift.Range<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::CountableRange_SummaryProvider,
                "Swift.CountableRange summary provider",
                ConstString("^Swift.CountableRange<.+>$"), summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::ClosedRange_SummaryProvider,
                "Swift.ClosedRange summary provider",
                ConstString("^Swift.ClosedRange<.+>$"), summary_flags, true);
  AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::CountableClosedRange_SummaryProvider,
      "Swift.CountableClosedRange summary provider",
      ConstString("^Swift.CountableClosedRange<.+>$"), summary_flags, true);

  TypeSummaryImpl::Flags simd_summary_flags;
  simd_summary_flags.SetCascades(true)
      .SetDontShowChildren(true)
      .SetHideItemNames(true)
      .SetShowMembersOneLiner(false);
  const char *SIMDTypes = "^(simd\\.)?(simd_)?("
                          "(int|uint|float|double)[234]|"
                          "(float|double)[234]x[234]|"
                          "quat(f|d)"
                          ")$";

  const char *newSIMDTypes = "^SIMD[0-9]+<.*>$";

  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::LegacySIMD_SummaryProvider,
                "SIMD (legacy) summary provider", ConstString(SIMDTypes),
                simd_summary_flags, true);
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::SIMDVector_SummaryProvider,
                "Vector SIMD summary provider", ConstString(newSIMDTypes),
                simd_summary_flags, true);

  const char *GLKitTypes = "^(GLKMatrix[234]|GLKVector[234]|GLKQuaternion)$";
  AddCXXSummary(swift_category_sp,
                lldb_private::formatters::swift::GLKit_SummaryProvider,
                "GLKit summary provider", ConstString(GLKitTypes),
                simd_summary_flags, true);

  AddStringSummary(swift_category_sp, "${var.native}",
                   ConstString("CoreGraphics.CGFloat"), summary_flags);
  AddStringSummary(swift_category_sp, "${var.native}",
                   ConstString("Foundation.CGFloat"), summary_flags);
  AddStringSummary(swift_category_sp, "${var.native}",
                   ConstString("CoreFoundation.CGFloat"), summary_flags);
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

  lldb_private::formatters::AddCXXSummary(
      swift_category_sp,
      lldb_private::formatters::swift::Decimal_SummaryProvider,
      "Decimal summary provider", ConstString("Foundation.Decimal"),
      TypeSummaryImpl::Flags(summary_flags).SetDontShowChildren(true));

  lldb_private::formatters::AddCXXSynthetic(
      swift_category_sp,
      lldb_private::formatters::swift::URLComponentsSyntheticFrontEndCreator,
      "URLComponents synthetic children",
      ConstString("Foundation.URLComponents"),
      SyntheticChildren::Flags()
          .SetSkipPointers(true)
          .SetCascades(true)
          .SetSkipReferences(false)
          .SetNonCacheable(false));
}

lldb::TypeCategoryImplSP SwiftLanguage::GetFormatters() {
  static std::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  std::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               g_category);
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
      llvm::StringRef tau_ = u8"$\u03C4_";
      if (valobj.GetName().GetLength() > 12 &&
          valobj.GetName().GetStringRef().startswith(tau_) &&
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

static CompilerType
ExtractSwiftTypeFromCxxInteropType(CompilerType type, TypeSystemSwift &ts,
                                   SwiftLanguageRuntime &runtime) {
  // Try to recognize a Swift type wrapped in a C++ interop wrapper class.
  // These types have a typedef from a char to the swift mangled name, and a
  // static constexpr char field whose type is the typedef, and whose name
  // is __swift_mangled_name.
  // These classes will look something like:
  // class CxxBridgedClass {
  //   [Layout specific variables]
  //   typedef char $sClassMangledNameHere;
  //   static inline constexpr $sClassMangledNameHere __swift_mangled_name = 0;
  // }

  Log *log(GetLog(LLDBLog::DataFormatters));
  // This only makes sense for Clang types.
  auto tsc = type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>();
  if (!tsc)
    return {};

  // We operate directly on the qualified type because the TypeSystem
  // interface doesn't allow us to check for static constexpr members.
  auto qual_type = TypeSystemClang::GetQualType(type.GetOpaqueQualType());
  auto *record_type =
      llvm::dyn_cast_or_null<clang::RecordType>(qual_type.getTypePtrOrNull());
  if (!record_type) {
    LLDB_LOGV(log, "[ExtractSwiftTypeFromCxxInteropType] "
                   "Type is not a record type.");
    return {};
  }

  const clang::RecordDecl *record_decl = record_type->getDecl();
  CompilerType swift_type;

  for (auto *child_decl : record_decl->decls()) {
    auto *var_decl = llvm::dyn_cast<clang::VarDecl>(child_decl);
    if (!var_decl)
      continue;

    auto name = var_decl->getName();
    if (name != "__swift_mangled_name")
      continue;

    const auto *typedef_type =
        llvm::dyn_cast<clang::TypedefType>(var_decl->getType());
    if (!typedef_type)
      break;

    auto *decl = typedef_type->getDecl();
    if (!decl)
      break;

    auto swift_name = decl->getName();
    if (!swift::Demangle::isMangledName(swift_name))
      break;

    swift_type = ts.GetTypeFromMangledTypename(ConstString(swift_name));
    break;
  }

  if (swift_type) {
    auto bound_type = runtime.BindGenericTypeParameters(
        swift_type, [&](unsigned depth, unsigned index) -> CompilerType {
          assert(depth == 0 && "Unexpected depth! C++ interop does not support "
                               "nested generic parameters");
          if (depth > 0)
            return {};

          auto templated_type = type.GetTypeTemplateArgument(index);
          auto substituted_type =
              ExtractSwiftTypeFromCxxInteropType(templated_type, ts, runtime);

          // The generic type might also not be a user defined type which
          // ExtractSwiftTypeFromCxxInteropType can find, but which is still
          // convertible to Swift (for example, int -> Int32). Attempt to
          // convert it to a Swift type.
          if (!substituted_type)
            substituted_type = ts.ConvertClangTypeToSwiftType(templated_type);
          return substituted_type;
        });
    return bound_type;
  }
  return {};
}

/// Synthetic child that wraps a value object.
class ValueObjectWrapperSyntheticChildren : public SyntheticChildren {
  class ValueObjectWrapperFrontEndProvider : public SyntheticChildrenFrontEnd {
  public:
    ValueObjectWrapperFrontEndProvider(ValueObject &backend)
        : SyntheticChildrenFrontEnd(backend) {}

    size_t CalculateNumChildren() override {
      return 1;
    }

    lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
      return idx == 0 ? m_backend.GetSP() : nullptr;
    }

    size_t GetIndexOfChildWithName(ConstString name) override {
      return m_backend.GetName() == name ? 0 : UINT32_MAX;
    }

    bool Update() override { return false; }

    bool MightHaveChildren() override { return true; }

    ConstString GetSyntheticTypeName() override {
      return m_backend.GetCompilerType().GetTypeName();
    }
  };

public:
  ValueObjectWrapperSyntheticChildren(ValueObjectSP valobj, const Flags &flags) 
      : SyntheticChildren(flags), m_valobj(valobj) {}

  SyntheticChildrenFrontEnd::AutoPointer
  GetFrontEnd(ValueObject &backend) override {
    if (!m_valobj)
      return nullptr;
    // We ignore the backend parameter here, as we have a more specific one
    // available.
    return std::make_unique<ValueObjectWrapperFrontEndProvider>(*m_valobj); 
  }

  bool IsScripted() override { return false; }

  std::string GetDescription() override {
    return "C++ bridged synthetic children";
  }

private:
  ValueObjectSP m_valobj;
};

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
        // FIXME: The classification of clang-imported enums may
        // change based on whether a Swift module is present or not.
        if (!valobj.GetValueAsCString())
          return nullptr;
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
                auto swift_ast_ctx =
                    type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
                if (swift_ast_ctx && swift_ast_ctx->IsImportedType(
                                         type.GetOpaqueQualType(), nullptr))
                  is_imported = true;
              }

              ExecutionContext exe_ctx(valobj.GetExecutionContextRef());
              if (is_imported && type.GetNumFields(&exe_ctx) == 0)
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
            // If this is a Swift tagged pointer, the Objective-C data
            // formatters may incorrectly classify it as an
            // Objective-C tagged pointer.
            AddressType address_type;
            lldb::addr_t ptr = valobj.GetPointerValue(&address_type);
            auto *swift_runtime = SwiftLanguageRuntime::Get(process_sp);
            if (!swift_runtime)
              return nullptr;
            if (swift_runtime->IsTaggedPointer(ptr, valobj.GetCompilerType()))
              return nullptr;
            ObjCLanguageRuntime *objc_runtime =
                ObjCLanguageRuntime::Get(*process_sp);
            if (!objc_runtime)
              return nullptr;
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
              if ((TypeSystemClang::IsObjCObjectPointerType(type) ||
                   TypeSystemClang::IsObjCObjectOrInterfaceType(type)) &&
                  SwiftLanguageRuntime::IsSwiftClassName(
                      type.GetTypeName().GetCString()))
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
            auto *swift_runtime = SwiftLanguageRuntime::Get(process_sp);
            return swift_runtime->GetBridgedSyntheticChildProvider(valobj);
          }
          return nullptr;
        });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType dyn_type,
                              FormatManager &format_manager)
                               -> lldb::SyntheticChildrenSP {
      Log *log(GetLog(LLDBLog::DataFormatters));

      ProcessSP process_sp(valobj.GetProcessSP());
      auto *swift_runtime = SwiftLanguageRuntime::Get(process_sp);
      if (!swift_runtime) {
        LLDB_LOGV(log, "[Matching CxxBridgedSyntheticChildProvider] - "
                       "Could not get the swift runtime.");
        return nullptr;
      }

      auto scratch_ctx = valobj.GetSwiftScratchContext();
      if (!scratch_ctx) {
        LLDB_LOGV(log, "[Matching CxxBridgedSyntheticChildProvider] - "
                       "Could not get the swift scratch context.");
        return nullptr;
      }
      auto &type_system_swift = **scratch_ctx;

      auto type = valobj.GetCompilerType();

      auto swift_type = ExtractSwiftTypeFromCxxInteropType(
          type, type_system_swift, *swift_runtime);
      if (!swift_type) {
        LLDB_LOGV(log, "[Matching CxxBridgedSyntheticChildProvider] - "
                       "Did not find Swift type.");
        return nullptr;
      }

      auto swift_valobj =
          SwiftLanguageRuntime::ExtractSwiftValueObjectFromCxxWrapper(valobj);
      if (!swift_valobj) {
        StreamString clang_desc;
        type.DumpTypeDescription(&clang_desc);

        StreamString swift_desc;
        type.DumpTypeDescription(&swift_desc);

        LLDB_LOGF(log,
                  "[Matching CxxBridgedSyntheticChildProvider] - "
                  "Was not able to extract Swift value object. Clang type: %s. "
                  "Swift type: %s",
                  clang_desc.GetData(), swift_desc.GetData());
        return nullptr;
      }
      // Cast it to a Swift type since thhe swift runtime expects a Swift value
      // object.
      auto casted_to_swift = swift_valobj->Cast(swift_type);
      if (!casted_to_swift) {
        LLDB_LOGF(log, "[Matching CxxBridgedSyntheticChildProvider] - "
                       "Could not cast value object to swift type.");
        return nullptr;
      }

      TypeAndOrName type_or_name;
      Address address;
      Value::ValueType value_type;
      // Try to find the dynamic type of the Swift type.
      // TODO: find a way to get the dyamic value type from the
      // command.
      if (swift_runtime->GetDynamicTypeAndAddress(
              *casted_to_swift.get(),
              lldb::DynamicValueType::eDynamicCanRunTarget, type_or_name,
              address, value_type)) {
        if (type_or_name.HasCompilerType()) {
          swift_type = type_or_name.GetCompilerType();
          // Cast it to the more specific type.
          casted_to_swift = casted_to_swift->Cast(swift_type);
          if (!casted_to_swift) {
            LLDB_LOGF(log,
                      "[Matching CxxBridgedSyntheticChildProvider] - "
                      "Could not cast value object to dynamic swift type.");
            return nullptr;
          }
        }
      }

      casted_to_swift->SetName(ConstString("Swift_Type"));

      SyntheticChildrenSP synth_sp =
          SyntheticChildrenSP(new ValueObjectWrapperSyntheticChildren(
              casted_to_swift, SyntheticChildren::Flags()));
      return synth_sp;
    });
    g_formatters.push_back([](lldb_private::ValueObject &valobj,
                              lldb::DynamicValueType,
                              FormatManager &) -> lldb::SyntheticChildrenSP {
      // If C++ interop is enabled, format imported clang types as clang types,
      // instead of attempting to disguise them as Swift types.

      Log *log(GetLog(LLDBLog::DataFormatters));

      if (!valobj.GetTargetSP()->IsSwiftCxxInteropEnabled())
        return nullptr;

      CompilerType type(valobj.GetCompilerType());
      auto swift_type_system =
          type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
      if (!swift_type_system)
        return nullptr;

      CompilerType original_type;
      if (!swift_type_system->IsImportedType(type.GetOpaqueQualType(),
                                        &original_type))
        return nullptr;

      if (!original_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>())
        return nullptr;

      auto qual_type =
          TypeSystemClang::GetQualType(original_type.GetOpaqueQualType());
      auto *decl = qual_type->getAsCXXRecordDecl();
      if (!decl) {
        LLDB_LOGV(log, "[Matching Clang imported type] - "
                       "Could not get decl from clang type");
        return nullptr;
      }
      auto casted = valobj.Cast(original_type);
      if (!casted) {
        LLDB_LOGV(log, "[Matching Clang imported type] - "
                       "Could not cast value object to clang type");
        return nullptr;
      }

      casted->SetName(ConstString("Clang_Type"));

      SyntheticChildrenSP synth_sp =
          SyntheticChildrenSP(new ValueObjectWrapperSyntheticChildren(
              casted, SyntheticChildren::Flags()));
      return synth_sp;
    });
  });

  return g_formatters;
}

bool SwiftLanguage::IsSourceFile(llvm::StringRef file_path) const {
  return file_path.endswith(".swift");
}

std::vector<ConstString> SwiftLanguage::GetPossibleFormattersMatches(
    ValueObject &valobj, lldb::DynamicValueType use_dynamic) {
  std::vector<ConstString> result;

  if (use_dynamic == lldb::eNoDynamicValues)
    return result;

  // There is no point in attempting to format Clang types here, since
  // FormatManager will try to format all Swift types also as
  // Objective-C types and vice versa.  Due to the incomplete
  // ClangImporter implementation for C++, continuing here for
  // Objective-C++ types can actually lead to crashes that can be
  // avoided by just formatting those types as Objective-C types.
  if (valobj.GetObjectRuntimeLanguage() == eLanguageTypeObjC)
    return result;

  SwiftScratchContextLock scratch_ctx_lock(&valobj.GetExecutionContextRef());
  CompilerType compiler_type(valobj.GetCompilerType());

  const bool check_cpp = false;
  const bool check_objc = false;
  bool canBeSwiftDynamic =
      compiler_type.IsPossibleDynamicType(nullptr, check_cpp, check_objc);

  if (!canBeSwiftDynamic)
    return result;
  lldb::ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return result;
  auto *runtime = SwiftLanguageRuntime::Get(process_sp);
  if (!runtime)
    return result;
  TypeAndOrName type_and_or_name;
  Address address;
  Value::ValueType value_type;
  if (!runtime->GetDynamicTypeAndAddress(valobj, use_dynamic, type_and_or_name,
                                         address, value_type))
    return result;
  if (ConstString name = type_and_or_name.GetName())
    result.push_back(name);
  return result;
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
    friend std::unique_ptr<Language::TypeScavenger>
    SwiftLanguage::GetTypeScavenger();

  private:
    typedef SwiftASTContext::TypeOrDecl TypeOrDecl;
    typedef SwiftASTContext::TypesOrDecls TypesOrDecls;

    class SwiftScavengerResult : public Language::TypeScavenger::Result {
    public:
      typedef SwiftASTContext::TypeOrDecl TypeOrDecl;

      SwiftScavengerResult(TypeOrDecl type)
          : Language::TypeScavenger::Result(), m_result(type) {}

      bool IsValid() override { return m_result.operator bool(); }

      bool DumpToStream(Stream &stream, bool print_help_if_available,
                        ExecutionContextScope *exe_scope = nullptr) override {
        if (IsValid()) {
          auto as_type = m_result.GetAs<CompilerType>();
          auto as_decl = m_result.GetAs<swift::Decl *>();

          if (as_type.hasValue() && as_type.getValue()) {
            if (auto swift_ast_ctx = as_type->GetTypeSystem()
                                         .dyn_cast_or_null<TypeSystemSwift>())
              swift_ast_ctx->DumpTypeDescription(
                  as_type->GetOpaqueQualType(), &stream,
                  print_help_if_available, true, eDescriptionLevelFull,
                  exe_scope);
            else
              as_type->DumpTypeDescription(
                  &stream, eDescriptionLevelFull,
                  exe_scope); // we should always have a swift type here..
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
              Status error;
              llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
                  target->GetSwiftScratchContext(error, *exe_scope,
                                                 create_on_demand);
              if (maybe_scratch_ctx)
                if (auto scratch_ctx = maybe_scratch_ctx->get())
                  if (SwiftASTContext *ast_ctx =
                          scratch_ctx->GetSwiftASTContext()) {
                    ConstString cs_input{input};
                    Mangled mangled(cs_input);
                    if (mangled.GuessLanguage() == eLanguageTypeSwift) {
                      auto candidate =
                          ast_ctx->GetTypeFromMangledTypename(cs_input);
                      if (candidate.IsValid())
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
                    result_type = TypeSystemSwift::GetInstanceType(result_type);
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
            Status error;
            llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
                target->GetSwiftScratchContext(error, *exe_scope,
                                               create_on_demand);
              if (maybe_scratch_ctx)
                if (auto scratch_ctx = maybe_scratch_ctx->get())
                  if (SwiftASTContext *ast_ctx =
                          scratch_ctx->GetSwiftASTContext()) {
                    auto iter = ast_ctx->GetModuleCache().begin(),
                         end = ast_ctx->GetModuleCache().end();

                    std::vector<llvm::StringRef> name_parts;
                    SplitDottedName(input, name_parts);

                    std::function<void(swift::ModuleDecl *)> lookup_func =
                        [&ast_ctx, input, name_parts,
                         &results](swift::ModuleDecl *module) -> void {
                      for (auto imported_module :
                           swift::namelookup::getAllImports(module)) {
                        auto module = imported_module.importedModule;
                        TypesOrDecls local_results;
                        ast_ctx->FindTypesOrDecls(input, module, local_results,
                                                  false);
                        llvm::Optional<TypeOrDecl> candidate;
                        if (local_results.empty() && name_parts.size() > 1) {
                          size_t idx_of_deeper = 1;
                          // if you're looking for Swift.Int in module Swift,
                          // try looking for Int
                          if (name_parts.front() == module->getName().str()) {
                            candidate = ast_ctx->FindTypeOrDecl(
                                name_parts[1].str().c_str(), module);
                            idx_of_deeper = 2;
                          }
                          // this is probably the top-level name of a nested
                          // type String.UTF8View
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
                                   name_parts.front() ==
                                       module->getName().str())
                          results.insert(
                              ToCompilerType(swift::ModuleType::get(module)));
                      }
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
         "- a mangled type name (e.g. $sSiD)\n"
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
  static ConstString g_NSNumberInt128("NSNumber:int128_t");
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
  if (type_hint == g_NSNumberInt128) {
    prefix = "Int128(";
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

LazyBool SwiftLanguage::IsLogicalTrue(ValueObject &valobj, Status &error) {
  static ConstString g_SwiftBool("Swift.Bool");
  static ConstString g_value("_value");

  Scalar scalar_value;

  auto swift_ty = GetCanonicalSwiftType(valobj.GetCompilerType());
  CompilerType valobj_type = ToCompilerType(swift_ty);
  Flags type_flags(valobj_type.GetTypeInfo());
  if (valobj_type.GetTypeSystem().isa_and_nonnull<TypeSystemSwift>()) {
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
  SwiftScratchContextLock scratch_ctx_lock(exe_ctx);
  switch (representation) {
  case Language::FunctionNameRepresentation::eName:
    break; // no need to customize this
  case Language::FunctionNameRepresentation::eNameWithNoArgs: {
    if (sc->function) {
      if (sc->function->GetLanguage() == eLanguageTypeSwift) {
        if (ConstString cs = sc->function->GetDisplayName(sc)) {
          s.Printf("%s", cs.AsCString());
          return true;
        }
      }
    }
    break;
  }
  case Language::FunctionNameRepresentation::eNameWithArgs: {
    if (sc->function) {
      if (sc->function->GetLanguage() == eLanguageTypeSwift) {
        if (const char *cstr = sc->function->GetDisplayName(sc).AsCString()) {
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
            cstr = inline_info->GetName().GetCString();
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
                      ss,
                      ValueObject::ValueObjectRepresentationStyle::
                          eValueObjectRepresentationStyleSummary,
                      eFormatDefault,
                      ValueObject::PrintableRepresentationSpecialCases::eAllow,
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

ConstString
SwiftLanguage::GetDemangledFunctionNameWithoutArguments(Mangled mangled) const {
  ConstString mangled_name = mangled.GetMangledName();
  ConstString demangled_name = mangled.GetDemangledName();
  if (demangled_name && mangled_name) {
    if (SwiftLanguageRuntime::IsSwiftMangledName(
            demangled_name.GetStringRef())) {
      lldb_private::ConstString basename;
      bool is_method = false;
      if (SwiftLanguageRuntime::MethodName::ExtractFunctionBasenameFromMangled(
              mangled_name, basename, is_method)) {
        if (basename && basename != mangled_name)
          return basename;
      }
    }
  }
  if (demangled_name)
    return demangled_name;
  return mangled_name;
}

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
