//===-- SwiftOptional.cpp ---------------------------------------*- C++ -*-===//
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

#include "SwiftOptional.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/ValueObjectPrinter.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/ValueObject/ValueObjectCast.h"
#include "lldb/ValueObject/ValueObjectMemory.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

std::string lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    GetDescription() {
  StreamString sstr;
  sstr.Printf("`%s `%s%s%s%s%s%s%s", GetName().c_str(),
              Cascades() ? "" : " (not cascading)", " (may show children)",
              !DoesPrintValue(nullptr) ? " (hide value)" : "",
              IsOneLiner() ? " (one-line printout)" : "",
              SkipsPointers() ? " (skip pointers)" : "",
              SkipsReferences() ? " (skip references)" : "",
              HideNames(nullptr) ? " (hide member names)" : "");
  return sstr.GetString().str();
}

std::string
lldb_private::formatters::swift::SwiftOptionalSummaryProvider::GetName() {
  return "Swift.Optional summary provider";
}

/// If this ValueObject is an Optional<T> with the Some(T) case selected,
/// retrieve the value of the Some case.
///
/// Returns {} on error, nullptr on .none, and a ValueObject on .some.
/// None of the callees can pass on errors messages, so this function
/// doesn't return them either.
static std::optional<ValueObjectSP>
ExtractSomeIfAny(ValueObject *optional,
                 bool synthetic_value = false) {
  if (!optional)
    return {};

  static ConstString g_some("some");
  static ConstString g_none("none");

  ValueObjectSP non_synth_valobj = optional->GetNonSyntheticValue();
  if (!non_synth_valobj)
    return {};

  if (!(non_synth_valobj->GetCompilerType()
            .GetTypeSystem()
            .dyn_cast_or_null<TypeSystemSwift>()
            .get()))
    return {};

  auto process_sp = optional->GetProcessSP();
  if (!process_sp)
    return {};
  auto *runtime = SwiftLanguageRuntime::Get(process_sp);
  if (!runtime)
    return {};

  ValueObjectSP value_sp;
  auto project_instance_ptr =
      [&](ValueObject &valobj, CompilerType projected_type,
          TypeSystemSwift::NonTriviallyManagedReferenceKind kind)
      -> llvm::Expected<ValueObjectSP> {
    // ObjC reference?
    if (kind == TypeSystemSwift::NonTriviallyManagedReferenceKind::eWeak &&
        runtime->IsObjCInstance(valobj))
      return ValueObjectCast::Create(
          valobj, valobj.GetPointerValue() ? g_some : g_none, projected_type);

    // Unowned/strong reference.
    if (kind != TypeSystemSwift::NonTriviallyManagedReferenceKind::eWeak)
      return ValueObjectCast::Create(
          valobj, valobj.GetPointerValue() ? g_some : g_none, projected_type);

    Status error;
    lldb::addr_t ptr =
        runtime->FixupAddress(valobj.GetPointerValue(), projected_type, error);
    // Needed for resilience.
    ptr = runtime->MaskMaybeBridgedPointer(ptr);
    auto exe_ctx = valobj.GetExecutionContextRef().Lock(true);
    return ValueObjectMemory::Create(exe_ctx.GetBestExecutionContextScope(),
                                     g_some, ptr, projected_type);
  };

  auto project_enum =
      [&](ValueObject &valobj) -> llvm::Expected<ValueObjectSP> {
    CompilerType type = valobj.GetCompilerType();
    auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!ts)
      return llvm::createStringError("not a Swift type");
    if (auto kind =
            ts->GetNonTriviallyManagedReferenceKind(type.GetOpaqueQualType()))
      return project_instance_ptr(
          valobj, ts->GetWeakReferent(type.GetOpaqueQualType()), *kind);

    return runtime->SwiftLanguageRuntime::ProjectEnum(valobj);
  };

  // ObjC pointer.
  auto static_valobj = non_synth_valobj->GetStaticValue();
  if (static_valobj && !(static_valobj->GetCompilerType()
                             .GetTypeSystem()
                             .dyn_cast_or_null<TypeSystemSwift>()
                             .get())) {
    value_sp = static_valobj;
    ValueObjectSP dyn_value_sp =
        value_sp->GetDynamicValue(eDynamicDontRunTarget);
    if (dyn_value_sp) {
      // GetDynamicValue may map the an ObjC pointer back into Swift
      // as an Optional type. Unwrap the Optional.
      value_sp = dyn_value_sp;
      CompilerType dyn_type = value_sp->GetCompilerType();
      auto ts =
          dyn_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
      if (!ts)
        return value_sp;
      if (ts->IsOptionalType(dyn_type.GetOpaqueQualType()))
        return ValueObjectCast::Create(
            *value_sp, g_some,
            TypeSystemSwiftTypeRef::GetOptionalType(dyn_type));
    }
    return value_sp;
  }

  llvm::Expected<ValueObjectSP> projected = project_enum(*non_synth_valobj);
  if (!projected) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), projected.takeError(),
                   "{0}");
    // Some Optionals (TestExternalProviderExtraInhabitants) cannot be
    // projected. They worked by accident in the old implementation,
    // this hack makes the test pass, but it is not correct.
    CompilerType projected_type =
        TypeSystemSwiftTypeRef::GetOptionalType(optional->GetCompilerType());
    if (!projected_type)
      return {};
    return ValueObjectCast::Create(
        *optional, optional->GetPointerValue() ? g_some : g_none,
        projected_type);
  }
  if (!*projected)
    return nullptr;

  ConstString value = (*projected)->GetName();

  if (!value)
    return {};

  if (value != g_some)
    return {};

  value_sp = *projected;
  if (!value_sp)
    return {};

  lldb::DynamicValueType use_dynamic;

  // FIXME: We usually want to display the dynamic value of an optional's
  // payload, but we don't have a simple way to determine whether the dynamic
  // value was actually requested. Consult the target setting as a workaround.
  if (runtime->CouldHaveDynamicValue(*value_sp))
    // FIXME (cont): Here, we'd like to use some new API to determine whether
    // a dynamic value was actually requested.
    use_dynamic = eDynamicDontRunTarget;
  else
    use_dynamic = value_sp->GetTargetSP()->GetPreferDynamicValue();

  ValueObjectSP dyn_value_sp = value_sp->GetDynamicValue(use_dynamic);
  if (dyn_value_sp)
    value_sp = dyn_value_sp;

  return value_sp;
}

static bool
SwiftOptional_SummaryProvider_Impl(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options) {
  std::optional<ValueObjectSP> maybe_some = ExtractSomeIfAny(&valobj, true);
  if (!maybe_some)
    return false;

  ValueObjectSP some = *maybe_some;
  if (!some) {
    stream.Printf("nil");
    return true;
  }

  if (some->HasSyntheticValue())
    some = some->GetSyntheticValue();

  const char *summary = some->GetSummaryAsCString(lldb::eLanguageTypeSwift);
  const char *value = some->GetValueAsCString();

  if (summary)
    stream.Printf("%s", summary);
  else if (value)
    stream.Printf("%s", value);
  else if (lldb_private::DataVisualization::ShouldPrintAsOneLiner(*some)) {
    TypeSummaryImpl::Flags oneliner_flags;
    oneliner_flags.SetHideItemNames(false)
        .SetCascades(true)
        .SetDontShowChildren(false)
        .SetDontShowValue(false)
        .SetShowMembersOneLiner(true)
        .SetSkipPointers(false)
        .SetSkipReferences(false);
    StringSummaryFormat oneliner(oneliner_flags, "");
    std::string buffer;
    oneliner.FormatObject(some.get(), buffer, options);
    stream.Printf("%s", buffer.c_str());
  }

  return true;
}

bool lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    FormatObject(ValueObject *target_valobj_sp, std::string &dest,
                 const TypeSummaryOptions &options) {
  if (!target_valobj_sp)
    return false;

  StreamString stream;

  bool is_ok =
      SwiftOptional_SummaryProvider_Impl(*target_valobj_sp, stream, options);
  dest.assign(stream.GetString().str());

  return is_ok;
}

bool lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    DoesPrintChildren(ValueObject *target_valobj) const {
  if (!target_valobj)
    return false;

  std::optional<ValueObjectSP> maybe_some =
      ExtractSomeIfAny(target_valobj, true);
  if (!maybe_some)
    return false;

  ValueObjectSP some = *maybe_some;
  if (!some)
    return true;

  lldb_private::Flags some_flags(some->GetCompilerType().GetTypeInfo());

  if (some_flags.AllSet(eTypeIsSwift)) {
    if (some_flags.AnySet(eTypeInstanceIsPointer | eTypeIsProtocol |
                          lldb::eTypeIsEnumeration))
      return true;
  }

  lldb::TypeSummaryImplSP summary_sp = some->GetSummaryFormat();

  if (!summary_sp) {
    if (lldb_private::DataVisualization::ShouldPrintAsOneLiner(*some))
      return false;
    return some->HasChildren();
  }
  return some->HasChildren() && summary_sp->DoesPrintChildren(some.get());
}

bool lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    DoesPrintValue(ValueObject *valobj) const {
  return false;
}

lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    SwiftOptionalSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
  Update();
}

bool lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::IsEmpty()
    const {
  return (m_is_none == true || m_some == nullptr);
}

llvm::Expected<uint32_t> lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::CalculateNumChildren() {
  if (IsEmpty())
    return 0;
  if (m_some->HasSyntheticValue())
    return m_some->GetSyntheticValue()->GetNumChildren();
  return m_some->GetNumChildren();
}

lldb::ValueObjectSP lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (IsEmpty())
    return nullptr;

  ValueObjectSP some = m_some;
  if (some->HasSyntheticValue())
    some = some->GetSyntheticValue();

  auto child = some->GetChildAtIndex(idx, true);
  if (child && some->IsSyntheticChildrenGenerated())
    child->SetSyntheticChildrenGenerated(true);
  return child;
}

lldb::ChildCacheState
lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::Update() {
  m_some = nullptr;
  m_is_none = true;

  std::optional<ValueObjectSP> maybe_some = ExtractSomeIfAny(&m_backend, false);
  if (!maybe_some)
    return ChildCacheState::eRefetch;

  m_some = *maybe_some;

  if (!m_some) {
    m_is_none = true;
    return ChildCacheState::eRefetch;
  }

  m_is_none = false;

  return ChildCacheState::eRefetch;
}

bool lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    MightHaveChildren() {
  return IsEmpty() ? false : true;
}

llvm::Expected<size_t> lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (IsEmpty())
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());

  return m_some->GetIndexOfChildWithName(name);
}

lldb::ValueObjectSP lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::GetSyntheticValue() {
  return m_some;
  if (m_some && m_some->CanProvideValue()) {
    if (m_some->HasSyntheticValue())
      return m_some->GetSyntheticValue();
    return m_some;
  }
  return nullptr;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return (new SwiftOptionalSyntheticFrontEnd(valobj_sp));
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::SwiftUncheckedOptionalSyntheticFrontEndCreator(
    CXXSyntheticChildren *cxx_synth, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return SwiftOptionalSyntheticFrontEndCreator(cxx_synth, valobj_sp);
}
