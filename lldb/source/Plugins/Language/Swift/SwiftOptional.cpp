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
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/ValueObjectPrinter.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

std::string lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    GetDescription() {
  StreamString sstr;
  sstr.Printf("`%s `%s%s%s%s%s%s%s", "Swift.Optional summary provider",
              Cascades() ? "" : " (not cascading)", " (may show children)",
              !DoesPrintValue(nullptr) ? " (hide value)" : "",
              IsOneLiner() ? " (one-line printout)" : "",
              SkipsPointers() ? " (skip pointers)" : "",
              SkipsReferences() ? " (skip references)" : "",
              HideNames(nullptr) ? " (hide member names)" : "");
  return sstr.GetString();
}

// if this ValueObject is an Optional<T> with the Some(T) case selected,
// retrieve the value of the Some case..
static PointerOrSP
ExtractSomeIfAny(ValueObject *optional,
                 bool synthetic_value = false) {
  if (!optional)
    return nullptr;

  static ConstString g_Some("some");
  static ConstString g_None("none");

  ValueObjectSP non_synth_valobj = optional->GetNonSyntheticValue();
  if (!non_synth_valobj)
    return nullptr;

  ConstString value(non_synth_valobj->GetValueAsCString());

  if (!value || value == g_None)
    return nullptr;

  PointerOrSP value_sp(
      non_synth_valobj->GetChildMemberWithName(g_Some, true).get());
  if (!value_sp)
    return nullptr;

  auto process_sp = optional->GetProcessSP();
  SwiftLanguageRuntime *swift_runtime =
      process_sp ? process_sp->GetSwiftLanguageRuntime() : nullptr;

  SwiftASTContext::NonTriviallyManagedReferenceStrategy strategy;
  if (SwiftASTContext::IsNonTriviallyManagedReferenceType(
          non_synth_valobj->GetCompilerType(), strategy) &&
      strategy ==
          SwiftASTContext::NonTriviallyManagedReferenceStrategy::eWeak) {
    if (swift_runtime) {
      lldb::addr_t original_ptr =
          value_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
      lldb::addr_t tweaked_ptr =
          swift_runtime->MaybeMaskNonTrivialReferencePointer(original_ptr,
                                                             strategy);
      if (original_ptr != tweaked_ptr) {
        CompilerType value_type(value_sp->GetCompilerType());
        DataBufferSP buffer_sp(
            new DataBufferHeap(&tweaked_ptr, sizeof(tweaked_ptr)));
        DataExtractor extractor(buffer_sp, process_sp->GetByteOrder(),
                                process_sp->GetAddressByteSize());
        ExecutionContext exe_ctx(process_sp);
        value_sp = PointerOrSP(ValueObject::CreateValueObjectFromData(
            value_sp->GetName().AsCString(), extractor, exe_ctx, value_type));
        if (!value_sp)
          return nullptr;
        else
          value_sp->SetSyntheticChildrenGenerated(true);
      }
    }
  }

  lldb::DynamicValueType use_dynamic;

  // FIXME: We usually want to display the dynamic value of an optional's
  // payload, but we don't have a simple way to determine whether the dynamic
  // value was actually requested. Consult the target setting as a workaround.
  if (swift_runtime->CouldHaveDynamicValue(*value_sp))
    // FIXME (cont): Here, we'd like to use some new API to determine whether
    // a dynamic value was actually requested.
    use_dynamic = eDynamicDontRunTarget;
  else
    use_dynamic = value_sp->GetTargetSP()->GetPreferDynamicValue();

  ValueObjectSP dyn_value_sp = value_sp->GetDynamicValue(use_dynamic);
  if (dyn_value_sp)
    value_sp = dyn_value_sp;

  if (synthetic_value && value_sp->HasSyntheticValue())
    value_sp = value_sp->GetSyntheticValue();

  return value_sp;
}

static bool
SwiftOptional_SummaryProvider_Impl(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options) {
  PointerOrSP some = ExtractSomeIfAny(&valobj, true);
  if (!some) {
    stream.Printf("nil");
    return true;
  }

  const char *value_summary = some->GetSummaryAsCString();

  if (value_summary)
    stream.Printf("%s", value_summary);
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
    oneliner.FormatObject(some, buffer, options);
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
  dest.assign(stream.GetString());

  return is_ok;
}

bool lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    DoesPrintChildren(ValueObject *target_valobj) const {
  if (!target_valobj)
    return false;

  PointerOrSP some = ExtractSomeIfAny(target_valobj, true);

  if (!some)
    return true;

  lldb_private::Flags some_flags(some->GetCompilerType().GetTypeInfo());

  if (some_flags.AllSet(eTypeIsSwift)) {
    if (some_flags.AnySet(eTypeInstanceIsPointer | eTypeIsProtocol))
      return true;
  }

  lldb::TypeSummaryImplSP summary_sp = some->GetSummaryFormat();

  if (!summary_sp) {
    if (lldb_private::DataVisualization::ShouldPrintAsOneLiner(*some))
      return false;
    else
      return (some->GetNumChildren() > 0);
  } else
    return (some->GetNumChildren() > 0) &&
           (summary_sp->DoesPrintChildren(some));
}

bool lldb_private::formatters::swift::SwiftOptionalSummaryProvider::
    DoesPrintValue(ValueObject *valobj) const {
  return false;
}

lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    SwiftOptionalSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()), m_is_none(false),
      m_children(false), m_some(nullptr) {}

bool lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::IsEmpty()
    const {
  return (m_is_none == true || m_children == false || m_some == nullptr);
}

size_t lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    CalculateNumChildren() {
  if (IsEmpty())
    return 0;
  return m_some->GetNumChildren();
}

lldb::ValueObjectSP lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (IsEmpty())
    return nullptr;
  auto child = m_some->GetChildAtIndex(idx, true);
  if (m_some->IsSyntheticChildrenGenerated())
    child->SetSyntheticChildrenGenerated(true);
  return child;
}

bool lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::Update() {
  m_some = nullptr;
  m_is_none = true;
  m_children = false;

  m_some = ExtractSomeIfAny(&m_backend, true);

  if (!m_some) {
    m_is_none = true;
    m_children = false;
    return false;
  }

  m_is_none = false;

  m_children = (m_some->GetNumChildren() > 0);

  return false;
}

bool lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    MightHaveChildren() {
  return IsEmpty() ? false : true;
}

size_t lldb_private::formatters::swift::SwiftOptionalSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  static ConstString g_Some("some");

  if (IsEmpty())
    return UINT32_MAX;

  return m_some->GetIndexOfChildWithName(name);
}

lldb::ValueObjectSP lldb_private::formatters::swift::
    SwiftOptionalSyntheticFrontEnd::GetSyntheticValue() {
  if (m_some && m_some->CanProvideValue())
    return m_some->GetSP();
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
