//===-- MsvcStlExpected.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

/// `_Value` / `_Unexpected` live in an anonymous union. Direct
/// GetChildMemberWithName from the class works on DWARF; PDB often needs the
/// same parent/index walk GenericOptional uses for `_Value`.
ValueObjectSP GetExpectedUnionMember(ValueObject &ns, const char *name) {
  if (ValueObjectSP direct = ns.GetChildMemberWithName(name))
    return direct;

  ValueObjectSP has_sp = ns.GetChildMemberWithName("_Has_value");
  if (!has_sp)
    return {};
  ValueObject *parent = has_sp->GetParent();
  if (!parent)
    return {};
  ValueObjectSP first_sp = parent->GetChildAtIndex(0);
  if (!first_sp)
    return {};
  if (ValueObjectSP nested = first_sp->GetChildMemberWithName(name))
    return nested;
  if (first_sp->GetName() == name)
    return first_sp;
  return {};
}

class MsvcStlExpectedFrontend : public SyntheticChildrenFrontEnd {
public:
  MsvcStlExpectedFrontend(ValueObject &valobj)
      : SyntheticChildrenFrontEnd(valobj) {
    if (valobj.GetTargetSP())
      Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_engaged)
      return llvm::createStringErrorV("type has no child named '{0}'", name);
    if (m_has_value) {
      if (name == "Value" || name == "$$dereference$$")
        return 0;
    } else if (name == "Unexpected") {
      return 0;
    }
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_engaged ? 1U : 0U;
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_engaged || idx != 0)
      return {};

    ValueObjectSP ns = m_backend.GetNonSyntheticValue();
    if (!ns)
      return {};

    if (m_has_value) {
      if (ValueObjectSP val_sp = GetExpectedUnionMember(*ns, "_Value"))
        return val_sp->Clone("Value");
      return {};
    }

    if (ValueObjectSP err_sp = GetExpectedUnionMember(*ns, "_Unexpected"))
      return err_sp->Clone("Unexpected");
    return {};
  }

  lldb::ChildCacheState Update() override {
    m_engaged = false;
    m_has_value = false;
    ValueObjectSP ns = m_backend.GetNonSyntheticValue();
    if (!ns)
      return lldb::ChildCacheState::eRefetch;

    ValueObjectSP has_sp = ns->GetChildMemberWithName("_Has_value");
    if (!has_sp)
      return lldb::ChildCacheState::eRefetch;

    m_has_value = has_sp->GetValueAsUnsigned(0) != 0;
    // expected<void, E> has no value child when engaged.
    if (m_has_value)
      m_engaged = GetExpectedUnionMember(*ns, "_Value") != nullptr;
    else
      m_engaged = GetExpectedUnionMember(*ns, "_Unexpected") != nullptr;
    return lldb::ChildCacheState::eRefetch;
  }

private:
  bool m_engaged = false;
  bool m_has_value = false;
};
} // namespace

bool formatters::IsMsvcStlExpected(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Has_value") != nullptr;
  return false;
}

bool formatters::MsvcStlExpectedSummaryProvider(ValueObject &valobj,
                                                Stream &stream,
                                                const TypeSummaryOptions &) {
  ValueObjectSP ns = valobj.GetNonSyntheticValue();
  if (!ns)
    return false;
  ValueObjectSP has_sp = ns->GetChildMemberWithName("_Has_value");
  if (!has_sp)
    return false;
  stream.Printf(" Has Value=%s ",
                has_sp->GetValueAsUnsigned(0) ? "true" : "false");
  return true;
}

SyntheticChildrenFrontEnd *formatters::MsvcStlExpectedSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new MsvcStlExpectedFrontend(*valobj_sp);
  return nullptr;
}
