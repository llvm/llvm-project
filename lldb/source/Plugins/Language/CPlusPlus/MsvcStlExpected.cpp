//===----------------------------------------------------------------------===//
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

class MsvcStlExpectedFrontend : public SyntheticChildrenFrontEnd {
public:
  MsvcStlExpectedFrontend(ValueObject &valobj)
      : SyntheticChildrenFrontEnd(valobj) {
    if (valobj.GetTargetSP())
      Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_active_sp)
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
    return m_active_sp ? 1U : 0U;
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_active_sp || idx != 0)
      return {};
    return m_active_sp->Clone(m_has_value ? "Value" : "Unexpected");
  }

  lldb::ChildCacheState Update() override {
    m_active_sp.reset();
    m_has_value = false;
    ValueObjectSP ns = m_backend.GetNonSyntheticValue();
    if (!ns)
      return lldb::ChildCacheState::eRefetch;

    ValueObjectSP has_sp = ns->GetChildMemberWithName("_Has_value");
    if (!has_sp)
      return lldb::ChildCacheState::eRefetch;

    m_has_value = has_sp->GetValueAsUnsigned(0) != 0;
    // expected<void, E> has no value child when engaged.
    m_active_sp =
        ns->GetChildMemberWithName(m_has_value ? "_Value" : "_Unexpected");
    return lldb::ChildCacheState::eRefetch;
  }

private:
  ValueObjectSP m_active_sp;
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
