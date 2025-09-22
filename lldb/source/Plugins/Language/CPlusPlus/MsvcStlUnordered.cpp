//===-- MsvcStlUnordered.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

using namespace lldb;
using namespace lldb_private;

namespace {

class UnorderedFrontEnd : public SyntheticChildrenFrontEnd {
public:
  UnorderedFrontEnd(ValueObject &valobj) : SyntheticChildrenFrontEnd(valobj) {
    Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_list_sp)
      return llvm::createStringError("Missing _List");
    return m_list_sp->GetIndexOfChildWithName(name);
  }

  lldb::ChildCacheState Update() override;

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_list_sp)
      return llvm::createStringError("Missing _List");
    return m_list_sp->GetNumChildren();
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_list_sp)
      return nullptr;
    return m_list_sp->GetChildAtIndex(idx);
  }

private:
  ValueObjectSP m_list_sp;
};

} // namespace

lldb::ChildCacheState UnorderedFrontEnd::Update() {
  m_list_sp = nullptr;
  ValueObjectSP list_sp = m_backend.GetChildMemberWithName("_List");
  if (!list_sp)
    return lldb::ChildCacheState::eRefetch;
  m_list_sp = list_sp->GetSyntheticValue();
  return lldb::ChildCacheState::eRefetch;
}

bool formatters::IsMsvcStlUnordered(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_List") != nullptr;
  return false;
}

SyntheticChildrenFrontEnd *formatters::MsvcStlUnorderedSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new UnorderedFrontEnd(*valobj_sp);
  return nullptr;
}
