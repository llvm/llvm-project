//===-- MsvcStlTuple.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;

namespace {

class TupleFrontEnd : public SyntheticChildrenFrontEnd {
public:
  TupleFrontEnd(ValueObject &valobj) : SyntheticChildrenFrontEnd(valobj) {
    Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }
    return *optional_idx;
  }

  lldb::ChildCacheState Update() override;
  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_elements.size();
  }
  ValueObjectSP GetChildAtIndex(uint32_t idx) override;

private:
  // The lifetime of a ValueObject and all its derivative ValueObjects
  // (children, clones, etc.) is managed by a ClusterManager. These
  // objects are only destroyed when every shared pointer to any of them
  // is destroyed, so we must not store a shared pointer to any ValueObject
  // derived from our backend ValueObject (since we're in the same cluster).
  std::vector<ValueObject *> m_elements;
};

} // namespace

lldb::ChildCacheState TupleFrontEnd::Update() {
  m_elements.clear();

  size_t n_elements = 0;
  for (CompilerType ty = m_backend.GetCompilerType();
       ty.GetNumDirectBaseClasses() > 0;
       ty = ty.GetDirectBaseClassAtIndex(0, nullptr))
    ++n_elements;

  m_elements.assign(n_elements, nullptr);
  return lldb::ChildCacheState::eRefetch;
}

ValueObjectSP TupleFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= m_elements.size())
    return nullptr;
  if (m_elements[idx])
    return m_elements[idx]->GetSP();

  CompilerType holder_ty = m_backend.GetCompilerType();
  for (uint32_t i = 0; i < idx; i++) {
    holder_ty = holder_ty.GetDirectBaseClassAtIndex(0, nullptr);
    if (!holder_ty.IsValid())
      return nullptr;
  }

  ValueObjectSP holder_sp = m_backend.Cast(holder_ty);
  if (!holder_sp)
    return nullptr;
  holder_sp = holder_sp->GetChildMemberWithName("_Myfirst");

  if (!holder_sp)
    return nullptr;

  ValueObjectSP val_sp = holder_sp->GetChildMemberWithName("_Val");
  if (!val_sp)
    return nullptr;

  m_elements[idx] =
      val_sp->Clone(ConstString(llvm::formatv("[{0}]", idx).str())).get();
  return m_elements[idx]->GetSP();
}

bool formatters::IsMsvcStlTuple(ValueObject &valobj) {
  // This returns false for empty tuples, but the libstdc++ formatter handles
  // this correctly.
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Myfirst") != nullptr;
  return false;
}

SyntheticChildrenFrontEnd *formatters::MsvcStlTupleSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new TupleFrontEnd(*valobj_sp);
  return nullptr;
}
