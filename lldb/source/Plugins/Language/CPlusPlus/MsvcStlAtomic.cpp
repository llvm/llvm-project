//===-- MsvcStlAtomic.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/TypeSynthetic.h"

using namespace lldb;

namespace lldb_private {
namespace formatters {

class MsvcStlAtomicSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObject *m_storage = nullptr;
  CompilerType m_element_type;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::
    MsvcStlAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlAtomicSyntheticFrontEnd::CalculateNumChildren() {
  return m_storage ? 1 : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (idx == 0 && m_storage && m_element_type.IsValid())
    return m_storage->Cast(m_element_type)->Clone(ConstString("Value"));
  return nullptr;
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::Update() {
  m_storage = nullptr;
  m_element_type.Clear();

  ValueObjectSP storage_sp = m_backend.GetChildMemberWithName("_Storage");
  if (!storage_sp)
    return lldb::ChildCacheState::eRefetch;

  m_element_type = m_backend.GetCompilerType().GetTypeTemplateArgument(0);
  if (!m_element_type)
    return lldb::ChildCacheState::eRefetch;

  m_storage = storage_sp.get();
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t> lldb_private::formatters::
    MsvcStlAtomicSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (name == "Value")
    return 0;
  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp && IsMsvcStlAtomic(*valobj_sp))
    return new MsvcStlAtomicSyntheticFrontEnd(valobj_sp);
  return nullptr;
}

bool lldb_private::formatters::MsvcStlAtomicSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto synth_sp = valobj.GetSyntheticValue();
  if (!synth_sp)
    return false;

  auto value_sp = synth_sp->GetChildAtIndex(0);
  std::string summary;
  if (value_sp->GetSummaryAsCString(summary, options) && !summary.empty()) {
    stream << summary;
    return true;
  }
  return false;
}

bool lldb_private::formatters::IsMsvcStlAtomic(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Storage") != nullptr;
  return false;
}
