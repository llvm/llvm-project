//===-- MsvcStlSmartPointer.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generic.h"
#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

using namespace lldb;

bool lldb_private::formatters::IsMsvcStlSmartPointer(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Ptr") != nullptr;

  return false;
}

bool lldb_private::formatters::MsvcStlSmartPointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
  if (!valobj_sp)
    return false;

  ValueObjectSP ptr_sp(valobj_sp->GetChildMemberWithName("_Ptr"));
  ValueObjectSP ctrl_sp(valobj_sp->GetChildMemberWithName("_Rep"));
  if (!ctrl_sp || !ptr_sp)
    return false;

  DumpCxxSmartPtrPointerSummary(stream, *ptr_sp, options);

  bool success;
  uint64_t ctrl_addr = ctrl_sp->GetValueAsUnsigned(0, &success);
  // Empty control field (expired)
  if (!success || ctrl_addr == 0)
    return true;

  uint64_t uses = 0;
  if (auto uses_sp = ctrl_sp->GetChildMemberWithName("_Uses")) {
    bool success;
    uses = uses_sp->GetValueAsUnsigned(0, &success);
    if (!success)
      return false;

    stream.Printf(" strong=%" PRIu64, uses);
  }

  // _Weaks is the number of weak references - (_Uses != 0).
  if (auto weak_count_sp = ctrl_sp->GetChildMemberWithName("_Weaks")) {
    bool success;
    uint64_t count = weak_count_sp->GetValueAsUnsigned(0, &success);
    if (!success)
      return false;

    stream.Printf(" weak=%" PRIu64, count - (uses != 0));
  }

  return true;
}

namespace lldb_private {
namespace formatters {

class MsvcStlSmartPointerSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlSmartPointerSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

  ~MsvcStlSmartPointerSyntheticFrontEnd() override;

private:
  ValueObject *m_ptr_obj = nullptr;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEnd::
    MsvcStlSmartPointerSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlSmartPointerSyntheticFrontEnd::CalculateNumChildren() {
  return (m_ptr_obj ? 1 : 0);
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (!m_ptr_obj)
    return lldb::ValueObjectSP();

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ValueObjectSP();

  if (idx == 0)
    return m_ptr_obj->GetSP();

  if (idx == 1) {
    Status status;
    ValueObjectSP value_sp = m_ptr_obj->Dereference(status);
    if (status.Success())
      return value_sp;
  }

  return lldb::ValueObjectSP();
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEnd::Update() {
  m_ptr_obj = nullptr;

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  auto ptr_obj_sp = valobj_sp->GetChildMemberWithName("_Ptr");
  if (!ptr_obj_sp)
    return lldb::ChildCacheState::eRefetch;

  auto cast_ptr_sp = GetDesugaredSmartPointerValue(*ptr_obj_sp, *valobj_sp);
  if (!cast_ptr_sp)
    return lldb::ChildCacheState::eRefetch;

  m_ptr_obj = cast_ptr_sp->Clone(ConstString("pointer")).get();
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (name == "pointer")
    return 0;

  if (name == "object" || name == "$$dereference$$")
    return 1;

  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEnd::
    ~MsvcStlSmartPointerSyntheticFrontEnd() = default;

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlSmartPointerSyntheticFrontEndCreator(
    lldb::ValueObjectSP valobj_sp) {
  return new MsvcStlSmartPointerSyntheticFrontEnd(valobj_sp);
}
