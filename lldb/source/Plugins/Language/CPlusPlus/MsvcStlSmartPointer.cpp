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

class MsvcStlUniquePtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlUniquePtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  lldb::ValueObjectSP m_value_ptr_sp;
  lldb::ValueObjectSP m_deleter_sp;
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

  ValueObjectSP ptr_obj_sp = valobj_sp->GetChildMemberWithName("_Ptr");
  if (!ptr_obj_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP cast_ptr_sp =
      GetDesugaredSmartPointerValue(*ptr_obj_sp, *valobj_sp);
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

bool lldb_private::formatters::IsMsvcStlUniquePtr(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Mypair") != nullptr;

  return false;
}

bool lldb_private::formatters::MsvcStlUniquePtrSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
  if (!valobj_sp)
    return false;

  ValueObjectSP ptr_sp(valobj_sp->GetChildAtNamePath({"_Mypair", "_Myval2"}));
  if (!ptr_sp)
    return false;

  DumpCxxSmartPtrPointerSummary(stream, *ptr_sp, options);

  return true;
}

lldb_private::formatters::MsvcStlUniquePtrSyntheticFrontEnd::
    MsvcStlUniquePtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlUniquePtrSyntheticFrontEnd::CalculateNumChildren() {
  if (m_value_ptr_sp)
    return m_deleter_sp ? 2 : 1;
  return 0;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlUniquePtrSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (!m_value_ptr_sp)
    return lldb::ValueObjectSP();

  if (idx == 0)
    return m_value_ptr_sp;

  if (idx == 1)
    return m_deleter_sp;

  if (idx == 2) {
    Status status;
    auto value_sp = m_value_ptr_sp->Dereference(status);
    if (status.Success()) {
      return value_sp;
    }
  }

  return lldb::ValueObjectSP();
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlUniquePtrSyntheticFrontEnd::Update() {
  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP pair_sp = valobj_sp->GetChildMemberWithName("_Mypair");
  if (!pair_sp)
    return lldb::ChildCacheState::eRefetch;

  if (auto value_ptr_sp = pair_sp->GetChildMemberWithName("_Myval2"))
    m_value_ptr_sp = value_ptr_sp->Clone(ConstString("pointer"));

  // Only present if the deleter is non-empty
  if (auto deleter_sp = pair_sp->GetChildMemberWithName("_Myval1"))
    m_deleter_sp = deleter_sp->Clone(ConstString("deleter"));

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::MsvcStlUniquePtrSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (name == "pointer")
    return 0;
  if (name == "deleter")
    return 1;
  if (name == "obj" || name == "object" || name == "$$dereference$$")
    return 2;
  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlUniquePtrSyntheticFrontEndCreator(
    lldb::ValueObjectSP valobj_sp) {
  return new MsvcStlUniquePtrSyntheticFrontEnd(valobj_sp);
}
