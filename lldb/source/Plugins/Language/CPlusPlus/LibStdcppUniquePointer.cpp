//===-- LibStdcppUniquePointer.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/ConstString.h"

#include <memory>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

class LibStdcppUniquePtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  explicit LibStdcppUniquePtrSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

  bool GetSummary(Stream &stream, const TypeSummaryOptions &options);

private:
  ValueObjectSP m_ptr_obj;
  ValueObjectSP m_obj_obj;
  ValueObjectSP m_del_obj;
};

} // end of anonymous namespace

LibStdcppUniquePtrSyntheticFrontEnd::LibStdcppUniquePtrSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  Update();
}

bool LibStdcppUniquePtrSyntheticFrontEnd::Update() {
  ValueObjectSP valobj_backend_sp = m_backend.GetSP();
  if (!valobj_backend_sp)
    return false;

  ValueObjectSP valobj_sp = valobj_backend_sp->GetNonSyntheticValue();
  if (!valobj_sp)
    return false;

  ValueObjectSP tuple_sp =
      valobj_sp->GetChildMemberWithName(ConstString("_M_t"), true);
  if (!tuple_sp)
    return false;

  std::unique_ptr<SyntheticChildrenFrontEnd> tuple_frontend(
      LibStdcppTupleSyntheticFrontEndCreator(nullptr, tuple_sp));

  m_ptr_obj = tuple_frontend->GetChildAtIndex(0);
  if (m_ptr_obj)
    m_ptr_obj->SetName(ConstString("pointer"));

  m_del_obj = tuple_frontend->GetChildAtIndex(1);
  if (m_del_obj)
    m_del_obj->SetName(ConstString("deleter"));

  if (m_ptr_obj) {
    Error error;
    m_obj_obj = m_ptr_obj->Dereference(error);
    if (error.Success()) {
      m_obj_obj->SetName(ConstString("object"));
    }
  }

  return false;
}

bool LibStdcppUniquePtrSyntheticFrontEnd::MightHaveChildren() { return true; }

lldb::ValueObjectSP
LibStdcppUniquePtrSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx == 0)
    return m_obj_obj;
  if (idx == 1)
    return m_del_obj;
  if (idx == 2)
    return m_ptr_obj;
  return lldb::ValueObjectSP();
}

size_t LibStdcppUniquePtrSyntheticFrontEnd::CalculateNumChildren() {
  if (m_del_obj)
    return 2;
  if (m_ptr_obj && m_ptr_obj->GetValueAsUnsigned(0) != 0)
    return 1;
  return 0;
}

size_t LibStdcppUniquePtrSyntheticFrontEnd::GetIndexOfChildWithName(
    const ConstString &name) {
  if (name == ConstString("obj") || name == ConstString("object"))
    return 0;
  if (name == ConstString("del") || name == ConstString("deleter"))
    return 1;
  if (name == ConstString("ptr") || name == ConstString("pointer"))
    return 2;
  return UINT32_MAX;
}

bool LibStdcppUniquePtrSyntheticFrontEnd::GetSummary(
    Stream &stream, const TypeSummaryOptions &options) {
  if (!m_ptr_obj)
    return false;

  bool success;
  uint64_t ptr_value = m_ptr_obj->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;
  if (ptr_value == 0)
    stream.Printf("nullptr");
  else
    stream.Printf("0x%" PRIx64, ptr_value);
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibStdcppUniquePtrSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new LibStdcppUniquePtrSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}

bool lldb_private::formatters::LibStdcppUniquePointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  LibStdcppUniquePtrSyntheticFrontEnd formatter(valobj.GetSP());
  return formatter.GetSummary(stream, options);
}
