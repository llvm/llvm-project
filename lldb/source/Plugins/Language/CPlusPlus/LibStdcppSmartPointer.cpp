//===-- LibStdcppSmartPointer.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/Target.h"

#include <memory>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

class SharedPtrFrontEnd : public SyntheticChildrenFrontEnd {
public:
  explicit SharedPtrFrontEnd(lldb::ValueObjectSP valobj_sp);

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

  bool GetSummary(Stream &stream, const TypeSummaryOptions &options);

private:
  ValueObjectSP m_ptr_obj;
  ValueObjectSP m_obj_obj;
  ValueObjectSP m_use_obj;
  ValueObjectSP m_weak_obj;

  uint8_t m_ptr_size = 0;
  lldb::ByteOrder m_byte_order = lldb::eByteOrderInvalid;

  bool IsEmpty();
  bool IsValid();
};

} // end of anonymous namespace

SharedPtrFrontEnd::SharedPtrFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  Update();
}

bool SharedPtrFrontEnd::Update() {
  ValueObjectSP valobj_backend_sp = m_backend.GetSP();
  if (!valobj_backend_sp)
    return false;

  ValueObjectSP valobj_sp = valobj_backend_sp->GetNonSyntheticValue();
  if (!valobj_sp)
    return false;

  TargetSP target_sp(valobj_sp->GetTargetSP());
  if (!target_sp)
    return false;

  m_byte_order = target_sp->GetArchitecture().GetByteOrder();
  m_ptr_size = target_sp->GetArchitecture().GetAddressByteSize();

  m_ptr_obj = valobj_sp->GetChildMemberWithName(ConstString("_M_ptr"), true);

  m_use_obj = valobj_sp->GetChildAtNamePath({ConstString("_M_refcount"),
                                             ConstString("_M_pi"),
                                             ConstString("_M_use_count")});

  m_weak_obj = valobj_sp->GetChildAtNamePath({ConstString("_M_refcount"),
                                              ConstString("_M_pi"),
                                              ConstString("_M_weak_count")});

  // libstdc++ implements the weak usage count in a way that it is offset by 1
  // if the strong count is not 0 (as part of a preformance optimization). We
  // want to undo this before showing the weak count to the user as an offseted
  // weak count would be very confusing.
  if (m_use_obj && m_weak_obj && m_use_obj->GetValueAsUnsigned(0) > 0) {
    bool success = false;
    uint64_t count = m_weak_obj->GetValueAsUnsigned(0, &success) - 1;
    if (success) {
      auto data = std::make_shared<DataBufferHeap>(&count, sizeof(count));
      m_weak_obj = CreateValueObjectFromData(
          "weak_count", DataExtractor(data, m_byte_order, m_ptr_size),
          m_weak_obj->GetExecutionContextRef(), m_weak_obj->GetCompilerType());
    }
  }

  if (m_ptr_obj && !IsEmpty()) {
    Error error;
    m_obj_obj = m_ptr_obj->Dereference(error);
    if (error.Success()) {
      m_obj_obj->SetName(ConstString("object"));
    }
  }

  return false;
}

bool SharedPtrFrontEnd::MightHaveChildren() { return true; }

lldb::ValueObjectSP SharedPtrFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx == 0)
    return m_obj_obj;
  if (idx == 1)
    return m_ptr_obj;
  if (idx == 2)
    return m_use_obj;
  if (idx == 3)
    return m_weak_obj;
  return lldb::ValueObjectSP();
}

size_t SharedPtrFrontEnd::CalculateNumChildren() {
  if (IsEmpty())
    return 0;
  return 1;
}

size_t SharedPtrFrontEnd::GetIndexOfChildWithName(const ConstString &name) {
  if (name == ConstString("obj") || name == ConstString("object"))
    return 0;
  if (name == ConstString("ptr") || name == ConstString("pointer") ||
      name == ConstString("_M_ptr"))
    return 1;
  if (name == ConstString("cnt") || name == ConstString("count") ||
      name == ConstString("use_count") || name == ConstString("strong") ||
      name == ConstString("_M_use_count"))
    return 2;
  if (name == ConstString("weak") || name == ConstString("weak_count") ||
      name == ConstString("_M_weak_count"))
    return 3;
  return UINT32_MAX;
}

bool SharedPtrFrontEnd::GetSummary(Stream &stream,
                                   const TypeSummaryOptions &options) {
  if (!IsValid())
    return false;

  if (IsEmpty()) {
    stream.Printf("nullptr");
  } else {
    Error error;
    bool print_pointee = false;
    if (m_obj_obj) {
      if (m_obj_obj->DumpPrintableRepresentation(
              stream, ValueObject::eValueObjectRepresentationStyleSummary,
              lldb::eFormatInvalid,
              ValueObject::ePrintableRepresentationSpecialCasesDisable,
              false)) {
        print_pointee = true;
      }
    }
    if (!print_pointee)
      stream.Printf("ptr = 0x%" PRIx64, m_ptr_obj->GetValueAsUnsigned(0));
  }

  if (m_use_obj && m_use_obj->GetError().Success())
    stream.Printf(" strong=%" PRIu64, m_use_obj->GetValueAsUnsigned(0));

  if (m_weak_obj && m_weak_obj->GetError().Success())
    stream.Printf(" weak=%" PRIu64, m_weak_obj->GetValueAsUnsigned(0));

  return true;
}

bool SharedPtrFrontEnd::IsValid() { return m_ptr_obj != nullptr; }

bool SharedPtrFrontEnd::IsEmpty() {
  return !IsValid() || m_ptr_obj->GetValueAsUnsigned(0) == 0 ||
         (m_use_obj && m_use_obj->GetValueAsUnsigned(0) == 0);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibStdcppSharedPtrSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return valobj_sp ? new SharedPtrFrontEnd(valobj_sp) : nullptr;
}

bool lldb_private::formatters::LibStdcppSmartPointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  SharedPtrFrontEnd formatter(valobj.GetSP());
  return formatter.GetSummary(stream, options);
}
