//===-- MsvcStlVector.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

using namespace lldb;

namespace lldb_private {
namespace formatters {

class MsvcStlVectorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObject *m_start = nullptr;
  ValueObject *m_finish = nullptr;
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
};

class MsvcStlVectorBoolSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlVectorBoolSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  CompilerType m_bool_type;
  ExecutionContextRef m_exe_ctx_ref;
  uint64_t m_count = 0;
  uint64_t m_element_bit_size = 0;
  lldb::addr_t m_base_data_address = 0;
  std::map<size_t, lldb::ValueObjectSP> m_children;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::MsvcStlVectorSyntheticFrontEnd::
    MsvcStlVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlVectorSyntheticFrontEnd::CalculateNumChildren() {
  if (!m_start || !m_finish)
    return llvm::createStringError(
        "failed to determine start/end of vector data");

  uint64_t start_val = m_start->GetValueAsUnsigned(0);
  uint64_t finish_val = m_finish->GetValueAsUnsigned(0);

  // A default-initialized empty vector.
  if (start_val == 0 && finish_val == 0)
    return 0;

  if (start_val == 0)
    return llvm::createStringError("invalid value for start of vector");

  if (finish_val == 0)
    return llvm::createStringError("invalid value for end of vector");

  if (start_val > finish_val)
    return llvm::createStringError(
        "start of vector data begins after end pointer");

  size_t num_children = (finish_val - start_val);
  if (num_children % m_element_size)
    return llvm::createStringError("size not multiple of element size");

  return num_children / m_element_size;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlVectorSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (!m_start || !m_finish)
    return lldb::ValueObjectSP();

  uint64_t offset = idx * m_element_size;
  offset = offset + m_start->GetValueAsUnsigned(0);
  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateValueObjectFromAddress(name.GetString(), offset,
                                      m_backend.GetExecutionContextRef(),
                                      m_element_type);
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlVectorSyntheticFrontEnd::Update() {
  m_start = m_finish = nullptr;
  ValueObjectSP data_sp(m_backend.GetChildAtNamePath({"_Mypair", "_Myval2"}));

  if (!data_sp)
    return lldb::ChildCacheState::eRefetch;

  m_start = data_sp->GetChildMemberWithName("_Myfirst").get();
  m_finish = data_sp->GetChildMemberWithName("_Mylast").get();
  if (!m_start || !m_finish)
    return lldb::ChildCacheState::eRefetch;

  m_element_type = m_start->GetCompilerType().GetPointeeType();
  llvm::Expected<uint64_t> size_or_err = m_element_type.GetByteSize(nullptr);
  if (size_or_err)
    m_element_size = *size_or_err;
  else
    LLDB_LOG_ERRORV(GetLog(LLDBLog::DataFormatters), size_or_err.takeError(),
                    "{0}");

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t> lldb_private::formatters::
    MsvcStlVectorSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (!m_start || !m_finish)
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  auto optional_idx = ExtractIndexFromString(name.GetCString());
  if (!optional_idx) {
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }
  return *optional_idx;
}

lldb_private::formatters::MsvcStlVectorBoolSyntheticFrontEnd::
    MsvcStlVectorBoolSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_bool_type(), m_exe_ctx_ref(),
      m_children() {
  if (valobj_sp) {
    Update();
    m_bool_type =
        valobj_sp->GetCompilerType().GetBasicTypeFromAST(lldb::eBasicTypeBool);
  }
}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlVectorBoolSyntheticFrontEnd::CalculateNumChildren() {
  return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlVectorBoolSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  auto iter = m_children.find(idx), end = m_children.end();
  if (iter != end)
    return iter->second;
  if (idx >= m_count)
    return {};
  if (m_base_data_address == 0 || m_count == 0)
    return {};
  if (!m_bool_type)
    return {};

  // The vector<bool> is represented as a sequence of `int`s.
  // The size of an `int` is in `m_element_bit_size` (most often 32b).
  // To access the element at index `i`:
  // (bool)((data_address[i / bit_size] >> (i % bit_size)) & 1)

  // int *byte_location = &data_address[i / bit_size]
  size_t byte_idx = (idx / m_element_bit_size) * (m_element_bit_size / 8);
  lldb::addr_t byte_location = m_base_data_address + byte_idx;

  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return {};
  Status err;
  Scalar scalar;
  size_t bytes_read = process_sp->ReadScalarIntegerFromMemory(
      byte_location, m_element_bit_size / 8, false, scalar, err);
  if (err.Fail() || bytes_read == 0 || !scalar.IsValid())
    return {};

  size_t bit_index = idx % m_element_bit_size;
  bool bit_set = scalar.GetAPSInt()[bit_index];
  std::optional<uint64_t> size =
      llvm::expectedToOptional(m_bool_type.GetByteSize(nullptr));
  if (!size)
    return {};
  WritableDataBufferSP buffer_sp(new DataBufferHeap(*size, 0));
  if (bit_set && buffer_sp && buffer_sp->GetBytes()) {
    // regardless of endianness, anything non-zero is true
    *(buffer_sp->GetBytes()) = 1;
  }
  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  ValueObjectSP retval_sp(CreateValueObjectFromData(
      name.GetString(),
      DataExtractor(buffer_sp, process_sp->GetByteOrder(),
                    process_sp->GetAddressByteSize()),
      m_exe_ctx_ref, m_bool_type));
  if (retval_sp)
    m_children[idx] = retval_sp;
  return retval_sp;
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlVectorBoolSyntheticFrontEnd::Update() {
  m_exe_ctx_ref.Clear();
  m_count = 0;
  m_element_bit_size = 0;
  m_base_data_address = 0;
  m_children.clear();

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;
  auto exe_ctx_ref = valobj_sp->GetExecutionContextRef();

  ValueObjectSP size_sp = valobj_sp->GetChildMemberWithName("_Mysize");
  if (!size_sp)
    return lldb::ChildCacheState::eRefetch;
  uint64_t count = size_sp->GetValueAsUnsigned(0);
  if (count == 0)
    return lldb::ChildCacheState::eReuse;

  ValueObjectSP begin_sp(valobj_sp->GetChildAtNamePath(
      {"_Myvec", "_Mypair", "_Myval2", "_Myfirst"}));
  if (!begin_sp)
    return lldb::ChildCacheState::eRefetch;

  // FIXME: the STL exposes _EEN_VBITS as a constant - it should be used instead
  CompilerType begin_ty = begin_sp->GetCompilerType().GetPointeeType();
  if (!begin_ty.IsValid())
    return lldb::ChildCacheState::eRefetch;
  llvm::Expected<uint64_t> element_bit_size = begin_ty.GetBitSize(nullptr);
  if (!element_bit_size)
    return lldb::ChildCacheState::eRefetch;

  uint64_t base_data_address = begin_sp->GetValueAsUnsigned(0);
  if (!base_data_address)
    return lldb::ChildCacheState::eRefetch;

  m_exe_ctx_ref = exe_ctx_ref;
  m_count = count;
  m_element_bit_size = *element_bit_size;
  m_base_data_address = base_data_address;
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::MsvcStlVectorBoolSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (!m_count || !m_base_data_address)
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  auto optional_idx = ExtractIndexFromString(name.AsCString());
  if (!optional_idx) {
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }
  uint32_t idx = *optional_idx;
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  return idx;
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlVectorSyntheticFrontEndCreator(
    lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  valobj_sp = valobj_sp->GetNonSyntheticValue();
  if (!valobj_sp)
    return nullptr;

  // We can't check the template parameter here, because PDB doesn't include
  // this information.

  // vector<T>
  if (valobj_sp->GetChildMemberWithName("_Mypair") != nullptr)
    return new MsvcStlVectorSyntheticFrontEnd(valobj_sp);
  // vector<bool>
  if (valobj_sp->GetChildMemberWithName("_Myvec") != nullptr)
    return new MsvcStlVectorBoolSyntheticFrontEnd(valobj_sp);

  return nullptr;
}
