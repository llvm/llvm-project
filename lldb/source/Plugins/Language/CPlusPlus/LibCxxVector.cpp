//===-- LibCxxVector.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibCxx.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorExtras.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace lldb_private {
namespace formatters {
class LibcxxStdVectorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibcxxStdVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~LibcxxStdVectorSyntheticFrontEnd() override;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  lldb::ChildCacheState UpdateVectorWithLayoutSubobject(ValueObject *layout);

  ValueObject *m_start = nullptr;
  ValueObject *m_finish = nullptr;
  enum class VectorLayout : bool { Pointer, Size };
  VectorLayout m_layout;
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
};

class LibcxxVectorBoolSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibcxxVectorBoolSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  CompilerType m_bool_type;
  ExecutionContextRef m_exe_ctx_ref;
  uint64_t m_count = 0;
  lldb::addr_t m_base_data_address = 0;
  std::map<size_t, lldb::ValueObjectSP> m_children;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::
    LibcxxStdVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
}

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::
    ~LibcxxStdVectorSyntheticFrontEnd() {
  // these need to stay around because they are child objects who will follow
  // their parent's life cycle
  // delete m_start;
  // delete m_finish;
}

static llvm::Expected<uint32_t> CalculateNumChildrenUsingPointerArithmetic(ValueObject* begin, ValueObject* end, uint64_t value_type_size) {
  uint64_t start_val = begin->GetValueAsUnsigned(0);
  uint64_t finish_val = end->GetValueAsUnsigned(0);

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
  if (num_children % value_type_size)
    return llvm::createStringError("size not multiple of element size");

  return num_children / value_type_size;
}

static llvm::Expected<uint32_t> GetNumChildren(ValueObject* size) {
  if (!size->GetCompilerType().IsInteger())
    return llvm::createStringError("size data member must be a built-in integer type");
  return size->GetValueAsUnsigned(0);
}

llvm::Expected<uint32_t> lldb_private::formatters::
    LibcxxStdVectorSyntheticFrontEnd::CalculateNumChildren() {
  if (!m_start || !m_finish)
    return llvm::createStringError(
        "failed to determine start/end of vector data");

  switch (m_layout) {
    case VectorLayout::Pointer:
      return CalculateNumChildrenUsingPointerArithmetic(m_start, m_finish, m_element_size);
    case VectorLayout::Size:
      return GetNumChildren(m_finish);
  }
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (!m_start || !m_finish)
    return lldb::ValueObjectSP();

  uint64_t offset = idx * m_element_size;
  offset = offset + m_start->GetValueAsUnsigned(0);
  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateChildValueObjectFromAddress(name.GetString(), offset,
                                           m_backend.GetExecutionContextRef(),
                                           m_element_type);
}

lldb::ChildCacheState
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::Update() {
  m_start = m_finish = nullptr;

  // Determine if this version of libc++'s `std::vector` uses `__vector_layout`.
  ValueObjectSP layout_sp = m_backend.GetChildMemberWithName("__layout_");
  ValueObject *target = layout_sp ? layout_sp.get() : &m_backend;

  ValueObjectSP begin_sp = target->GetChildMemberWithName("__begin_");
  if (!begin_sp)
    return lldb::ChildCacheState::eRefetch;

  m_element_type = begin_sp->GetCompilerType().GetPointeeType();
  llvm::Expected<uint64_t> size_or_err = m_element_type.GetByteSize(nullptr);
  if (!size_or_err) {
    LLDB_LOG_ERRORV(GetLog(LLDBLog::DataFormatters), size_or_err.takeError(),
                    "{0}");
    return lldb::ChildCacheState::eRefetch;
  }

  m_element_size = *size_or_err;
  if (m_element_size == 0) {
    return lldb::ChildCacheState::eRefetch;
  }

  // store raw pointers or end up with a circular dependency
  m_start = begin_sp.get();

  if (ValueObjectSP end_sp = target->GetChildMemberWithName("__end_")) {
    m_finish = end_sp.get();
    m_layout = VectorLayout::Pointer;
    return lldb::ChildCacheState::eRefetch;
  }

  ValueObjectSP size_sp = target->GetChildMemberWithName("__size_");
  if (!size_sp)
    return lldb::ChildCacheState::eRefetch;

  m_finish = size_sp.get();
  m_layout = VectorLayout::Size;
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (!m_start || !m_finish)
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
  if (!optional_idx) {
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }
  return *optional_idx;
}

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::
    LibcxxVectorBoolSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_bool_type(), m_exe_ctx_ref(),
      m_children() {
  if (valobj_sp) {
    Update();
    m_bool_type =
        valobj_sp->GetCompilerType().GetBasicTypeFromAST(lldb::eBasicTypeBool);
  }
}

llvm::Expected<uint32_t> lldb_private::formatters::
    LibcxxVectorBoolSyntheticFrontEnd::CalculateNumChildren() {
  return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetChildAtIndex(
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
  size_t byte_idx = (idx >> 3); // divide by 8 to get byte index
  size_t bit_index = (idx & 7); // efficient idx % 8 for bit index
  lldb::addr_t byte_location = m_base_data_address + byte_idx;
  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return {};
  uint8_t byte = 0;
  uint8_t mask = 0;
  Status err;
  size_t bytes_read = process_sp->ReadMemory(byte_location, &byte, 1, err);
  if (err.Fail() || bytes_read == 0)
    return {};
  mask = 1 << bit_index;
  bool bit_set = ((byte & mask) != 0);
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
  ValueObjectSP retval_sp = CreateChildValueObjectFromData(
      name.GetString(),
      DataExtractor(buffer_sp, process_sp->GetByteOrder(),
                    process_sp->GetAddressByteSize()),
      m_exe_ctx_ref, m_bool_type);
  if (retval_sp)
    m_children[idx] = retval_sp;
  return retval_sp;
}

lldb::ChildCacheState
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::Update() {
  m_children.clear();
  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
  ValueObjectSP size_sp(valobj_sp->GetChildMemberWithName("__size_"));
  if (!size_sp)
    return lldb::ChildCacheState::eRefetch;
  m_count = size_sp->GetValueAsUnsigned(0);
  if (!m_count)
    return lldb::ChildCacheState::eReuse;
  ValueObjectSP begin_sp(valobj_sp->GetChildMemberWithName("__begin_"));
  if (!begin_sp) {
    m_count = 0;
    return lldb::ChildCacheState::eRefetch;
  }
  m_base_data_address = begin_sp->GetValueAsUnsigned(0);
  if (!m_base_data_address) {
    m_count = 0;
    return lldb::ChildCacheState::eRefetch;
  }
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::
    GetIndexOfChildWithName(ConstString name) {
  if (!m_count || !m_base_data_address)
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  auto optional_idx = ExtractIndexFromString(name.AsCString(nullptr));
  if (!optional_idx) {
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }
  uint32_t idx = *optional_idx;
  if (idx >= CalculateNumChildrenIgnoringErrors())
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  return idx;
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  CompilerType type = valobj_sp->GetCompilerType();
  if (!type.IsValid() || type.GetNumTemplateArguments() == 0)
    return nullptr;
  CompilerType arg_type = type.GetTypeTemplateArgument(0);
  if (arg_type.GetTypeName() == "bool")
    return new LibcxxVectorBoolSyntheticFrontEnd(valobj_sp);
  return new LibcxxStdVectorSyntheticFrontEnd(valobj_sp);
}
