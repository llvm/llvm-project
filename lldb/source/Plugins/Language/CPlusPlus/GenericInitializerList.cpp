//===-- GenericInitializerList.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/ErrorExtras.h"
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace {

struct LibCxx {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("__begin_");
  }

  static uint64_t GetNumElements(ValueObject &backend, uint32_t) {
    if (ValueObjectSP size_sp = backend.GetChildMemberWithName("__size_"))
      return size_sp->GetValueAsUnsigned(0);
    return 0;
  }
};

struct LibStdcpp {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_M_array");
  }

  static uint64_t GetNumElements(ValueObject &backend, uint32_t) {
    if (ValueObjectSP size_sp = backend.GetChildMemberWithName("_M_len"))
      return size_sp->GetValueAsUnsigned(0);
    return 0;
  }
};

struct MsvcStl {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_First");
  }

  static uint64_t GetNumElements(ValueObject &backend, uint32_t element_size) {
    ValueObjectSP first_sp = backend.GetChildMemberWithName("_First");
    ValueObjectSP last_sp = backend.GetChildMemberWithName("_Last");
    if (!first_sp || !last_sp || element_size == 0)
      return 0;
    uint64_t first = first_sp->GetValueAsUnsigned(0);
    uint64_t last = last_sp->GetValueAsUnsigned(0);
    if (last < first)
      return 0;
    uint64_t bytes = last - first;
    if (bytes % element_size)
      return 0;
    return bytes / element_size;
  }
};

template <class StandardImpl>
class GenericInitializerListSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  GenericInitializerListSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
    if (valobj_sp)
      Update();
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return StandardImpl::GetNumElements(m_backend, m_element_size);
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_start)
      return {};

    uint64_t offset = static_cast<uint64_t>(idx) * m_element_size;
    offset = offset + m_start->GetValueAsUnsigned(0);
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateChildValueObjectFromAddress(name.GetString(), offset,
                                             m_backend.GetExecutionContextRef(),
                                             m_element_type);
  }

  lldb::ChildCacheState Update() override {
    m_start = nullptr;
    m_element_type = m_backend.GetCompilerType().GetTypeTemplateArgument(0);
    if (!m_element_type.IsValid())
      return lldb::ChildCacheState::eRefetch;

    llvm::Expected<uint64_t> size_or_err = m_element_type.GetByteSize(nullptr);
    if (!size_or_err)
      LLDB_LOG_ERRORV(GetLog(LLDBLog::DataFormatters), size_or_err.takeError(),
                      "{0}");
    else {
      m_element_size = *size_or_err;
      // Store raw pointers or end up with a circular dependency.
      m_start = StandardImpl::GetStartMember(m_backend).get();
    }

    return lldb::ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_start) {
      return llvm::createStringErrorV("type has no child named '{0}'", name);
    }
    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringErrorV("type has no child named '{0}'", name);
    }
    return *optional_idx;
  }

private:
  ValueObject *m_start = nullptr;
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
};

} // namespace

namespace lldb_private::formatters {

SyntheticChildrenFrontEnd *GenericInitializerListSyntheticFrontEndCreator(
    CXXSyntheticChildren * /*unused*/, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  if (LibCxx::GetStartMember(*valobj_sp) != nullptr)
    return new GenericInitializerListSyntheticFrontEnd<LibCxx>(valobj_sp);
  if (MsvcStl::GetStartMember(*valobj_sp) != nullptr)
    return new GenericInitializerListSyntheticFrontEnd<MsvcStl>(valobj_sp);

  return new GenericInitializerListSyntheticFrontEnd<LibStdcpp>(valobj_sp);
}
} // namespace lldb_private::formatters
