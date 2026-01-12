//===-- GenericInitializerList.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/ADT/STLForwardCompat.h"
#include <cstddef>
#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace generic_check {
template <class T>
using size_func = decltype(T::GetSizeMember(std::declval<ValueObject &>()));
template <class T>
using start_func = decltype(T::GetStartMember(std::declval<ValueObject &>()));
template <class T>
using end_func = decltype(T::GetEndMember(std::declval<ValueObject &>()));

template <typename T>
using has_start_function = llvm::is_detected<start_func, T>;
template <typename T> using has_size_function = llvm::is_detected<size_func, T>;
template <typename T> using has_end_function = llvm::is_detected<end_func, T>;
} // namespace generic_check

struct LibCxx {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("__begin_");
  }

  static ValueObjectSP GetSizeMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("__size_");
  }
};

struct LibStdcpp {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_M_array");
  }

  static ValueObjectSP GetSizeMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_M_len");
  }
};

struct MsvcStl {
  static ValueObjectSP GetStartMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_First");
  }

  static ValueObjectSP GetEndMember(ValueObject &backend) {
    return backend.GetChildMemberWithName("_Last");
  }
};

namespace lldb_private::formatters {

template <class StandardImpl>
class GenericInitializerListSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  static_assert(generic_check::has_start_function<StandardImpl>::value,
                "Missing start function.");

  GenericInitializerListSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
    if (valobj_sp)
      Update();
  }

  ~GenericInitializerListSyntheticFrontEnd() override {
    // this needs to stay around because it's a child object who will follow its
    // parent's life cycle
    // delete m_start;
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_num_elements;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_start || idx >= m_num_elements)
      return {};

    uint64_t offset = static_cast<uint64_t>(idx) * m_element_size;
    offset = offset + m_start->GetValueAsUnsigned(0);
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateValueObjectFromAddress(name.GetString(), offset,
                                        m_backend.GetExecutionContextRef(),
                                        m_element_type);
  }

  lldb::ChildCacheState Update() override {
    m_start = nullptr;
    m_num_elements = 0;
    // Store raw pointers or end up with a circular dependency.
    m_start = StandardImpl::GetStartMember(m_backend).get();
    if (!m_start)
      return lldb::ChildCacheState::eRefetch;

    m_element_type = m_backend.GetCompilerType().GetTypeTemplateArgument(0);
    if (!m_element_type) {
      // PDB doesn't have template types, so get the element type from the start
      // pointer.
      m_element_type = m_start->GetCompilerType().GetPointeeType();
      if (!m_element_type)
        return lldb::ChildCacheState::eRefetch;
    }

    llvm::Expected<uint64_t> size_or_err = m_element_type.GetByteSize(nullptr);
    if (!size_or_err)
      LLDB_LOG_ERRORV(GetLog(LLDBLog::DataFormatters), size_or_err.takeError(),
                      "{0}");
    else
      m_element_size = *size_or_err;

    if (m_element_size == 0)
      return lldb::ChildCacheState::eRefetch;

    if constexpr (generic_check::has_size_function<StandardImpl>::value) {
      const ValueObjectSP size_sp(StandardImpl::GetSizeMember(m_backend));
      if (size_sp)
        m_num_elements = size_sp->GetValueAsUnsigned(0);
    } else {
      static_assert(generic_check::has_end_function<StandardImpl>::value,
                    "Must have size or end function");
      ValueObjectSP end_sp = StandardImpl::GetEndMember(m_backend);
      if (!end_sp)
        return lldb::ChildCacheState::eRefetch;

      uint64_t start = m_start->GetValueAsUnsigned(0);
      uint64_t end = end_sp->GetValueAsUnsigned(0);
      if (end < start)
        return lldb::ChildCacheState::eRefetch;
      m_num_elements = (end - start) / m_element_size;
    }

    return lldb::ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_start) {
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }
    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }
    return *optional_idx;
  }

private:
  ValueObject *m_start = nullptr;
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
  size_t m_num_elements = 0;
};

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
