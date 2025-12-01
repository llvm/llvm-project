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
#include <cstddef>
#include <optional>
#include <type_traits>

using namespace lldb;
using namespace lldb_private;

namespace generic_check {
template <class T>
using size_func = decltype(T::GetSizeMember(std::declval<ValueObject &>()));
template <class T>
using start_func = decltype(T::GetStartMember(std::declval<ValueObject &>()));
namespace {
template <typename...> struct check_func : std::true_type {};
} // namespace

template <typename T>
using has_functions = check_func<size_func<T>, start_func<T>>;
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

namespace lldb_private::formatters {

template <class StandardImpl>
class GenericInitializerListSyntheticFrontEnd
    : public SyntheticChildrenFrontEnd {
public:
  static_assert(generic_check::has_functions<StandardImpl>::value,
                "Missing Required Functions.");

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
    m_num_elements = 0;

    const ValueObjectSP size_sp(StandardImpl::GetSizeMember(m_backend));
    if (size_sp)
      m_num_elements = size_sp->GetValueAsUnsigned(0);
    return m_num_elements;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_start)
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

  return new GenericInitializerListSyntheticFrontEnd<LibStdcpp>(valobj_sp);
}
} // namespace lldb_private::formatters
