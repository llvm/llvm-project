//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <optional>

using namespace lldb;

namespace lldb_private::formatters {

class LibStdcppSpanSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibStdcppSpanSyntheticFrontEnd(const lldb::ValueObjectSP &valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {
    if (valobj_sp)
      Update();
  }

  ~LibStdcppSpanSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_num_elements;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_start)
      return {};

    uint64_t offset = (static_cast<uint64_t>(idx) * m_element_size);
    offset += m_start->GetValueAsUnsigned(0);
    const std::string name = llvm::formatv("[{0}]", idx);
    return CreateValueObjectFromAddress(
        name, offset, m_backend.GetExecutionContextRef(), m_element_type);
  }

  lldb::ChildCacheState Update() override {
    const ValueObjectSP element_ptr =
        m_backend.GetChildMemberWithName(ConstString("_M_ptr"));
    if (!element_ptr)
      return lldb::ChildCacheState::eRefetch;

    m_element_type = element_ptr->GetCompilerType().GetPointeeType();

    // Get element size.
    llvm::Expected<uint64_t> size_or_err = m_element_type.GetByteSize(nullptr);
    if (!size_or_err) {
      LLDB_LOG_ERRORV(GetLog(LLDBLog::DataFormatters), size_or_err.takeError(),
                      "{0}");
      return lldb::ChildCacheState::eReuse;
    }

    m_element_size = *size_or_err;
    if (m_element_size > 0) {
      m_start = element_ptr.get();
    }

    // Get number of elements.
    if (auto size_sp = m_backend.GetChildAtNamePath(
            {ConstString("_M_extent"), ConstString("_M_extent_value")})) {
      m_num_elements = size_sp->GetValueAsUnsigned(0);
    } else if (auto arg =
                   m_backend.GetCompilerType().GetIntegralTemplateArgument(1)) {

      m_num_elements = arg->value.GetAPSInt().getLimitedValue();
    }

    return lldb::ChildCacheState::eReuse;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_start)
      return llvm::createStringError(
          llvm::formatv("Type has no child named {0}", name.GetStringRef()));

    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringError(
          llvm::formatv("Type has no child named {0}", name.GetStringRef()));
    }
    return *optional_idx;
  }

private:
  ValueObject *m_start = nullptr; ///< First element of span. Held, not owned.
  CompilerType m_element_type;    ///< Type of span elements.
  size_t m_num_elements = 0;      ///< Number of elements in span.
  uint32_t m_element_size = 0;    ///< Size in bytes of each span element.
};

SyntheticChildrenFrontEnd *
LibStdcppSpanSyntheticFrontEndCreator(CXXSyntheticChildren * /*unused*/,
                                      lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  const CompilerType type = valobj_sp->GetCompilerType();
  if (!type || type.GetNumTemplateArguments() != 2)
    return nullptr;
  return new LibStdcppSpanSyntheticFrontEnd(valobj_sp);
}

} // namespace lldb_private::formatters
