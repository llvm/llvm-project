//===-- RustVector.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "RustLanguage.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
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
class RustVectorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  RustVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~RustVectorSyntheticFrontEnd() override;

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(llvm::StringRef name) override;

private:
  ValueObject *m_start = nullptr;

  ValueObject *m_len = nullptr;
  enum class VectorLayout : bool { Pointer, Size };
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
};

} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::RustVectorSyntheticFrontEnd::
    RustVectorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_element_type() {
  if (valobj_sp)
    Update();
}

lldb_private::formatters::RustVectorSyntheticFrontEnd::
    ~RustVectorSyntheticFrontEnd() {
  // these need to stay around because they are child objects who will follow
  // their parent's life cycle
  // delete m_start;
  // delete m_len;
}

llvm::Expected<uint32_t>
lldb_private::formatters::RustVectorSyntheticFrontEnd::CalculateNumChildren() {
  if (!m_len || !m_len->GetCompilerType().IsInteger())
    return llvm::createStringError("Invalid vector length");
  return m_len->GetValueAsUnsigned(0);
}

lldb::ValueObjectSP
lldb_private::formatters::RustVectorSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (!m_start || !m_len)
    return lldb::ValueObjectSP();

  if (idx >= CalculateNumChildrenIgnoringErrors())
    return lldb::ValueObjectSP();

  uint64_t offset = m_start->GetValueAsUnsigned(0) + idx * m_element_size;
  StreamString name;
  name.Printf("[%" PRIu64 "]", (uint64_t)idx);
  return CreateChildValueObjectFromAddress(name.GetString(), offset,
                                           m_backend.GetExecutionContextRef(),
                                           m_element_type);
}

lldb::ChildCacheState
lldb_private::formatters::RustVectorSyntheticFrontEnd::Update() {
  m_start = m_len = nullptr;

  m_start =
      m_backend
          .GetChildAtNamePath({"buf", "inner", "ptr", "pointer", "pointer"})
          .get();

  // If rust program was compiled with -Zrandomize-layout, the second
  // "pointer" child may be omitted.
  if (!m_start)
    m_start =
        m_backend
            .GetChildAtNamePath({"buf", "inner", "ptr", "pointer"})
        .get();


  m_len = m_backend.GetChildMemberWithName("len").get();

  m_element_type =
      m_backend.GetCompilerType().GetNonReferenceType().GetTypeTemplateArgument(
          0);

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

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::RustVectorSyntheticFrontEnd::GetIndexOfChildWithName(
    llvm::StringRef name) {
  if (!m_start || !m_len)
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  auto optional_idx = formatters::ExtractIndexFromString(name);
  if (!optional_idx) {
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }
  return *optional_idx;
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::RustVectorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  CompilerType type = valobj_sp->GetCompilerType();
  if (!type.IsValid() ||
      type.GetNonReferenceType().GetNumTemplateArguments() == 0)
    return nullptr;
  return new RustVectorSyntheticFrontEnd(valobj_sp);
}
