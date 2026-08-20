//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/ErrorExtras.h"
#include <cinttypes>
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {
class MsvcStlValarraySyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlValarraySyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {
    if (valobj_sp)
      Update();
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_start || m_element_size == 0)
      return 0;
    return m_count;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_start || idx >= m_count)
      return {};

    uint64_t offset = m_start->GetValueAsUnsigned(0) +
                      static_cast<uint64_t>(idx) * m_element_size;
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateChildValueObjectFromAddress(name.GetString(), offset,
                                             m_backend.GetExecutionContextRef(),
                                             m_element_type);
  }

  lldb::ChildCacheState Update() override {
    m_start = nullptr;
    m_count = 0;
    m_element_size = 0;

    ValueObjectSP start = m_backend.GetChildMemberWithName("_Myptr");
    ValueObjectSP size_sp = m_backend.GetChildMemberWithName("_Mysize");
    if (!start || !size_sp)
      return ChildCacheState::eRefetch;

    // Prefer the pointer's pointee type: template arguments are missing on
    // references unless the frontend asked for a dereference, and PDB may
    // omit them entirely.
    m_element_type = start->GetCompilerType().GetPointeeType();
    if (!m_element_type.IsValid()) {
      CompilerType type = m_backend.GetCompilerType().GetNonReferenceType();
      m_element_type = type.GetTypeTemplateArgument(0);
    }
    if (!m_element_type.IsValid())
      return ChildCacheState::eRefetch;

    if (std::optional<uint64_t> size =
            llvm::expectedToOptional(m_element_type.GetByteSize(nullptr)))
      m_element_size = *size;

    if (m_element_size == 0)
      return ChildCacheState::eRefetch;

    m_start = start.get();
    m_count = size_sp->GetValueAsUnsigned(0);
    return ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_start)
      return llvm::createStringErrorV("type has no child named '{0}'", name);
    auto optional_idx = ExtractIndexFromString(name.GetCString());
    if (!optional_idx)
      return llvm::createStringErrorV("type has no child named '{0}'", name);
    return *optional_idx;
  }

private:
  /// A non-owning pointer to valarray's _Myptr member.
  ValueObject *m_start = nullptr;
  CompilerType m_element_type;
  uint32_t m_element_size = 0;
  uint64_t m_count = 0;
};
} // namespace

bool formatters::IsMsvcStlValarray(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue())
    return valobj_sp->GetChildMemberWithName("_Myptr") != nullptr &&
           valobj_sp->GetChildMemberWithName("_Mysize") != nullptr;
  return false;
}

SyntheticChildrenFrontEnd *formatters::MsvcStlValarraySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new MsvcStlValarraySyntheticFrontEnd(valobj_sp);
}
