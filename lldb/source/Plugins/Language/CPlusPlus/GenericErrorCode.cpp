//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generic.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;

namespace {

ValueObjectSP GetValue(ValueObject &valobj) {
  ValueObjectSP value_sp = valobj.GetChildMemberWithName("_Myval");
  if (!value_sp)
    value_sp = valobj.GetChildMemberWithName("_M_value");
  return value_sp;
}

ValueObjectSP GetCategory(ValueObject &valobj) {
  ValueObjectSP category_sp = valobj.GetChildMemberWithName("_Mycat");
  if (!category_sp)
    category_sp = valobj.GetChildMemberWithName("_M_cat");
  return category_sp;
}

class GenericErrorCodeFrontend : public SyntheticChildrenFrontEnd {
public:
  explicit GenericErrorCodeFrontend(ValueObject &valobj)
      : SyntheticChildrenFrontEnd(valobj) {
    Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (name == "Category" && m_category)
      return 0;
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_category ? 1U : 0U;
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (idx != 0 || !m_category)
      return {};
    return m_category->Clone(ConstString("Category"));
  }

  lldb::ChildCacheState Update() override {
    m_category = GetCategory(m_backend).get();
    return lldb::ChildCacheState::eRefetch;
  }

private:
  // Children derived from the backend share its ClusterManager. Keeping a
  // shared pointer here would create an ownership cycle.
  ValueObject *m_category = nullptr;
};

} // namespace

bool lldb_private::formatters::GenericErrorCodeSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  ValueObjectSP valobj_sp = valobj.GetNonSyntheticValue();
  if (!valobj_sp)
    return false;

  ValueObjectSP value_sp = GetValue(*valobj_sp);
  if (!value_sp)
    return false;

  const char *value = value_sp->GetValueAsCString();
  if (!value)
    return false;

  stream.Printf("value=%s", value);
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::GenericErrorCodeSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new GenericErrorCodeFrontend(*valobj_sp);
}
