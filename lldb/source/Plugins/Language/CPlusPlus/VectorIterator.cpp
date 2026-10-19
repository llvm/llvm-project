//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VectorIterator.h"

#include "Plugins/Language/CPlusPlus/LibCxx.h"

#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

VectorIteratorSyntheticFrontEnd::VectorIteratorSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp, llvm::ArrayRef<ConstString> item_names)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_exe_ctx_ref(),
      m_item_names(item_names), m_item_sp() {
  if (valobj_sp)
    Update();
}

lldb::ChildCacheState VectorIteratorSyntheticFrontEnd::Update() {
  m_item_sp.reset();

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP item_ptr =
      formatters::GetChildMemberWithName(*valobj_sp, m_item_names);
  if (!item_ptr)
    return lldb::ChildCacheState::eRefetch;
  if (item_ptr->GetValueAsUnsigned(0) == 0)
    return lldb::ChildCacheState::eRefetch;
  Status err;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
  m_item_sp = CreateChildValueObjectFromAddress(
      "item", item_ptr->GetValueAsUnsigned(0), m_exe_ctx_ref,
      item_ptr->GetCompilerType().GetPointeeType());
  if (err.Fail())
    m_item_sp.reset();
  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<uint32_t>
VectorIteratorSyntheticFrontEnd::CalculateNumChildren() {
  return 1;
}

lldb::ValueObjectSP
VectorIteratorSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx == 0)
    return m_item_sp;
  return lldb::ValueObjectSP();
}

llvm::Expected<size_t>
VectorIteratorSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (name == "item")
    return 0;
  return llvm::createStringErrorV("type has no child named '{0}'", name);
}
