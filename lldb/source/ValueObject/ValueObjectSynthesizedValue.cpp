//===-- ValueObjectSynthesizedValue.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/ValueObjectSynthesizedValue.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

lldb::ValueObjectSP ValueObjectSynthesizedValue::Create(ValueObject &valobj,
                                                        lldb::ValueType type) {
  ExecutionContext exe_ctx(valobj.GetExecutionContextRef());
  auto manager_sp = ValueObjectManager::Create();
  return (new ValueObjectSynthesizedValue(
              exe_ctx.GetBestExecutionContextScope(), *manager_sp,
              valobj.GetSP(), type))
      ->GetSP();
}

ValueObjectSynthesizedValue::ValueObjectSynthesizedValue(
    ExecutionContextScope *exe_scope, ValueObjectManager &manager,
    const lldb::ValueObjectSP &valobj_sp, lldb::ValueType type)
    : ValueObject(exe_scope, manager), m_valobj_sp(valobj_sp), m_type(type) {
  SetName(valobj_sp->GetName());
}

ValueObjectSynthesizedValue::~ValueObjectSynthesizedValue() = default;

llvm::Expected<uint64_t> ValueObjectSynthesizedValue::GetByteSize() {
  return m_valobj_sp->GetByteSize();
}

llvm::Expected<uint32_t>
ValueObjectSynthesizedValue::CalculateNumChildren(uint32_t max) {
  return m_valobj_sp->GetNumChildren(max);
}

bool ValueObjectSynthesizedValue::IsInScope() {
  return m_valobj_sp->IsInScope();
}

CompilerType ValueObjectSynthesizedValue::GetCompilerTypeImpl() {
  return m_valobj_sp->GetCompilerType();
}

bool ValueObjectSynthesizedValue::UpdateValue() {
  SetValueIsValid(false);
  m_error.Clear();

  if (!m_valobj_sp->UpdateValueIfNeeded(false)) {
    if (m_error.Success() && m_valobj_sp->GetError().Fail())
      m_error = m_valobj_sp->GetError().Clone();
    return false;
  }

  m_update_point.SetUpdated();
  m_value = m_valobj_sp->GetValue();
  SetAddressTypeOfChildren(m_valobj_sp->GetAddressTypeOfChildren());
  ExecutionContext exe_ctx(GetExecutionContextRef());
  m_error = m_value.GetValueAsData(&exe_ctx, m_data, GetModule().get());
  SetValueDidChange(m_valobj_sp->GetValueDidChange());
  SetValueIsValid(m_error.Success());
  return m_error.Success();
}
