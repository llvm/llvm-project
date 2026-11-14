//===-- ValueObjectCast.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/ValueObjectCast.h"

#include "lldb/Core/Value.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObject.h"
#include <optional>

using namespace lldb_private;

lldb::ValueObjectSP ValueObjectCast::Create(ValueObject &parent,
                                            llvm::StringRef name,
                                            const CompilerType &cast_type) {
  ValueObjectCast *cast_valobj_ptr =
      new ValueObjectCast(parent, name, cast_type);
  return cast_valobj_ptr->GetSP();
}

ValueObjectCast::ValueObjectCast(ValueObject &parent, llvm::StringRef name,
                                 const CompilerType &cast_type)
    : ValueObject(parent), m_cast_type(cast_type) {
  SetName(name);
  m_value.SetCompilerType(cast_type);
}

ValueObjectCast::~ValueObjectCast() = default;

CompilerType ValueObjectCast::GetCompilerTypeImpl() { return m_cast_type; }

llvm::Expected<uint32_t> ValueObjectCast::CalculateNumChildren(uint32_t max) {
  ExecutionContext exe_ctx(GetExecutionContextRef());
  auto children_count = GetCompilerType().GetNumChildren(true, &exe_ctx);
  if (!children_count)
    return children_count;
  return *children_count <= max ? *children_count : max;
}

llvm::Expected<uint64_t> ValueObjectCast::GetByteSize() {
  ExecutionContext exe_ctx(GetExecutionContextRef());
  return m_value.GetValueByteSize(nullptr, &exe_ctx);
}

lldb::ValueType ValueObjectCast::GetValueType() const {
  // Let our parent answer global, local, argument, etc...
  return m_parent->GetValueType();
}

bool ValueObjectCast::UpdateValue() {
  SetValueIsValid(false);
  m_error.Clear();

  if (m_parent->UpdateValueIfNeeded(false)) {
    Value old_value(m_value);
    m_update_point.SetUpdated();
    m_value = m_parent->GetValue();
    CompilerType compiler_type(GetCompilerType());
    m_value.SetCompilerType(compiler_type);
    SetAddressTypeOfChildren(m_parent->GetAddressTypeOfChildren());
    // A cast that drops __ptrauth strips the signature bits from the value.
    if (m_parent->GetCompilerType().HasPointerAuthQualifier() &&
        !compiler_type.HasPointerAuthQualifier()) {
      if (auto stripped = m_parent->GetStrippedPointerValue(
              m_parent->GetPointerValue().address)) {
        m_value.GetScalar() = *stripped;
        m_value.SetValueType(Value::ValueType::Scalar);
      }
    }
    if (!CanProvideValue()) {
      // this value object represents an aggregate type whose children have
      // values, but this object does not. So we say we are changed if our
      // location has changed.
      SetValueDidChange(m_value.GetValueType() != old_value.GetValueType() ||
                        m_value.GetScalar() != old_value.GetScalar());
    }
    ExecutionContext exe_ctx(GetExecutionContextRef());
    m_error = m_value.GetValueAsData(&exe_ctx, m_data, GetModule().get());
    SetValueDidChange(m_parent->GetValueDidChange());
    return true;
  }

  // The dynamic value failed to get an error, pass the error along
  if (m_error.Success() && m_parent->GetError().Fail())
    m_error = m_parent->GetError().Clone();
  SetValueIsValid(false);
  return false;
}

bool ValueObjectCast::IsInScope() { return m_parent->IsInScope(); }

bool ValueObjectCast::LivesInParentStorage() {
  // A parent without storage, such as a register variable, keeps the base path.
  return UpdateValueIfNeeded(false) &&
         m_value.GetValueType() == Value::ValueType::Scalar &&
         m_parent->GetAddressOf(/*scalar_is_load_address=*/false).type !=
             eAddressTypeInvalid;
}

ValueObject::AddrAndType
ValueObjectCast::GetAddressOf(bool scalar_is_load_address) {
  if (LivesInParentStorage())
    return m_parent->GetAddressOf(scalar_is_load_address);
  return ValueObject::GetAddressOf(scalar_is_load_address);
}

const char *ValueObjectCast::GetLocationAsCString() {
  if (LivesInParentStorage())
    return m_parent->GetLocationAsCString();
  return ValueObject::GetLocationAsCString();
}

llvm::Error ValueObjectCast::CanSetValue() {
  // UpdateValue rebuilds a stripped pointer from the parent; a write is lost.
  if (m_parent->GetCompilerType().HasPointerAuthQualifier() &&
      !GetCompilerType().HasPointerAuthQualifier())
    return llvm::createStringError("value is not in a writable location");
  return ValueObject::CanSetValue();
}
