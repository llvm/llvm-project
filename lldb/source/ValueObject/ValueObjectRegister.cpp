//===-- ValueObjectRegister.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/ValueObjectRegister.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterType.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorExtras.h"

#include <cassert>
#include <memory>
#include <optional>

namespace lldb_private {
class ExecutionContextScope;
}

using namespace lldb;
using namespace lldb_private;

#pragma mark ValueObjectRegisterSet

ValueObjectSP
ValueObjectRegisterSet::Create(ExecutionContextScope *exe_scope,
                               lldb::ThreadRegisterContextSP &reg_ctx_sp,
                               uint32_t set_idx) {
  auto manager_sp = ValueObjectManager::Create();
  return (new ValueObjectRegisterSet(exe_scope, *manager_sp, reg_ctx_sp,
                                     set_idx))
      ->GetSP();
}

ValueObjectRegisterSet::ValueObjectRegisterSet(ExecutionContextScope *exe_scope,
                                               ValueObjectManager &manager,
                                               lldb::ThreadRegisterContextSP &reg_ctx,
                                               uint32_t reg_set_idx)
    : ValueObject(exe_scope, manager), m_reg_ctx_sp(reg_ctx),
      m_reg_set(nullptr), m_reg_set_idx(reg_set_idx) {
  assert(reg_ctx);
  m_description = reg_ctx->GetDescription();
  m_reg_set = m_description->GetGroupAtIndex(m_reg_set_idx);
  if (m_reg_set) {
    m_name.SetCString(m_reg_set->name.c_str());
  }
}

ValueObjectRegisterSet::~ValueObjectRegisterSet() = default;

CompilerType ValueObjectRegisterSet::GetCompilerTypeImpl() {
  return CompilerType();
}

ConstString ValueObjectRegisterSet::GetTypeName() { return ConstString(); }

ConstString ValueObjectRegisterSet::GetQualifiedTypeName() {
  return ConstString();
}

llvm::Expected<uint32_t>
ValueObjectRegisterSet::CalculateNumChildren(uint32_t max) {
  const RegisterGroup *reg_set = m_reg_ctx_sp->GetRegisterGroup(m_reg_set_idx);
  if (reg_set) {
    auto reg_count = reg_set->registers.size();
    return reg_count <= max ? reg_count : max;
  }
  return 0;
}

llvm::Expected<uint64_t> ValueObjectRegisterSet::GetByteSize() { return 0; }

bool ValueObjectRegisterSet::UpdateValue() {
  m_error.Clear();
  SetValueDidChange(false);
  ExecutionContext exe_ctx(GetExecutionContextRef());
  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame == nullptr)
    m_reg_ctx_sp.reset();
  else {
    m_reg_ctx_sp = frame->GetRegisterContext();
    if (m_reg_ctx_sp) {
      const RegisterGroup *reg_set =
          m_reg_ctx_sp->GetRegisterGroup(m_reg_set_idx);
      if (reg_set == nullptr)
        m_reg_ctx_sp.reset();
      else if (m_reg_set != reg_set) {
        SetValueDidChange(true);
        m_name.SetCString(reg_set->name.c_str());
        m_children.Clear();
      }
    }
  }
  if (m_reg_ctx_sp) {
    m_description = m_reg_ctx_sp->GetDescription();
    m_reg_set = m_description->GetGroupAtIndex(m_reg_set_idx);
    SetValueIsValid(true);
  } else {
    SetValueIsValid(false);
    m_error = Status::FromErrorString("no register context");
    m_children.Clear();
  }
  return m_error.Success();
}

ValueObject *ValueObjectRegisterSet::CreateChildAtIndex(size_t idx) {
  if (m_reg_ctx_sp && m_reg_set) {
    return new ValueObjectRegister(
        *this, m_reg_ctx_sp,
        m_reg_ctx_sp->GetDescription()->Find(m_reg_set->registers[idx]));
  }
  return nullptr;
}

bool ValueObjectRegisterSet::IsSameRegister(const RegisterDescriptor *a,
                                            const RegisterDescriptor *b) {
  return a && b && a->id == b->id;
}

std::optional<std::pair<size_t, const RegisterDescriptor *>>
ValueObjectRegisterSet::LookupChildWithName(llvm::StringRef name) {
  if (!m_reg_ctx_sp || !m_reg_set)
    return {};

  // See if the register exists at all in any set.
  const RegisterDescriptor *reg_info =
      m_reg_ctx_sp->GetRegisterByName(name);

  // See if this register is in this register set, falling back to the name as
  // written. The generic name may alias a register in another set while this
  // set has a register of that exact name. A match on the name itself always
  // wins over one on the alt name.
  std::optional<std::pair<size_t, const RegisterDescriptor *>> match;
  for (size_t i = 0; i < m_reg_set->registers.size(); ++i) {
    const RegisterDescriptor *contained_reg_info =
        m_reg_ctx_sp->GetDescription()->Find(m_reg_set->registers[i]);
    if (!contained_reg_info)
      continue;
    if (IsSameRegister(contained_reg_info, reg_info))
      return std::make_pair(i, reg_info);
    if (name.equals_insensitive(contained_reg_info->name))
      match = std::make_pair(i, contained_reg_info);
    else if (!match && name.equals_insensitive(contained_reg_info->alt_name))
      match = std::make_pair(i, contained_reg_info);
  }

  return match;
}

lldb::ValueObjectSP
ValueObjectRegisterSet::GetChildMemberWithName(llvm::StringRef name,
                                               bool can_create) {
  if (auto maybe_child = LookupChildWithName(name))
    return (new ValueObjectRegister(*this, m_reg_ctx_sp, maybe_child->second))
        ->GetSP();

  return {};
}

llvm::Expected<size_t>
ValueObjectRegisterSet::GetIndexOfChildWithName(llvm::StringRef name) {
  if (auto maybe_child = LookupChildWithName(name))
    return maybe_child->first;

  return llvm::createStringErrorV("type has no child named '{0}'", name);
}

#pragma mark -
#pragma mark ValueObjectRegister

void ValueObjectRegister::ConstructObject(const RegisterDescriptor *reg_info) {
  if (reg_info) {
    m_description = m_reg_ctx_sp->GetDescription();
    m_register_id = reg_info->id;
    if (!reg_info->name.empty())
      m_name.SetCString(reg_info->name.c_str());
    else if (!reg_info->alt_name.empty())
      m_name.SetCString(reg_info->alt_name.c_str());
  }
}

ValueObjectRegister::ValueObjectRegister(
    ValueObject &parent, lldb::ThreadRegisterContextSP &reg_ctx_sp,
    const RegisterDescriptor *reg_info)
    : ValueObject(parent), m_reg_ctx_sp(reg_ctx_sp), m_description(),
      m_reg_value(), m_type_name(), m_compiler_type() {
  assert(reg_ctx_sp.get());
  ConstructObject(reg_info);
}

ValueObjectSP ValueObjectRegister::Create(ExecutionContextScope *exe_scope,
                                          lldb::ThreadRegisterContextSP &reg_ctx_sp,
                            const RegisterDescriptor *reg_info) {
  auto manager_sp = ValueObjectManager::Create();
  return (new ValueObjectRegister(exe_scope, *manager_sp, reg_ctx_sp, reg_info))
      ->GetSP();
}

ValueObjectRegister::ValueObjectRegister(ExecutionContextScope *exe_scope,
                                         ValueObjectManager &manager,
                                         lldb::ThreadRegisterContextSP &reg_ctx,
                                         const RegisterDescriptor *reg_info)
    : ValueObject(exe_scope, manager), m_reg_ctx_sp(reg_ctx), m_description(),
      m_reg_value(), m_type_name(), m_compiler_type() {
  assert(reg_ctx);
  ConstructObject(reg_info);
}

ValueObjectRegister::~ValueObjectRegister() = default;

CompilerType ValueObjectRegister::GetCompilerTypeImpl() {
  if (m_compiler_type.IsValid())
    return m_compiler_type;

  ExecutionContext exe_ctx(GetExecutionContextRef());
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return {};

  if (llvm::isa_and_present<RegisterTypeBuiltin, RegisterTypeComposite>(
          GetDescriptor().type)) {
    m_compiler_type = target->GetRegisterType(GetDescriptor());
    if (m_compiler_type.IsValid())
      return m_compiler_type;
  }

  auto *exe_module = target->GetExecutableModulePointer();
  if (!exe_module)
    return {};
  auto type_system_or_err =
      exe_module->GetTypeSystemForLanguage(eLanguageTypeC);
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), std::move(err),
                   "Unable to get CompilerType from TypeSystem: {0}");
  } else {
    if (auto ts = *type_system_or_err)
      m_compiler_type = ts->GetBuiltinTypeForEncodingAndBitSize(
          GetDescriptor().encoding, GetDescriptor().byte_size * 8);
  }
  return m_compiler_type;
}

ConstString ValueObjectRegister::GetTypeName() {
  if (m_type_name.IsEmpty())
    m_type_name = GetCompilerType().GetTypeName();
  return m_type_name;
}

llvm::Expected<uint32_t>
ValueObjectRegister::CalculateNumChildren(uint32_t max) {
  ExecutionContext exe_ctx(GetExecutionContextRef());
  auto children_count = GetCompilerType().GetNumChildren(true, &exe_ctx);
  if (!children_count)
    return children_count;
  return *children_count <= max ? *children_count : max;
}

llvm::Expected<uint64_t> ValueObjectRegister::GetByteSize() {
  return GetDescriptor().byte_size;
}

bool ValueObjectRegister::UpdateValue() {
  if (m_reg_ctx_sp) {
    auto description = m_reg_ctx_sp->GetDescription();
    const auto *reg = description ? description->Find(m_register_id) : nullptr;
    const auto &previous = GetDescriptor();
    if (description && !previous.name.empty() &&
        (!reg || reg->name != previous.name))
      reg = description->Find(previous.name);

    if (!reg) {
      m_error = Status::FromErrorString(
          "register is unavailable in this configuration");
      return false;
    }
    if (reg->type != GetDescriptor().type ||
        reg->byte_size != GetDescriptor().byte_size)
      m_compiler_type.Clear();
    m_register_id = reg->id;
    m_description = std::move(description);
  }

  m_error.Clear();
  ExecutionContext exe_ctx(GetExecutionContextRef());
  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame == nullptr) {
    m_reg_ctx_sp.reset();
    m_reg_value.Clear();
  }

  if (m_reg_ctx_sp) {
    RegisterValue m_old_reg_value(m_reg_value);
    if (!llvm::errorToBool(m_reg_ctx_sp->ReadRegister(GetDescriptor().id)
                               .moveInto(m_reg_value))) {
      Target *target = exe_ctx.GetTargetPtr();
      const bool has_structured_type =
          llvm::isa_and_present<RegisterTypeComposite>(GetDescriptor().type);
      // Scalar registers remain in host byte order, while CompilerType
      // children need target-order bytes to interpret structured layouts.
      const bool got_data =
          has_structured_type && target
              ? m_reg_value.GetData(m_data, GetDescriptor().byte_size,
                                    target->GetArchitecture().GetByteOrder())
              : m_reg_value.GetData(m_data);
      if (got_data) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process)
          m_data.SetAddressByteSize(process->GetAddressByteSize());
        m_value.SetRegister(m_reg_ctx_sp->GetDescription(), GetDescriptor().id);
        m_value.SetValueType(Value::ValueType::HostAddress);
        m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
        SetValueIsValid(true);
        SetValueDidChange(!(m_old_reg_value == m_reg_value));
        return true;
      }
    }
  }

  SetValueIsValid(false);
  m_error = Status::FromErrorString("no register context");
  return false;
}

bool ValueObjectRegister::SetValueFromCString(const char *value_str,
                                              Status &error) {
  // The new value will be in the m_data.  Copy that into our register value.
  error = m_reg_value.SetValueFromString(GetDescriptor().encoding,
                                         GetDescriptor().byte_size,
                                         llvm::StringRef(value_str));
  if (!error.Success())
    return false;

  if (llvm::errorToBool(
          m_reg_ctx_sp->WriteRegister(GetDescriptor().id, m_reg_value))) {
    error = Status::FromErrorString("unable to write back to register");
    return false;
  }

  SetNeedsUpdate();
  return true;
}

bool ValueObjectRegister::SetData(DataExtractor &data, Status &error) {
  error = m_reg_value.SetValueFromData(GetDescriptor().encoding,
                                       GetDescriptor().byte_size, data, 0,
                                       false, GetDescriptor().name);
  if (!error.Success())
    return false;

  if (llvm::errorToBool(
          m_reg_ctx_sp->WriteRegister(GetDescriptor().id, m_reg_value))) {
    error = Status::FromErrorString("unable to write back to register");
    return false;
  }

  SetNeedsUpdate();
  return true;
}

bool ValueObjectRegister::ResolveValue(Scalar &scalar) {
  if (UpdateValueIfNeeded(
          false)) // make sure that you are up to date before returning anything
    return m_reg_value.GetScalarValue(scalar);
  return false;
}

void ValueObjectRegister::GetExpressionPath(Stream &s,
                                            GetExpressionPathFormat epformat) {
  s.Printf("$%s", GetDescriptor().name.c_str());
}
