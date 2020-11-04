//===-- ABI.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ABI.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/TargetRegistry.h"
#include <cctype>

using namespace lldb;
using namespace lldb_private;

ABISP
ABI::FindPlugin(lldb::ProcessSP process_sp, const ArchSpec &arch) {
  ABISP abi_sp;
  ABICreateInstance create_callback;

  for (uint32_t idx = 0;
       (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) !=
       nullptr;
       ++idx) {
    abi_sp = create_callback(process_sp, arch);

    if (abi_sp)
      return abi_sp;
  }
  abi_sp.reset();
  return abi_sp;
}

ABI::~ABI() = default;

bool RegInfoBasedABI::GetRegisterInfoByName(llvm::StringRef name,
                                            RegisterInfo &info) {
  uint32_t count = 0;
  const RegisterInfo *register_info_array = GetRegisterInfoArray(count);
  if (register_info_array) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
      const char *reg_name = register_info_array[i].name;
      if (reg_name == name) {
        info = register_info_array[i];
        return true;
      }
    }
    for (i = 0; i < count; ++i) {
      const char *reg_alt_name = register_info_array[i].alt_name;
      if (reg_alt_name == name) {
        info = register_info_array[i];
        return true;
      }
    }
  }
  return false;
}

ValueObjectSP ABI::GetReturnValueObject(Thread &thread, CompilerType &ast_type,
                                        bool persistent) const {
  if (!ast_type.IsValid())
    return ValueObjectSP();

  ValueObjectSP return_valobj_sp;

  return_valobj_sp = GetReturnValueObjectImpl(thread, ast_type);
  if (!return_valobj_sp)
    return return_valobj_sp;

  // Now turn this into a persistent variable.
  // FIXME: This code is duplicated from Target::EvaluateExpression, and it is
  // used in similar form in a couple
  // of other places.  Figure out the correct Create function to do all this
  // work.

  if (persistent) {
    Target &target = *thread.CalculateTarget();
    PersistentExpressionState *persistent_expression_state =
        target.GetPersistentExpressionStateForLanguage(
            ast_type.GetMinimumLanguage());

    if (!persistent_expression_state)
      return {};

    ConstString persistent_variable_name =
        persistent_expression_state->GetNextPersistentVariableName();

    lldb::ValueObjectSP const_valobj_sp;

    // Check in case our value is already a constant value
    if (return_valobj_sp->GetIsConstant()) {
      const_valobj_sp = return_valobj_sp;
      const_valobj_sp->SetName(persistent_variable_name);
    } else
      const_valobj_sp =
          return_valobj_sp->CreateConstantValue(persistent_variable_name);

    lldb::ValueObjectSP live_valobj_sp = return_valobj_sp;

    return_valobj_sp = const_valobj_sp;

    ExpressionVariableSP expr_variable_sp(
        persistent_expression_state->CreatePersistentVariable(
            return_valobj_sp));

    assert(expr_variable_sp);

    // Set flags and live data as appropriate

    const Value &result_value = live_valobj_sp->GetValue();

    switch (result_value.GetValueType()) {
    case Value::eValueTypeHostAddress:
    case Value::eValueTypeFileAddress:
      // we don't do anything with these for now
      break;
    case Value::eValueTypeScalar:
      expr_variable_sp->m_flags |=
          ExpressionVariable::EVIsFreezeDried;
      expr_variable_sp->m_flags |=
          ExpressionVariable::EVIsLLDBAllocated;
      expr_variable_sp->m_flags |=
          ExpressionVariable::EVNeedsAllocation;
      break;
    case Value::eValueTypeLoadAddress:
      expr_variable_sp->m_live_sp = live_valobj_sp;
      expr_variable_sp->m_flags |=
          ExpressionVariable::EVIsProgramReference;
      break;
    }

    return_valobj_sp = expr_variable_sp->GetValueObject();
  }
  return return_valobj_sp;
}

ValueObjectSP ABI::GetReturnValueObject(Thread &thread, llvm::Type &ast_type,
                                        bool persistent) const {
  ValueObjectSP return_valobj_sp;
  return_valobj_sp = GetReturnValueObjectImpl(thread, ast_type);
  return return_valobj_sp;
}

// specialized to work with llvm IR types
//
// for now we will specify a default implementation so that we don't need to
// modify other ABIs
lldb::ValueObjectSP ABI::GetReturnValueObjectImpl(Thread &thread,
                                                  llvm::Type &ir_type) const {
  ValueObjectSP return_valobj_sp;

  /* this is a dummy and will only be called if an ABI does not override this */

  return return_valobj_sp;
}

bool ABI::PrepareTrivialCall(Thread &thread, lldb::addr_t sp,
                             lldb::addr_t functionAddress,
                             lldb::addr_t returnAddress, llvm::Type &returntype,
                             llvm::ArrayRef<ABI::CallArgument> args) const {
  // dummy prepare trivial call
  llvm_unreachable("Should never get here!");
}

bool ABI::GetFallbackRegisterLocation(
    const RegisterInfo *reg_info,
    UnwindPlan::Row::RegisterLocation &unwind_regloc) {
  // Did the UnwindPlan fail to give us the caller's stack pointer? The stack
  // pointer is defined to be the same as THIS frame's CFA, so return the CFA
  // value as the caller's stack pointer.  This is true on x86-32/x86-64 at
  // least.
  if (reg_info->kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_SP) {
    unwind_regloc.SetIsCFAPlusOffset(0);
    return true;
  }

  // If a volatile register is being requested, we don't want to forward the
  // next frame's register contents up the stack -- the register is not
  // retrievable at this frame.
  if (RegisterIsVolatile(reg_info)) {
    unwind_regloc.SetUndefined();
    return true;
  }

  return false;
}

std::unique_ptr<llvm::MCRegisterInfo> ABI::MakeMCRegisterInfo(const ArchSpec &arch) {
  std::string triple = arch.GetTriple().getTriple();
  std::string lookup_error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, lookup_error);
  if (!target) {
    LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS),
             "Failed to create an llvm target for {0}: {1}", triple,
             lookup_error);
    return nullptr;
  }
  std::unique_ptr<llvm::MCRegisterInfo> info_up(
      target->createMCRegInfo(triple));
  assert(info_up);
  return info_up;
}

void RegInfoBasedABI::AugmentRegisterInfo(RegisterInfo &info) {
  if (info.kinds[eRegisterKindEHFrame] != LLDB_INVALID_REGNUM &&
      info.kinds[eRegisterKindDWARF] != LLDB_INVALID_REGNUM)
    return;

  RegisterInfo abi_info;
  if (!GetRegisterInfoByName(info.name, abi_info))
    return;

  if (info.kinds[eRegisterKindEHFrame] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindEHFrame] = abi_info.kinds[eRegisterKindEHFrame];
  if (info.kinds[eRegisterKindDWARF] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindDWARF] = abi_info.kinds[eRegisterKindDWARF];
  if (info.kinds[eRegisterKindGeneric] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindGeneric] = abi_info.kinds[eRegisterKindGeneric];
}

void MCBasedABI::AugmentRegisterInfo(RegisterInfo &info) {
  uint32_t eh, dwarf;
  std::tie(eh, dwarf) = GetEHAndDWARFNums(info.name);

  if (info.kinds[eRegisterKindEHFrame] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindEHFrame] = eh;
  if (info.kinds[eRegisterKindDWARF] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindDWARF] = dwarf;
  if (info.kinds[eRegisterKindGeneric] == LLDB_INVALID_REGNUM)
    info.kinds[eRegisterKindGeneric] = GetGenericNum(info.name);
}

std::pair<uint32_t, uint32_t>
MCBasedABI::GetEHAndDWARFNums(llvm::StringRef name) {
  std::string mc_name = GetMCName(name.str());
  for (char &c : mc_name)
    c = std::toupper(c);
  int eh = -1;
  int dwarf = -1;
  for (unsigned reg = 0; reg < m_mc_register_info_up->getNumRegs(); ++reg) {
    if (m_mc_register_info_up->getName(reg) == mc_name) {
      eh = m_mc_register_info_up->getDwarfRegNum(reg, /*isEH=*/true);
      dwarf = m_mc_register_info_up->getDwarfRegNum(reg, /*isEH=*/false);
      break;
    }
  }
  return std::pair<uint32_t, uint32_t>(eh == -1 ? LLDB_INVALID_REGNUM : eh,
                                       dwarf == -1 ? LLDB_INVALID_REGNUM
                                                   : dwarf);
}

void MCBasedABI::MapRegisterName(std::string &name, llvm::StringRef from_prefix,
                                 llvm::StringRef to_prefix) {
  llvm::StringRef name_ref = name;
  if (!name_ref.consume_front(from_prefix))
    return;
  uint64_t _;
  if (name_ref.empty() || to_integer(name_ref, _, 10))
    name = (to_prefix + name_ref).str();
}
