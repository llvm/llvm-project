//===---EmulateInstructionLoongArch.cpp------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "EmulateInstructionLoongArch.h"
#include "Plugins/Process/Utility/InstructionUtils.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_loongarch64.h"
#include "Plugins/Process/Utility/lldb-loongarch-register-enums.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/OptionValueArray.h"
#include "lldb/Interpreter/OptionValueDictionary.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(EmulateInstructionLoongArch, InstructionLoongArch)

namespace lldb_private {

EmulateInstructionLoongArch::Opcode *
EmulateInstructionLoongArch::GetOpcodeForInstruction(uint32_t inst) {
  // TODO: Add the mask of jump instruction.
  static EmulateInstructionLoongArch::Opcode g_opcodes[] = {
      {0x00000000, 0x00000000, &EmulateInstructionLoongArch::EmulateNonJMP,
       "NonJMP"}};
  static const size_t num_loongarch_opcodes = std::size(g_opcodes);

  for (size_t i = 0; i < num_loongarch_opcodes; ++i)
    if ((g_opcodes[i].mask & inst) == g_opcodes[i].value)
      return &g_opcodes[i];
  return nullptr;
}

bool EmulateInstructionLoongArch::EvaluateInstruction(uint32_t options) {
  uint32_t inst_size = m_opcode.GetByteSize();
  uint32_t inst = m_opcode.GetOpcode32();
  bool increase_pc = options & eEmulateInstructionOptionAutoAdvancePC;
  bool success = false;

  Opcode *opcode_data = GetOpcodeForInstruction(inst);
  if (!opcode_data)
    return false;

  lldb::addr_t old_pc = 0;
  if (increase_pc) {
    old_pc = ReadPC(&success);
    if (!success)
      return false;
  }

  // Call the Emulate... function.
  if (!(this->*opcode_data->callback)(inst))
    return false;

  if (increase_pc) {
    lldb::addr_t new_pc = ReadPC(&success);
    if (!success)
      return false;

    if (new_pc == old_pc && !WritePC(old_pc + inst_size))
      return false;
  }
  return true;
}

bool EmulateInstructionLoongArch::ReadInstruction() {
  bool success = false;
  m_addr = ReadPC(&success);
  if (!success) {
    m_addr = LLDB_INVALID_ADDRESS;
    return false;
  }

  Context ctx;
  ctx.type = eContextReadOpcode;
  ctx.SetNoArgs();
  uint32_t inst = (uint32_t)ReadMemoryUnsigned(ctx, m_addr, 4, 0, &success);
  m_opcode.SetOpcode32(inst, GetByteOrder());

  return true;
}

lldb::addr_t EmulateInstructionLoongArch::ReadPC(bool *success) {
  return ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC,
                              LLDB_INVALID_ADDRESS, success);
}

bool EmulateInstructionLoongArch::WritePC(lldb::addr_t pc) {
  EmulateInstruction::Context ctx;
  ctx.type = eContextAdvancePC;
  ctx.SetNoArgs();
  return WriteRegisterUnsigned(ctx, eRegisterKindGeneric,
                               LLDB_REGNUM_GENERIC_PC, pc);
}

llvm::Optional<RegisterInfo>
EmulateInstructionLoongArch::GetRegisterInfo(lldb::RegisterKind reg_kind,
                                             uint32_t reg_index) {
  if (reg_kind == eRegisterKindGeneric) {
    switch (reg_index) {
    case LLDB_REGNUM_GENERIC_PC:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_pc_loongarch;
      break;
    case LLDB_REGNUM_GENERIC_SP:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_sp_loongarch;
      break;
    case LLDB_REGNUM_GENERIC_FP:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_fp_loongarch;
      break;
    case LLDB_REGNUM_GENERIC_RA:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_ra_loongarch;
      break;
    // We may handle LLDB_REGNUM_GENERIC_ARGx when more instructions are
    // supported.
    default:
      llvm_unreachable("unsupported register");
    }
  }

  const RegisterInfo *array =
      RegisterInfoPOSIX_loongarch64::GetRegisterInfoPtr(m_arch);
  const uint32_t length =
      RegisterInfoPOSIX_loongarch64::GetRegisterInfoCount(m_arch);

  if (reg_index >= length || reg_kind != eRegisterKindLLDB)
    return {};
  return array[reg_index];
}

bool EmulateInstructionLoongArch::SetTargetTriple(const ArchSpec &arch) {
  return SupportsThisArch(arch);
}

bool EmulateInstructionLoongArch::TestEmulation(
    Stream *out_stream, ArchSpec &arch, OptionValueDictionary *test_data) {
  return false;
}

void EmulateInstructionLoongArch::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void EmulateInstructionLoongArch::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::EmulateInstruction *
EmulateInstructionLoongArch::CreateInstance(const ArchSpec &arch,
                                            InstructionType inst_type) {
  if (EmulateInstructionLoongArch::SupportsThisInstructionType(inst_type) &&
      SupportsThisArch(arch))
    return new EmulateInstructionLoongArch(arch);
  return nullptr;
}

bool EmulateInstructionLoongArch::SupportsThisArch(const ArchSpec &arch) {
  return arch.GetTriple().isLoongArch();
}

bool EmulateInstructionLoongArch::EmulateNonJMP(uint32_t inst) { return false; }

} // namespace lldb_private
