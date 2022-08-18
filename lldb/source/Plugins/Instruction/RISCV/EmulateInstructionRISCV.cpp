//===-- EmulateInstructionRISCV.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "EmulateInstructionRISCV.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"
#include "Plugins/Process/Utility/lldb-riscv-register-enums.h"

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

LLDB_PLUGIN_DEFINE_ADV(EmulateInstructionRISCV, InstructionRISCV)

namespace lldb_private {

// Masks for detecting instructions types. According to riscv-spec Chap 26.
constexpr uint32_t I_MASK = 0b111000001111111;
constexpr uint32_t J_MASK = 0b000000001111111;
// no funct3 in the b-mask because the logic executing B<CMP> is quite similar.
constexpr uint32_t B_MASK = 0b000000001111111;

// The funct3 is the type of compare in B<CMP> instructions.
// funct3 means "3-bits function selector", which RISC-V ISA uses as minor
// opcode. It reuses the major opcode encoding space.
constexpr uint32_t BEQ = 0b000;
constexpr uint32_t BNE = 0b001;
constexpr uint32_t BLT = 0b100;
constexpr uint32_t BGE = 0b101;
constexpr uint32_t BLTU = 0b110;
constexpr uint32_t BGEU = 0b111;

constexpr uint32_t DecodeRD(uint32_t inst) { return (inst & 0xF80) >> 7; }
constexpr uint32_t DecodeRS1(uint32_t inst) { return (inst & 0xF8000) >> 15; }
constexpr uint32_t DecodeRS2(uint32_t inst) { return (inst & 0x1F00000) >> 20; }
constexpr uint32_t DecodeFunct3(uint32_t inst) { return (inst & 0x7000) >> 12; }

constexpr int32_t SignExt(uint32_t imm) { return int32_t(imm); }

constexpr uint32_t DecodeJImm(uint32_t inst) {
  return (uint64_t(int64_t(int32_t(inst & 0x80000000)) >> 11)) // imm[20]
         | (inst & 0xff000)                                    // imm[19:12]
         | ((inst >> 9) & 0x800)                               // imm[11]
         | ((inst >> 20) & 0x7fe);                             // imm[10:1]
}

constexpr uint32_t DecodeIImm(uint32_t inst) {
  return int64_t(int32_t(inst)) >> 20; // imm[11:0]
}

constexpr uint32_t DecodeBImm(uint32_t inst) {
  return (uint64_t(int64_t(int32_t(inst & 0x80000000)) >> 19)) // imm[12]
         | ((inst & 0x80) << 4)                                // imm[11]
         | ((inst >> 20) & 0x7e0)                              // imm[10:5]
         | ((inst >> 7) & 0x1e);                               // imm[4:1]
}

static uint32_t GPREncodingToLLDB(uint32_t reg_encode) {
  if (reg_encode == 0)
    return gpr_x0_riscv;
  if (reg_encode >= 1 && reg_encode <= 31)
    return gpr_x1_riscv + reg_encode - 1;
  return LLDB_INVALID_REGNUM;
}

static bool ReadRegister(EmulateInstructionRISCV *emulator, uint32_t reg_encode,
                         RegisterValue &value) {
  uint32_t lldb_reg = GPREncodingToLLDB(reg_encode);
  return emulator->ReadRegister(eRegisterKindLLDB, lldb_reg, value);
}

static bool WriteRegister(EmulateInstructionRISCV *emulator,
                          uint32_t reg_encode, const RegisterValue &value) {
  uint32_t lldb_reg = GPREncodingToLLDB(reg_encode);
  EmulateInstruction::Context ctx;
  ctx.type = EmulateInstruction::eContextRegisterStore;
  ctx.SetNoArgs();
  return emulator->WriteRegister(ctx, eRegisterKindLLDB, lldb_reg, value);
}

static bool ExecJAL(EmulateInstructionRISCV *emulator, uint32_t inst, bool) {
  bool success = false;
  int64_t offset = SignExt(DecodeJImm(inst));
  int64_t pc = emulator->ReadPC(&success);
  return success && emulator->WritePC(pc + offset) &&
         WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(uint64_t(pc + 4)));
}

static bool ExecJALR(EmulateInstructionRISCV *emulator, uint32_t inst, bool) {
  int64_t offset = SignExt(DecodeIImm(inst));
  RegisterValue value;
  if (!ReadRegister(emulator, DecodeRS1(inst), value))
    return false;
  bool success = false;
  int64_t pc = emulator->ReadPC(&success);
  int64_t rs1 = int64_t(value.GetAsUInt64());
  // JALR clears the bottom bit. According to riscv-spec:
  // "The JALR instruction now clears the lowest bit of the calculated target
  // address, to simplify hardware and to allow auxiliary information to be
  // stored in function pointers."
  return emulator->WritePC((rs1 + offset) & ~1) &&
         WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(uint64_t(pc + 4)));
}

static bool CompareB(uint64_t rs1, uint64_t rs2, uint32_t funct3) {
  switch (funct3) {
  case BEQ:
    return rs1 == rs2;
  case BNE:
    return rs1 != rs2;
  case BLT:
    return int64_t(rs1) < int64_t(rs2);
  case BGE:
    return int64_t(rs1) >= int64_t(rs2);
  case BLTU:
    return rs1 < rs2;
  case BGEU:
    return rs1 >= rs2;
  default:
    llvm_unreachable("unexpected funct3");
  }
}

static bool ExecB(EmulateInstructionRISCV *emulator, uint32_t inst,
                  bool ignore_cond) {
  bool success = false;
  uint64_t pc = emulator->ReadPC(&success);
  if (!success)
    return false;

  uint64_t offset = SignExt(DecodeBImm(inst));
  uint64_t target = pc + offset;
  if (ignore_cond)
    return emulator->WritePC(target);

  RegisterValue value1;
  RegisterValue value2;
  if (!ReadRegister(emulator, DecodeRS1(inst), value1) ||
      !ReadRegister(emulator, DecodeRS2(inst), value2))
    return false;

  uint32_t funct3 = DecodeFunct3(inst);
  if (CompareB(value1.GetAsUInt64(), value2.GetAsUInt64(), funct3))
    return emulator->WritePC(target);

  return true;
}

struct InstrPattern {
  const char *name;
  /// Bit mask to check the type of a instruction (B-Type, I-Type, J-Type, etc.)
  uint32_t type_mask;
  /// Characteristic value after bitwise-and with type_mask.
  uint32_t eigen;
  bool (*exec)(EmulateInstructionRISCV *emulator, uint32_t inst,
               bool ignore_cond);
};

static InstrPattern PATTERNS[] = {
    {"JAL", J_MASK, 0b1101111, ExecJAL},
    {"JALR", I_MASK, 0b000000001100111, ExecJALR},
    {"B<CMP>", B_MASK, 0b1100011, ExecB},
    // TODO: {LR/SC}.{W/D} and ECALL
};

/// This function only determines the next instruction address for software
/// sigle stepping by emulating branching instructions including:
/// - from Base Instruction Set  : JAL, JALR, B<CMP>, ECALL
/// - from Atomic Instruction Set: LR -> BNE -> SC -> BNE
/// We will get rid of this tedious code when the riscv debug spec is ratified.
bool EmulateInstructionRISCV::DecodeAndExecute(uint32_t inst,
                                               bool ignore_cond) {
  Log *log = GetLog(LLDBLog::Process | LLDBLog::Breakpoints);
  for (const InstrPattern &pat : PATTERNS) {
    if ((inst & pat.type_mask) == pat.eigen) {
      LLDB_LOGF(log, "EmulateInstructionRISCV::%s: inst(%x) was decoded to %s",
                __FUNCTION__, inst, pat.name);
      return pat.exec(this, inst, ignore_cond);
    }
  }

  LLDB_LOGF(log,
            "EmulateInstructionRISCV::%s: inst(0x%x) does not branch: "
            "no need to calculate the next pc address which is trivial.",
            __FUNCTION__, inst);
  return true;
}

bool EmulateInstructionRISCV::EvaluateInstruction(uint32_t options) {
  uint32_t inst_size = m_opcode.GetByteSize();
  uint32_t inst = m_opcode.GetOpcode32();
  bool increase_pc = options & eEmulateInstructionOptionAutoAdvancePC;
  bool ignore_cond = options & eEmulateInstructionOptionIgnoreConditions;
  bool success = false;

  lldb::addr_t old_pc = 0;
  if (increase_pc) {
    old_pc = ReadPC(&success);
    if (!success)
      return false;
  }

  if (inst_size == 2) {
    // TODO: execute RVC
    return false;
  }

  success = DecodeAndExecute(inst, ignore_cond);
  if (!success)
    return false;

  if (increase_pc) {
    lldb::addr_t new_pc = ReadPC(&success);
    if (!success)
      return false;

    if (new_pc == old_pc) {
      if (!WritePC(old_pc + inst_size))
        return false;
    }
  }
  return true;
}

bool EmulateInstructionRISCV::ReadInstruction() {
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
  uint16_t try_rvc = (uint16_t)(inst & 0x0000ffff);
  // check whether the compressed encode could be valid
  uint16_t mask = try_rvc & 0b11;
  if (try_rvc != 0 && mask != 3) {
    m_opcode.SetOpcode16(try_rvc, GetByteOrder());
  } else {
    m_opcode.SetOpcode32(inst, GetByteOrder());
  }

  return true;
}

lldb::addr_t EmulateInstructionRISCV::ReadPC(bool *success) {
  return ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC,
                              LLDB_INVALID_ADDRESS, success);
}

bool EmulateInstructionRISCV::WritePC(lldb::addr_t pc) {
  EmulateInstruction::Context ctx;
  ctx.type = eContextAdvancePC;
  ctx.SetNoArgs();
  return WriteRegisterUnsigned(ctx, eRegisterKindGeneric,
                               LLDB_REGNUM_GENERIC_PC, pc);
}

bool EmulateInstructionRISCV::GetRegisterInfo(lldb::RegisterKind reg_kind,
                                              uint32_t reg_index,
                                              RegisterInfo &reg_info) {
  if (reg_kind == eRegisterKindGeneric) {
    switch (reg_index) {
    case LLDB_REGNUM_GENERIC_PC:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_pc_riscv;
      break;
    case LLDB_REGNUM_GENERIC_SP:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_sp_riscv;
      break;
    case LLDB_REGNUM_GENERIC_FP:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_fp_riscv;
      break;
    case LLDB_REGNUM_GENERIC_RA:
      reg_kind = eRegisterKindLLDB;
      reg_index = gpr_ra_riscv;
      break;
    // We may handle LLDB_REGNUM_GENERIC_ARGx when more instructions are
    // supported.
    default:
      llvm_unreachable("unsupported register");
    }
  }

  const RegisterInfo *array =
      RegisterInfoPOSIX_riscv64::GetRegisterInfoPtr(m_arch);
  const uint32_t length =
      RegisterInfoPOSIX_riscv64::GetRegisterInfoCount(m_arch);

  if (reg_index >= length || reg_kind != eRegisterKindLLDB)
    return false;

  reg_info = array[reg_index];
  return true;
}

bool EmulateInstructionRISCV::SetTargetTriple(const ArchSpec &arch) {
  return SupportsThisArch(arch);
}

bool EmulateInstructionRISCV::TestEmulation(Stream *out_stream, ArchSpec &arch,
                                            OptionValueDictionary *test_data) {
  return false;
}

void EmulateInstructionRISCV::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void EmulateInstructionRISCV::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::EmulateInstruction *
EmulateInstructionRISCV::CreateInstance(const ArchSpec &arch,
                                        InstructionType inst_type) {
  if (EmulateInstructionRISCV::SupportsThisInstructionType(inst_type) &&
      SupportsThisArch(arch)) {
    return new EmulateInstructionRISCV(arch);
  }

  return nullptr;
}

bool EmulateInstructionRISCV::SupportsThisArch(const ArchSpec &arch) {
  return arch.GetTriple().isRISCV();
}

} // namespace lldb_private
