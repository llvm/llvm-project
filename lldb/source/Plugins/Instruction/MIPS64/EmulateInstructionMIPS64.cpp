//===-- EmulateInstructionMIPS64.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionMIPS64.h"

#include <cstdlib>
#include <optional>

#include "lldb/Core/Address.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Stream.h"
#include "llvm-c/Disassembler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/ADT/STLExtras.h"

#include "Plugins/Process/Utility/InstructionUtils.h"
#include "lldb/Register/Arch/lldb-mips64-dwarf-register-enums.h"

using namespace lldb;
using namespace lldb_private;
using namespace registers::mips64::dwarf;

LLDB_PLUGIN_DEFINE_ADV(EmulateInstructionMIPS64, InstructionMIPS64)

#define UInt(x) ((uint64_t)x)
#define integer int64_t

//
// EmulateInstructionMIPS64 implementation
//

#ifdef __mips__
extern "C" {
void LLVMInitializeMipsTargetInfo();
void LLVMInitializeMipsTarget();
void LLVMInitializeMipsAsmPrinter();
void LLVMInitializeMipsTargetMC();
void LLVMInitializeMipsDisassembler();
}
#endif

EmulateInstructionMIPS64::EmulateInstructionMIPS64(
    const lldb_private::ArchSpec &arch)
    : EmulateInstruction(arch) {
  /* Create instance of llvm::MCDisassembler */
  std::string Status;
  llvm::Triple triple = arch.GetTriple();
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, Status);

/*
 * If we fail to get the target then we haven't registered it. The
 * SystemInitializerCommon
 * does not initialize targets, MCs and disassemblers. However we need the
 * MCDisassembler
 * to decode the instructions so that the decoding complexity stays with LLVM.
 * Initialize the MIPS targets and disassemblers.
*/
#ifdef __mips__
  if (!target) {
    LLVMInitializeMipsTargetInfo();
    LLVMInitializeMipsTarget();
    LLVMInitializeMipsAsmPrinter();
    LLVMInitializeMipsTargetMC();
    LLVMInitializeMipsDisassembler();
    target = llvm::TargetRegistry::lookupTarget(triple, Status);
  }
#endif

  assert(target);

  llvm::StringRef cpu;

  switch (arch.GetCore()) {
  case ArchSpec::eCore_mips32:
  case ArchSpec::eCore_mips32el:
    cpu = "mips32";
    break;
  case ArchSpec::eCore_mips32r2:
  case ArchSpec::eCore_mips32r2el:
    cpu = "mips32r2";
    break;
  case ArchSpec::eCore_mips32r3:
  case ArchSpec::eCore_mips32r3el:
    cpu = "mips32r3";
    break;
  case ArchSpec::eCore_mips32r5:
  case ArchSpec::eCore_mips32r5el:
    cpu = "mips32r5";
    break;
  case ArchSpec::eCore_mips32r6:
  case ArchSpec::eCore_mips32r6el:
    cpu = "mips32r6";
    break;
  case ArchSpec::eCore_mips64:
  case ArchSpec::eCore_mips64el:
    cpu = "mips64";
    break;
  case ArchSpec::eCore_mips64r2:
  case ArchSpec::eCore_mips64r2el:
    cpu = "mips64r2";
    break;
  case ArchSpec::eCore_mips64r3:
  case ArchSpec::eCore_mips64r3el:
    cpu = "mips64r3";
    break;
  case ArchSpec::eCore_mips64r5:
  case ArchSpec::eCore_mips64r5el:
    cpu = "mips64r5";
    break;
  case ArchSpec::eCore_mips64r6:
  case ArchSpec::eCore_mips64r6el:
    cpu = "mips64r6";
    break;
  default:
    cpu = "generic";
    break;
  }

  std::string features;
  uint32_t arch_flags = arch.GetFlags();
  if (arch_flags & ArchSpec::eMIPSAse_msa)
    features += "+msa,";
  if (arch_flags & ArchSpec::eMIPSAse_dsp)
    features += "+dsp,";
  if (arch_flags & ArchSpec::eMIPSAse_dspr2)
    features += "+dspr2,";
  if (arch_flags & ArchSpec::eMIPSAse_mips16)
    features += "+mips16,";
  if (arch_flags & ArchSpec::eMIPSAse_micromips)
    features += "+micromips,";

  m_reg_info.reset(target->createMCRegInfo(triple));
  assert(m_reg_info.get());

  m_insn_info.reset(target->createMCInstrInfo());
  assert(m_insn_info.get());

  m_asm_info.reset(target->createMCAsmInfo(*m_reg_info, triple, m_mc_options));
  m_subtype_info.reset(target->createMCSubtargetInfo(triple, cpu, features));
  assert(m_asm_info.get() && m_subtype_info.get());

  m_context = std::make_unique<llvm::MCContext>(triple, *m_asm_info,
                                                *m_reg_info, *m_subtype_info);
  assert(m_context.get());

  m_disasm.reset(target->createMCDisassembler(*m_subtype_info, *m_context));
  assert(m_disasm.get());
}

void EmulateInstructionMIPS64::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void EmulateInstructionMIPS64::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef EmulateInstructionMIPS64::GetPluginDescriptionStatic() {
  return "Emulate instructions for the MIPS64 architecture.";
}

EmulateInstruction *
EmulateInstructionMIPS64::CreateInstance(const ArchSpec &arch,
                                         InstructionType inst_type) {
  if (EmulateInstructionMIPS64::SupportsEmulatingInstructionsOfTypeStatic(
          inst_type)) {
    if (arch.GetTriple().getArch() == llvm::Triple::mips64 ||
        arch.GetTriple().getArch() == llvm::Triple::mips64el) {
      return new EmulateInstructionMIPS64(arch);
    }
  }

  return nullptr;
}

bool EmulateInstructionMIPS64::SetTargetTriple(const ArchSpec &arch) {
  return arch.GetTriple().getArch() == llvm::Triple::mips64 ||
         arch.GetTriple().getArch() == llvm::Triple::mips64el;
}

const char *EmulateInstructionMIPS64::GetRegisterName(unsigned reg_num,
                                                      bool alternate_name) {
  if (alternate_name) {
    switch (reg_num) {
    case registers::mips64::dwarf::sp:
      return "r29";
    case registers::mips64::dwarf::r30:
      return "r30";
    case registers::mips64::dwarf::ra:
      return "r31";
    case registers::mips64::dwarf::f0:
      return "f0";
    case registers::mips64::dwarf::f1:
      return "f1";
    case registers::mips64::dwarf::f2:
      return "f2";
    case registers::mips64::dwarf::f3:
      return "f3";
    case registers::mips64::dwarf::f4:
      return "f4";
    case registers::mips64::dwarf::f5:
      return "f5";
    case registers::mips64::dwarf::f6:
      return "f6";
    case registers::mips64::dwarf::f7:
      return "f7";
    case registers::mips64::dwarf::f8:
      return "f8";
    case registers::mips64::dwarf::f9:
      return "f9";
    case registers::mips64::dwarf::f10:
      return "f10";
    case registers::mips64::dwarf::f11:
      return "f11";
    case registers::mips64::dwarf::f12:
      return "f12";
    case registers::mips64::dwarf::f13:
      return "f13";
    case registers::mips64::dwarf::f14:
      return "f14";
    case registers::mips64::dwarf::f15:
      return "f15";
    case registers::mips64::dwarf::f16:
      return "f16";
    case registers::mips64::dwarf::f17:
      return "f17";
    case registers::mips64::dwarf::f18:
      return "f18";
    case registers::mips64::dwarf::f19:
      return "f19";
    case registers::mips64::dwarf::f20:
      return "f20";
    case registers::mips64::dwarf::f21:
      return "f21";
    case registers::mips64::dwarf::f22:
      return "f22";
    case registers::mips64::dwarf::f23:
      return "f23";
    case registers::mips64::dwarf::f24:
      return "f24";
    case registers::mips64::dwarf::f25:
      return "f25";
    case registers::mips64::dwarf::f26:
      return "f26";
    case registers::mips64::dwarf::f27:
      return "f27";
    case registers::mips64::dwarf::f28:
      return "f28";
    case registers::mips64::dwarf::f29:
      return "f29";
    case registers::mips64::dwarf::f30:
      return "f30";
    case registers::mips64::dwarf::f31:
      return "f31";
    case registers::mips64::dwarf::w0:
      return "w0";
    case registers::mips64::dwarf::w1:
      return "w1";
    case registers::mips64::dwarf::w2:
      return "w2";
    case registers::mips64::dwarf::w3:
      return "w3";
    case registers::mips64::dwarf::w4:
      return "w4";
    case registers::mips64::dwarf::w5:
      return "w5";
    case registers::mips64::dwarf::w6:
      return "w6";
    case registers::mips64::dwarf::w7:
      return "w7";
    case registers::mips64::dwarf::w8:
      return "w8";
    case registers::mips64::dwarf::w9:
      return "w9";
    case registers::mips64::dwarf::w10:
      return "w10";
    case registers::mips64::dwarf::w11:
      return "w11";
    case registers::mips64::dwarf::w12:
      return "w12";
    case registers::mips64::dwarf::w13:
      return "w13";
    case registers::mips64::dwarf::w14:
      return "w14";
    case registers::mips64::dwarf::w15:
      return "w15";
    case registers::mips64::dwarf::w16:
      return "w16";
    case registers::mips64::dwarf::w17:
      return "w17";
    case registers::mips64::dwarf::w18:
      return "w18";
    case registers::mips64::dwarf::w19:
      return "w19";
    case registers::mips64::dwarf::w20:
      return "w20";
    case registers::mips64::dwarf::w21:
      return "w21";
    case registers::mips64::dwarf::w22:
      return "w22";
    case registers::mips64::dwarf::w23:
      return "w23";
    case registers::mips64::dwarf::w24:
      return "w24";
    case registers::mips64::dwarf::w25:
      return "w25";
    case registers::mips64::dwarf::w26:
      return "w26";
    case registers::mips64::dwarf::w27:
      return "w27";
    case registers::mips64::dwarf::w28:
      return "w28";
    case registers::mips64::dwarf::w29:
      return "w29";
    case registers::mips64::dwarf::w30:
      return "w30";
    case registers::mips64::dwarf::w31:
      return "w31";
    case registers::mips64::dwarf::mir:
      return "mir";
    case registers::mips64::dwarf::mcsr:
      return "mcsr";
    case registers::mips64::dwarf::config5:
      return "config5";
    default:
      break;
    }
    return nullptr;
  }

  switch (reg_num) {
  case registers::mips64::dwarf::zero:
    return "r0";
  case registers::mips64::dwarf::r1:
    return "r1";
  case registers::mips64::dwarf::r2:
    return "r2";
  case registers::mips64::dwarf::r3:
    return "r3";
  case registers::mips64::dwarf::r4:
    return "r4";
  case registers::mips64::dwarf::r5:
    return "r5";
  case registers::mips64::dwarf::r6:
    return "r6";
  case registers::mips64::dwarf::r7:
    return "r7";
  case registers::mips64::dwarf::r8:
    return "r8";
  case registers::mips64::dwarf::r9:
    return "r9";
  case registers::mips64::dwarf::r10:
    return "r10";
  case registers::mips64::dwarf::r11:
    return "r11";
  case registers::mips64::dwarf::r12:
    return "r12";
  case registers::mips64::dwarf::r13:
    return "r13";
  case registers::mips64::dwarf::r14:
    return "r14";
  case registers::mips64::dwarf::r15:
    return "r15";
  case registers::mips64::dwarf::r16:
    return "r16";
  case registers::mips64::dwarf::r17:
    return "r17";
  case registers::mips64::dwarf::r18:
    return "r18";
  case registers::mips64::dwarf::r19:
    return "r19";
  case registers::mips64::dwarf::r20:
    return "r20";
  case registers::mips64::dwarf::r21:
    return "r21";
  case registers::mips64::dwarf::r22:
    return "r22";
  case registers::mips64::dwarf::r23:
    return "r23";
  case registers::mips64::dwarf::r24:
    return "r24";
  case registers::mips64::dwarf::r25:
    return "r25";
  case registers::mips64::dwarf::r26:
    return "r26";
  case registers::mips64::dwarf::r27:
    return "r27";
  case registers::mips64::dwarf::gp:
    return "gp";
  case registers::mips64::dwarf::sp:
    return "sp";
  case registers::mips64::dwarf::r30:
    return "fp";
  case registers::mips64::dwarf::ra:
    return "ra";
  case registers::mips64::dwarf::sr:
    return "sr";
  case registers::mips64::dwarf::lo:
    return "lo";
  case registers::mips64::dwarf::hi:
    return "hi";
  case registers::mips64::dwarf::bad:
    return "bad";
  case registers::mips64::dwarf::cause:
    return "cause";
  case registers::mips64::dwarf::pc:
    return "pc";
  case registers::mips64::dwarf::f0:
    return "f0";
  case registers::mips64::dwarf::f1:
    return "f1";
  case registers::mips64::dwarf::f2:
    return "f2";
  case registers::mips64::dwarf::f3:
    return "f3";
  case registers::mips64::dwarf::f4:
    return "f4";
  case registers::mips64::dwarf::f5:
    return "f5";
  case registers::mips64::dwarf::f6:
    return "f6";
  case registers::mips64::dwarf::f7:
    return "f7";
  case registers::mips64::dwarf::f8:
    return "f8";
  case registers::mips64::dwarf::f9:
    return "f9";
  case registers::mips64::dwarf::f10:
    return "f10";
  case registers::mips64::dwarf::f11:
    return "f11";
  case registers::mips64::dwarf::f12:
    return "f12";
  case registers::mips64::dwarf::f13:
    return "f13";
  case registers::mips64::dwarf::f14:
    return "f14";
  case registers::mips64::dwarf::f15:
    return "f15";
  case registers::mips64::dwarf::f16:
    return "f16";
  case registers::mips64::dwarf::f17:
    return "f17";
  case registers::mips64::dwarf::f18:
    return "f18";
  case registers::mips64::dwarf::f19:
    return "f19";
  case registers::mips64::dwarf::f20:
    return "f20";
  case registers::mips64::dwarf::f21:
    return "f21";
  case registers::mips64::dwarf::f22:
    return "f22";
  case registers::mips64::dwarf::f23:
    return "f23";
  case registers::mips64::dwarf::f24:
    return "f24";
  case registers::mips64::dwarf::f25:
    return "f25";
  case registers::mips64::dwarf::f26:
    return "f26";
  case registers::mips64::dwarf::f27:
    return "f27";
  case registers::mips64::dwarf::f28:
    return "f28";
  case registers::mips64::dwarf::f29:
    return "f29";
  case registers::mips64::dwarf::f30:
    return "f30";
  case registers::mips64::dwarf::f31:
    return "f31";
  case registers::mips64::dwarf::fcsr:
    return "fcsr";
  case registers::mips64::dwarf::fir:
    return "fir";
  case registers::mips64::dwarf::w0:
    return "w0";
  case registers::mips64::dwarf::w1:
    return "w1";
  case registers::mips64::dwarf::w2:
    return "w2";
  case registers::mips64::dwarf::w3:
    return "w3";
  case registers::mips64::dwarf::w4:
    return "w4";
  case registers::mips64::dwarf::w5:
    return "w5";
  case registers::mips64::dwarf::w6:
    return "w6";
  case registers::mips64::dwarf::w7:
    return "w7";
  case registers::mips64::dwarf::w8:
    return "w8";
  case registers::mips64::dwarf::w9:
    return "w9";
  case registers::mips64::dwarf::w10:
    return "w10";
  case registers::mips64::dwarf::w11:
    return "w11";
  case registers::mips64::dwarf::w12:
    return "w12";
  case registers::mips64::dwarf::w13:
    return "w13";
  case registers::mips64::dwarf::w14:
    return "w14";
  case registers::mips64::dwarf::w15:
    return "w15";
  case registers::mips64::dwarf::w16:
    return "w16";
  case registers::mips64::dwarf::w17:
    return "w17";
  case registers::mips64::dwarf::w18:
    return "w18";
  case registers::mips64::dwarf::w19:
    return "w19";
  case registers::mips64::dwarf::w20:
    return "w20";
  case registers::mips64::dwarf::w21:
    return "w21";
  case registers::mips64::dwarf::w22:
    return "w22";
  case registers::mips64::dwarf::w23:
    return "w23";
  case registers::mips64::dwarf::w24:
    return "w24";
  case registers::mips64::dwarf::w25:
    return "w25";
  case registers::mips64::dwarf::w26:
    return "w26";
  case registers::mips64::dwarf::w27:
    return "w27";
  case registers::mips64::dwarf::w28:
    return "w28";
  case registers::mips64::dwarf::w29:
    return "w29";
  case registers::mips64::dwarf::w30:
    return "w30";
  case registers::mips64::dwarf::w31:
    return "w31";
  case registers::mips64::dwarf::mcsr:
    return "mcsr";
  case registers::mips64::dwarf::mir:
    return "mir";
  case registers::mips64::dwarf::config5:
    return "config5";
  }
  return nullptr;
}

std::optional<RegisterDescriptor>
EmulateInstructionMIPS64::GetRegisterInfo(RegisterKind reg_kind,
                                          uint32_t reg_num) {
  if (reg_kind == eRegisterKindGeneric) {
    switch (reg_num) {
    case LLDB_REGNUM_GENERIC_PC:
      reg_kind = eRegisterKindDWARF;
      reg_num = registers::mips64::dwarf::pc;
      break;
    case LLDB_REGNUM_GENERIC_SP:
      reg_kind = eRegisterKindDWARF;
      reg_num = registers::mips64::dwarf::sp;
      break;
    case LLDB_REGNUM_GENERIC_FP:
      reg_kind = eRegisterKindDWARF;
      reg_num = registers::mips64::dwarf::r30;
      break;
    case LLDB_REGNUM_GENERIC_RA:
      reg_kind = eRegisterKindDWARF;
      reg_num = registers::mips64::dwarf::ra;
      break;
    case LLDB_REGNUM_GENERIC_FLAGS:
      reg_kind = eRegisterKindDWARF;
      reg_num = registers::mips64::dwarf::sr;
      break;
    default:
      return {};
    }
  }

  if (reg_kind == eRegisterKindDWARF) {
    RegisterDescriptor reg_info;

    if (reg_num == registers::mips64::dwarf::sr ||
        reg_num == registers::mips64::dwarf::fcsr ||
        reg_num == registers::mips64::dwarf::fir ||
        reg_num == registers::mips64::dwarf::mcsr ||
        reg_num == registers::mips64::dwarf::mir ||
        reg_num == registers::mips64::dwarf::config5) {
      reg_info.byte_size = 4;
      reg_info.format = eFormatHex;
      reg_info.encoding = eEncodingUint;
    } else if (reg_num <= registers::mips64::dwarf::f31) {
      reg_info.byte_size = 8;
      reg_info.format = eFormatHex;
      reg_info.encoding = eEncodingUint;
    } else if (reg_num >= registers::mips64::dwarf::w0 &&
               reg_num <= registers::mips64::dwarf::w31) {
      reg_info.byte_size = 16;
      reg_info.format = eFormatVectorOfUInt8;
      reg_info.encoding = eEncodingVector;
    } else {
      return {};
    }

    reg_info.name = GetRegisterName(reg_num, false);
    if (const char *alternate = GetRegisterName(reg_num, true))
      reg_info.alt_name = alternate;
    reg_info.numbers[eRegisterKindDWARF] = reg_num;

    switch (reg_num) {
    case registers::mips64::dwarf::r30:
      reg_info.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
      break;
    case registers::mips64::dwarf::ra:
      reg_info.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA;
      break;
    case registers::mips64::dwarf::sp:
      reg_info.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
      break;
    case registers::mips64::dwarf::pc:
      reg_info.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
      break;
    case registers::mips64::dwarf::sr:
      reg_info.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      break;
    default:
      break;
    }
    return reg_info;
  }
  return {};
}

EmulateInstructionMIPS64::MipsOpcode *
EmulateInstructionMIPS64::GetOpcodeForInstruction(llvm::StringRef op_name) {
  static EmulateInstructionMIPS64::MipsOpcode g_opcodes[] = {
      // Prologue/Epilogue instructions
      {"DADDiu", &EmulateInstructionMIPS64::Emulate_DADDiu,
       "DADDIU rt, rs, immediate"},
      {"ADDiu", &EmulateInstructionMIPS64::Emulate_DADDiu,
       "ADDIU  rt, rs, immediate"},
      {"SD", &EmulateInstructionMIPS64::Emulate_SD, "SD     rt, offset(rs)"},
      {"LD", &EmulateInstructionMIPS64::Emulate_LD, "LD     rt, offset(base)"},
      {"DSUBU", &EmulateInstructionMIPS64::Emulate_DSUBU_DADDU,
       "DSUBU  rd, rs, rt"},
      {"SUBU", &EmulateInstructionMIPS64::Emulate_DSUBU_DADDU,
       "SUBU   rd, rs, rt"},
      {"DADDU", &EmulateInstructionMIPS64::Emulate_DSUBU_DADDU,
       "DADDU  rd, rs, rt"},
      {"ADDU", &EmulateInstructionMIPS64::Emulate_DSUBU_DADDU,
       "ADDU   rd, rs, rt"},
      {"LUI", &EmulateInstructionMIPS64::Emulate_LUI, "LUI    rt, immediate"},

      // Load/Store  instructions
      /* Following list of emulated instructions are required by implementation
         of hardware watchpoint
         for MIPS in lldb. As we just need the address accessed by instructions,
         we have generalised
         all these instructions in 2 functions depending on their addressing
         modes */

      {"LB", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LB    rt, offset(base)"},
      {"LBE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LBE   rt, offset(base)"},
      {"LBU", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LBU   rt, offset(base)"},
      {"LBUE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LBUE  rt, offset(base)"},
      {"LDC1", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LDC1  ft, offset(base)"},
      {"LDL", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LDL   rt, offset(base)"},
      {"LDR", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LDR   rt, offset(base)"},
      {"LLD", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LLD   rt, offset(base)"},
      {"LDC2", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LDC2  rt, offset(base)"},
      {"LDXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "LDXC1 fd, index (base)"},
      {"LH", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LH    rt, offset(base)"},
      {"LHE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LHE   rt, offset(base)"},
      {"LHU", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LHU   rt, offset(base)"},
      {"LHUE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LHUE  rt, offset(base)"},
      {"LL", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LL    rt, offset(base)"},
      {"LLE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LLE   rt, offset(base)"},
      {"LUXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "LUXC1 fd, index (base)"},
      {"LW", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LW    rt, offset(rs)"},
      {"LWC1", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWC1  ft, offset(base)"},
      {"LWC2", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWC2  rt, offset(base)"},
      {"LWE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWE   rt, offset(base)"},
      {"LWL", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWL   rt, offset(base)"},
      {"LWLE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWLE  rt, offset(base)"},
      {"LWR", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWR   rt, offset(base)"},
      {"LWRE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "LWRE  rt, offset(base)"},
      {"LWXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "LWXC1 fd, index (base)"},

      {"SB", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SB    rt, offset(base)"},
      {"SBE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SBE   rt, offset(base)"},
      {"SC", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SC    rt, offset(base)"},
      {"SCE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SCE   rt, offset(base)"},
      {"SCD", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SCD   rt, offset(base)"},
      {"SDL", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SDL   rt, offset(base)"},
      {"SDR", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SDR   rt, offset(base)"},
      {"SDC1", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SDC1  ft, offset(base)"},
      {"SDC2", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SDC2  rt, offset(base)"},
      {"SDXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "SDXC1 fs, index (base)"},
      {"SH", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SH    rt, offset(base)"},
      {"SHE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SHE   rt, offset(base)"},
      {"SUXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "SUXC1 fs, index (base)"},
      {"SW", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SW    rt, offset(rs)"},
      {"SWC1", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWC1  ft, offset(base)"},
      {"SWC2", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWC2  rt, offset(base)"},
      {"SWE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWE   rt, offset(base)"},
      {"SWL", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWL   rt, offset(base)"},
      {"SWLE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWLE  rt, offset(base)"},
      {"SWR", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWR   rt, offset(base)"},
      {"SWRE", &EmulateInstructionMIPS64::Emulate_LDST_Imm,
       "SWRE  rt, offset(base)"},
      {"SWXC1", &EmulateInstructionMIPS64::Emulate_LDST_Reg,
       "SWXC1 fs, index (base)"},

      // Branch instructions
      {"BEQ", &EmulateInstructionMIPS64::Emulate_BXX_3ops, "BEQ rs,rt,offset"},
      {"BEQ64", &EmulateInstructionMIPS64::Emulate_BXX_3ops, "BEQ rs,rt,offset"},
      {"BNE", &EmulateInstructionMIPS64::Emulate_BXX_3ops, "BNE rs,rt,offset"},
      {"BNE64", &EmulateInstructionMIPS64::Emulate_BXX_3ops, "BNE rs,rt,offset"},
      {"BEQL", &EmulateInstructionMIPS64::Emulate_BXX_3ops,
       "BEQL rs,rt,offset"},
      {"BNEL", &EmulateInstructionMIPS64::Emulate_BXX_3ops,
       "BNEL rs,rt,offset"},
      {"BGEZALL", &EmulateInstructionMIPS64::Emulate_Bcond_Link,
       "BGEZALL rt,offset"},
      {"BAL", &EmulateInstructionMIPS64::Emulate_BAL, "BAL offset"},
      {"BGEZAL", &EmulateInstructionMIPS64::Emulate_Bcond_Link,
       "BGEZAL rs,offset"},
      {"BALC", &EmulateInstructionMIPS64::Emulate_BALC, "BALC offset"},
      {"BC", &EmulateInstructionMIPS64::Emulate_BC, "BC offset"},
      {"BGEZ", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGEZ rs,offset"},
      {"BGEZ64", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGEZ rs,offset"},
      {"BLEZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BLEZALC rs,offset"},
      {"BGEZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BGEZALC rs,offset"},
      {"BLTZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BLTZALC rs,offset"},
      {"BGTZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BGTZALC rs,offset"},
      {"BEQZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BEQZALC rs,offset"},
      {"BNEZALC", &EmulateInstructionMIPS64::Emulate_Bcond_Link_C,
       "BNEZALC rs,offset"},
      {"BEQC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BEQC rs,rt,offset"},
      {"BEQC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BEQC rs,rt,offset"},
      {"BNEC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BNEC rs,rt,offset"},
      {"BNEC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BNEC rs,rt,offset"},
      {"BLTC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BLTC rs,rt,offset"},
      {"BLTC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BLTC rs,rt,offset"},
      {"BGEC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BGEC rs,rt,offset"},
      {"BGEC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BGEC rs,rt,offset"},
      {"BLTUC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BLTUC rs,rt,offset"},
      {"BLTUC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BLTUC rs,rt,offset"},
      {"BGEUC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BGEUC rs,rt,offset"},
      {"BGEUC64", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BGEUC rs,rt,offset"},
      {"BLTZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BLTZC rt,offset"},
      {"BLTZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BLTZC rt,offset"},
      {"BLEZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BLEZC rt,offset"},
      {"BLEZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BLEZC rt,offset"},
      {"BGEZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BGEZC rt,offset"},
      {"BGEZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BGEZC rt,offset"},
      {"BGTZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BGTZC rt,offset"},
      {"BGTZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BGTZC rt,offset"},
      {"BEQZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BEQZC rt,offset"},
      {"BEQZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BEQZC rt,offset"},
      {"BNEZC", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BNEZC rt,offset"},
      {"BNEZC64", &EmulateInstructionMIPS64::Emulate_BXX_2ops_C,
       "BNEZC rt,offset"},
      {"BGEZL", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGEZL rt,offset"},
      {"BGTZ", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGTZ rt,offset"},
      {"BGTZ64", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGTZ rt,offset"},
      {"BGTZL", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BGTZL rt,offset"},
      {"BLEZ", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLEZ rt,offset"},
      {"BLEZ64", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLEZ rt,offset"},
      {"BLEZL", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLEZL rt,offset"},
      {"BLTZ", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLTZ rt,offset"},
      {"BLTZ64", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLTZ rt,offset"},
      {"BLTZAL", &EmulateInstructionMIPS64::Emulate_Bcond_Link,
       "BLTZAL rt,offset"},
      {"BLTZALL", &EmulateInstructionMIPS64::Emulate_Bcond_Link,
       "BLTZALL rt,offset"},
      {"BLTZL", &EmulateInstructionMIPS64::Emulate_BXX_2ops, "BLTZL rt,offset"},
      {"BOVC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BOVC rs,rt,offset"},
      {"BNVC", &EmulateInstructionMIPS64::Emulate_BXX_3ops_C,
       "BNVC rs,rt,offset"},
      {"J", &EmulateInstructionMIPS64::Emulate_J, "J target"},
      {"JAL", &EmulateInstructionMIPS64::Emulate_JAL, "JAL target"},
      {"JALX", &EmulateInstructionMIPS64::Emulate_JAL, "JALX target"},
      {"JALR", &EmulateInstructionMIPS64::Emulate_JALR, "JALR target"},
      {"JALR64", &EmulateInstructionMIPS64::Emulate_JALR, "JALR target"},
      {"JALR_HB", &EmulateInstructionMIPS64::Emulate_JALR, "JALR.HB target"},
      {"JIALC", &EmulateInstructionMIPS64::Emulate_JIALC, "JIALC rt,offset"},
      {"JIALC64", &EmulateInstructionMIPS64::Emulate_JIALC, "JIALC rt,offset"},
      {"JIC", &EmulateInstructionMIPS64::Emulate_JIC, "JIC rt,offset"},
      {"JIC64", &EmulateInstructionMIPS64::Emulate_JIC, "JIC rt,offset"},
      {"JR", &EmulateInstructionMIPS64::Emulate_JR, "JR target"},
      {"JR64", &EmulateInstructionMIPS64::Emulate_JR, "JR target"},
      {"JR_HB", &EmulateInstructionMIPS64::Emulate_JR, "JR.HB target"},
      {"BC1F", &EmulateInstructionMIPS64::Emulate_FP_branch, "BC1F cc, offset"},
      {"BC1T", &EmulateInstructionMIPS64::Emulate_FP_branch, "BC1T cc, offset"},
      {"BC1FL", &EmulateInstructionMIPS64::Emulate_FP_branch,
       "BC1FL cc, offset"},
      {"BC1TL", &EmulateInstructionMIPS64::Emulate_FP_branch,
       "BC1TL cc, offset"},
      {"BC1EQZ", &EmulateInstructionMIPS64::Emulate_BC1EQZ,
       "BC1EQZ ft, offset"},
      {"BC1NEZ", &EmulateInstructionMIPS64::Emulate_BC1NEZ,
       "BC1NEZ ft, offset"},
      {"BC1ANY2F", &EmulateInstructionMIPS64::Emulate_3D_branch,
       "BC1ANY2F cc, offset"},
      {"BC1ANY2T", &EmulateInstructionMIPS64::Emulate_3D_branch,
       "BC1ANY2T cc, offset"},
      {"BC1ANY4F", &EmulateInstructionMIPS64::Emulate_3D_branch,
       "BC1ANY4F cc, offset"},
      {"BC1ANY4T", &EmulateInstructionMIPS64::Emulate_3D_branch,
       "BC1ANY4T cc, offset"},
      {"BNZ_B", &EmulateInstructionMIPS64::Emulate_BNZB, "BNZ.b wt,s16"},
      {"BNZ_H", &EmulateInstructionMIPS64::Emulate_BNZH, "BNZ.h wt,s16"},
      {"BNZ_W", &EmulateInstructionMIPS64::Emulate_BNZW, "BNZ.w wt,s16"},
      {"BNZ_D", &EmulateInstructionMIPS64::Emulate_BNZD, "BNZ.d wt,s16"},
      {"BZ_B", &EmulateInstructionMIPS64::Emulate_BZB, "BZ.b wt,s16"},
      {"BZ_H", &EmulateInstructionMIPS64::Emulate_BZH, "BZ.h wt,s16"},
      {"BZ_W", &EmulateInstructionMIPS64::Emulate_BZW, "BZ.w wt,s16"},
      {"BZ_D", &EmulateInstructionMIPS64::Emulate_BZD, "BZ.d wt,s16"},
      {"BNZ_V", &EmulateInstructionMIPS64::Emulate_BNZV, "BNZ.V wt,s16"},
      {"BZ_V", &EmulateInstructionMIPS64::Emulate_BZV, "BZ.V wt,s16"},
  };

  for (MipsOpcode &opcode : g_opcodes) {
    if (op_name.equals_insensitive(opcode.op_name))
      return &opcode;
  }
  return nullptr;
}

bool EmulateInstructionMIPS64::ReadInstruction() {
  bool success = false;
  m_addr = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC,
                                LLDB_INVALID_ADDRESS, &success);
  if (success) {
    Context read_inst_context;
    read_inst_context.type = eContextReadOpcode;
    read_inst_context.SetNoArgs();
    m_opcode.SetOpcode32(
        ReadMemoryUnsigned(read_inst_context, m_addr, 4, 0, &success),
        GetByteOrder());
  }
  if (!success)
    m_addr = LLDB_INVALID_ADDRESS;
  return success;
}

bool EmulateInstructionMIPS64::EvaluateInstruction(uint32_t evaluate_options) {
  bool success = false;
  llvm::MCInst mc_insn;
  uint64_t insn_size;
  DataExtractor data;

  /* Keep the complexity of the decode logic with the llvm::MCDisassembler
   * class. */
  if (m_opcode.GetData(data)) {
    llvm::MCDisassembler::DecodeStatus decode_status;
    llvm::ArrayRef<uint8_t> raw_insn(data.GetDataStart(), data.GetByteSize());
    decode_status = m_disasm->getInstruction(mc_insn, insn_size, raw_insn,
                                             m_addr, llvm::nulls());
    if (decode_status != llvm::MCDisassembler::Success)
      return false;
  }

  /*
   * mc_insn.getOpcode() returns decoded opcode. However to make use
   * of llvm::Mips::<insn> we would need "MipsGenInstrInfo.inc".
  */
  llvm::StringRef op_name = m_insn_info->getName(mc_insn.getOpcode());

  /*
   * Decoding has been done already. Just get the call-back function
   * and emulate the instruction.
  */
  MipsOpcode *opcode_data = GetOpcodeForInstruction(op_name);

  if (opcode_data == nullptr)
    return false;

  uint64_t old_pc = 0, new_pc = 0;
  const bool auto_advance_pc =
      evaluate_options & eEmulateInstructionOptionAutoAdvancePC;

  if (auto_advance_pc) {
    old_pc =
        ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0, &success);
    if (!success)
      return false;
  }

  /* emulate instruction */
  success = (this->*opcode_data->callback)(mc_insn);
  if (!success)
    return false;

  if (auto_advance_pc) {
    new_pc =
        ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0, &success);
    if (!success)
      return false;

    /* If we haven't changed the PC, change it here */
    if (old_pc == new_pc) {
      new_pc += 4;
      Context context;
      if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                                 new_pc))
        return false;
    }
  }

  return true;
}

bool EmulateInstructionMIPS64::CreateFunctionEntryUnwind(
    UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(eRegisterKindDWARF);

  UnwindPlan::Row row;
  const bool can_replace = false;

  // Our previous Call Frame Address is the stack pointer
  row.GetCFAValue().SetIsRegisterPlusOffset(registers::mips64::dwarf::sp, 0);

  // Our previous PC is in the RA
  row.SetRegisterLocationToRegister(registers::mips64::dwarf::pc, registers::mips64::dwarf::ra,
                                    can_replace);

  unwind_plan.AppendRow(std::move(row));

  // All other registers are the same.
  unwind_plan.SetSourceName("EmulateInstructionMIPS64");
  unwind_plan.SetSourcedFromCompiler(eLazyBoolNo);
  unwind_plan.SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  unwind_plan.SetUnwindPlanForSignalTrap(eLazyBoolNo);
  unwind_plan.SetReturnAddressRegister(registers::mips64::dwarf::ra);

  return true;
}

bool EmulateInstructionMIPS64::nonvolatile_reg_p(uint64_t regnum) {
  switch (regnum) {
  case registers::mips64::dwarf::r16:
  case registers::mips64::dwarf::r17:
  case registers::mips64::dwarf::r18:
  case registers::mips64::dwarf::r19:
  case registers::mips64::dwarf::r20:
  case registers::mips64::dwarf::r21:
  case registers::mips64::dwarf::r22:
  case registers::mips64::dwarf::r23:
  case registers::mips64::dwarf::gp:
  case registers::mips64::dwarf::sp:
  case registers::mips64::dwarf::r30:
  case registers::mips64::dwarf::ra:
    return true;
  default:
    return false;
  }
  return false;
}

bool EmulateInstructionMIPS64::Emulate_DADDiu(llvm::MCInst &insn) {
  // DADDIU rt, rs, immediate
  // GPR[rt] <- GPR[rs] + sign_extend(immediate)

  uint8_t dst, src;
  bool success = false;
  const uint32_t imm16 = insn.getOperand(2).getImm();
  int64_t imm = SignedBits(imm16, 15, 0);

  dst = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  src = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());

  // If immediate is greater than 2^16 - 1 then clang generate LUI,
  // (D)ADDIU,(D)SUBU instructions in prolog. Example lui    $1, 0x2 daddiu $1,
  // $1, -0x5920 dsubu  $sp, $sp, $1 In this case, (D)ADDIU dst and src will be
  // same and not equal to sp
  if (dst == src) {
    Context context;

    /* read <src> register */
    const uint64_t src_opd_val = ReadRegisterUnsigned(
        eRegisterKindDWARF, registers::mips64::dwarf::zero + src, 0, &success);
    if (!success)
      return false;

    /* Check if this is daddiu sp, sp, imm16 */
    if (dst == registers::mips64::dwarf::sp) {
      /*
       * From the MIPS IV spec:
       *
       * The term “unsigned” in the instruction name is a misnomer; this
       * operation is 64-bit modulo arithmetic that does not trap on overflow.
       * It is appropriate for arithmetic which is not signed, such as address
       * arithmetic, or integer arithmetic environments that ignore overflow,
       * such as “C” language arithmetic.
       *
       * Assume 2's complement and rely on unsigned overflow here.
       */
      uint64_t result = src_opd_val + imm;
      std::optional<RegisterDescriptor> reg_info_sp =
          GetRegisterInfo(eRegisterKindDWARF, registers::mips64::dwarf::sp);
      if (reg_info_sp)
        context.SetRegisterPlusOffset(*reg_info_sp, imm);

      /* We are allocating bytes on stack */
      context.type = eContextAdjustStackPointer;

      WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::sp,
                            result);
      return true;
    }

    imm += src_opd_val;
    context.SetImmediateSigned(imm);
    context.type = eContextImmediate;

    if (!WriteRegisterUnsigned(context, eRegisterKindDWARF,
                               registers::mips64::dwarf::zero + dst, imm))
      return false;
  }

  return true;
}

bool EmulateInstructionMIPS64::Emulate_SD(llvm::MCInst &insn) {
  uint64_t address;
  bool success = false;
  uint32_t imm16 = insn.getOperand(2).getImm();
  uint64_t imm = SignedBits(imm16, 15, 0);
  uint32_t src, base;
  Context bad_vaddr_context;

  src = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  base = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());

  std::optional<RegisterDescriptor> reg_info_base =
      GetRegisterInfo(eRegisterKindDWARF, registers::mips64::dwarf::zero + base);
  std::optional<RegisterDescriptor> reg_info_src =
      GetRegisterInfo(eRegisterKindDWARF, registers::mips64::dwarf::zero + src);
  if (!reg_info_base || !reg_info_src)
    return false;

  /* read SP */
  address = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + base,
                                 0, &success);
  if (!success)
    return false;

  /* destination address */
  address = address + imm;

  /* We look for sp based non-volatile register stores */
  if (nonvolatile_reg_p(src)) {
    Context context;
    context.type = eContextPushRegisterOnStack;
    context.SetRegisterToRegisterPlusOffset(*reg_info_src, *reg_info_base, 0);

    std::optional<RegisterValue> data_src = ReadRegister(*reg_info_base);
    if (!data_src)
      return false;

    Status error;
    RegisterValue::BytesContainer buffer(reg_info_src->byte_size);
    if (data_src->GetAsMemoryData(reg_info_src->byte_size, buffer.data(),
                                  reg_info_src->byte_size, eByteOrderLittle,
                                  error, reg_info_src->name) == 0)
      return false;

    if (!WriteMemory(context, address, buffer.data(), reg_info_src->byte_size))
      return false;
  }

  /* Set the bad_vaddr register with base address used in the instruction */
  bad_vaddr_context.type = eContextInvalid;
  WriteRegisterUnsigned(bad_vaddr_context, eRegisterKindDWARF, registers::mips64::dwarf::bad,
                        address);

  return true;
}

bool EmulateInstructionMIPS64::Emulate_LD(llvm::MCInst &insn) {
  bool success = false;
  uint32_t src, base;
  int64_t imm, address;
  Context bad_vaddr_context;

  src = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  base = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());
  imm = insn.getOperand(2).getImm();

  if (!GetRegisterInfo(eRegisterKindDWARF,
                       registers::mips64::dwarf::zero + base))
    return false;

  /* read base register */
  address = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + base,
                                 0, &success);
  if (!success)
    return false;

  /* destination address */
  address = address + imm;

  /* Set the bad_vaddr register with base address used in the instruction */
  bad_vaddr_context.type = eContextInvalid;
  WriteRegisterUnsigned(bad_vaddr_context, eRegisterKindDWARF, registers::mips64::dwarf::bad,
                        address);

  if (nonvolatile_reg_p(src)) {
    RegisterValue data_src;
    std::optional<RegisterDescriptor> reg_info_src =
        GetRegisterInfo(eRegisterKindDWARF, registers::mips64::dwarf::zero + src);
    if (!reg_info_src)
      return false;

    Context context;
    context.type = eContextRegisterLoad;

    return WriteRegister(context, *reg_info_src, data_src);
  }

  return false;
}

bool EmulateInstructionMIPS64::Emulate_LUI(llvm::MCInst &insn) {
  // LUI rt, immediate
  // GPR[rt] <- sign_extend(immediate << 16)

  const uint32_t imm32 = insn.getOperand(1).getImm() << 16;
  int64_t imm = SignedBits(imm32, 31, 0);
  uint8_t rt;
  Context context;

  rt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  context.SetImmediateSigned(imm);
  context.type = eContextImmediate;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF,
                               registers::mips64::dwarf::zero + rt, imm);
}

bool EmulateInstructionMIPS64::Emulate_DSUBU_DADDU(llvm::MCInst &insn) {
  // DSUBU sp, <src>, <rt>
  // DADDU sp, <src>, <rt>
  // DADDU dst, sp, <rt>

  bool success = false;
  uint64_t result;
  uint8_t src, dst, rt;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  dst = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  src = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());

  /* Check if sp is destination register */
  if (dst == registers::mips64::dwarf::sp) {
    rt = m_reg_info->getEncodingValue(insn.getOperand(2).getReg());

    /* read <src> register */
    uint64_t src_opd_val = ReadRegisterUnsigned(
        eRegisterKindDWARF, registers::mips64::dwarf::zero + src, 0, &success);
    if (!success)
      return false;

    /* read <rt > register */
    uint64_t rt_opd_val = ReadRegisterUnsigned(
        eRegisterKindDWARF, registers::mips64::dwarf::zero + rt, 0, &success);
    if (!success)
      return false;

    if (op_name.equals_insensitive("DSUBU") ||
        op_name.equals_insensitive("SUBU"))
      result = src_opd_val - rt_opd_val;
    else
      result = src_opd_val + rt_opd_val;

    Context context;
    std::optional<RegisterDescriptor> reg_info_sp =
        GetRegisterInfo(eRegisterKindDWARF, registers::mips64::dwarf::sp);
    if (reg_info_sp)
      context.SetRegisterPlusOffset(*reg_info_sp, rt_opd_val);

    /* We are allocating bytes on stack */
    context.type = eContextAdjustStackPointer;

    WriteRegisterUnsigned(context, eRegisterKindDWARF,
                          registers::mips64::dwarf::sp, result);

    return true;
  } else if (src == registers::mips64::dwarf::sp) {
    rt = m_reg_info->getEncodingValue(insn.getOperand(2).getReg());

    /* read <src> register */
    uint64_t src_opd_val = ReadRegisterUnsigned(
        eRegisterKindDWARF, registers::mips64::dwarf::zero + src, 0, &success);
    if (!success)
      return false;

    /* read <rt> register */
    uint64_t rt_opd_val = ReadRegisterUnsigned(
        eRegisterKindDWARF, registers::mips64::dwarf::zero + rt, 0, &success);
    if (!success)
      return false;

    Context context;

    if (op_name.equals_insensitive("DSUBU") ||
        op_name.equals_insensitive("SUBU"))
      result = src_opd_val - rt_opd_val;
    else
      result = src_opd_val + rt_opd_val;

    context.SetImmediateSigned(result);
    context.type = eContextImmediate;

    if (!WriteRegisterUnsigned(context, eRegisterKindDWARF,
                               registers::mips64::dwarf::zero + dst, result))
      return false;
  }

  return true;
}

/*
    Emulate below MIPS branch instructions.
    BEQ, BNE : Branch on condition
    BEQL, BNEL : Branch likely
*/
bool EmulateInstructionMIPS64::Emulate_BXX_3ops(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs, rt;
  int64_t offset, pc, rs_val, rt_val, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  rt = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());
  offset = insn.getOperand(2).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  rt_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rt, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BEQ") || op_name.equals_insensitive("BEQL") ||
      op_name.equals_insensitive("BEQ64")) {
    if (rs_val == rt_val)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BNE") ||
             op_name.equals_insensitive("BNEL") ||
             op_name.equals_insensitive("BNE64")) {
    if (rs_val != rt_val)
      target = pc + offset;
    else
      target = pc + 8;
  }

  Context context;
  context.type = eContextRelativeBranchImmediate;
  context.SetImmediate(offset);

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

/*
    Emulate below MIPS Non-Compact conditional branch and link instructions.
    BLTZAL, BGEZAL      :
    BLTZALL, BGEZALL    : Branch likely
*/
bool EmulateInstructionMIPS64::Emulate_Bcond_Link(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs;
  int64_t offset, pc, target = 0;
  int64_t rs_val;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BLTZAL") ||
      op_name.equals_insensitive("BLTZALL")) {
    if (rs_val < 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BGEZAL") ||
             op_name.equals_insensitive("BGEZALL")) {
    if (rs_val >= 0)
      target = pc + offset;
    else
      target = pc + 8;
  }

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 8))
    return false;

  return true;
}

bool EmulateInstructionMIPS64::Emulate_BAL(llvm::MCInst &insn) {
  bool success = false;
  int64_t offset, pc, target;

  /*
   * BAL offset
   *      offset = sign_ext (offset << 2)
   *      RA = PC + 8
   *      PC = PC + offset
  */
  offset = insn.getOperand(0).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  target = pc + offset;

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 8))
    return false;

  return true;
}

bool EmulateInstructionMIPS64::Emulate_BALC(llvm::MCInst &insn) {
  bool success = false;
  int64_t offset, pc, target;

  /*
   * BALC offset
   *      offset = sign_ext (offset << 2)
   *      RA = PC + 4
   *      PC = PC + 4 + offset
  */
  offset = insn.getOperand(0).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  target = pc + offset;

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 4))
    return false;

  return true;
}

/*
    Emulate below MIPS conditional branch and link instructions.
    BLEZALC, BGEZALC, BLTZALC, BGTZALC, BEQZALC, BNEZALC : Compact branches
*/
bool EmulateInstructionMIPS64::Emulate_Bcond_Link_C(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs;
  int64_t offset, pc, rs_val, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BLEZALC")) {
    if (rs_val <= 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGEZALC")) {
    if (rs_val >= 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BLTZALC")) {
    if (rs_val < 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGTZALC")) {
    if (rs_val > 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BEQZALC")) {
    if (rs_val == 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BNEZALC")) {
    if (rs_val != 0)
      target = pc + offset;
    else
      target = pc + 4;
  }

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 4))
    return false;

  return true;
}

/*
    Emulate below MIPS branch instructions.
    BLTZL, BGEZL, BGTZL, BLEZL : Branch likely
    BLTZ, BGEZ, BGTZ, BLEZ     : Non-compact branches
*/
bool EmulateInstructionMIPS64::Emulate_BXX_2ops(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs;
  int64_t offset, pc, rs_val, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BLTZL") ||
      op_name.equals_insensitive("BLTZ") ||
      op_name.equals_insensitive("BLTZ64")) {
    if (rs_val < 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BGEZL") ||
             op_name.equals_insensitive("BGEZ") ||
             op_name.equals_insensitive("BGEZ64")) {
    if (rs_val >= 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BGTZL") ||
             op_name.equals_insensitive("BGTZ") ||
             op_name.equals_insensitive("BGTZ64")) {
    if (rs_val > 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BLEZL") ||
             op_name.equals_insensitive("BLEZ") ||
             op_name.equals_insensitive("BLEZ64")) {
    if (rs_val <= 0)
      target = pc + offset;
    else
      target = pc + 8;
  }

  Context context;
  context.type = eContextRelativeBranchImmediate;
  context.SetImmediate(offset);

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_BC(llvm::MCInst &insn) {
  bool success = false;
  int64_t offset, pc, target;

  /*
   * BC offset
   *      offset = sign_ext (offset << 2)
   *      PC = PC + 4 + offset
  */
  offset = insn.getOperand(0).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  target = pc + offset;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

static int IsAdd64bitOverflow(int64_t a, int64_t b) {
  int64_t r = (uint64_t)a + (uint64_t)b;
  return (a < 0 && b < 0 && r >= 0) || (a >= 0 && b >= 0 && r < 0);
}

/*
    Emulate below MIPS branch instructions.
    BEQC, BNEC, BLTC, BGEC, BLTUC, BGEUC, BOVC, BNVC: Compact branch
   instructions with no delay slot
*/
bool EmulateInstructionMIPS64::Emulate_BXX_3ops_C(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs, rt;
  int64_t offset, pc, rs_val, rt_val, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());
  uint32_t current_inst_size = m_insn_info->get(insn.getOpcode()).getSize();

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  rt = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());
  offset = insn.getOperand(2).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  rt_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rt, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BEQC") ||
      op_name.equals_insensitive("BEQC64")) {
    if (rs_val == rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BNEC") ||
             op_name.equals_insensitive("BNEC64")) {
    if (rs_val != rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BLTC") ||
             op_name.equals_insensitive("BLTC64")) {
    if (rs_val < rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGEC64") ||
             op_name.equals_insensitive("BGEC")) {
    if (rs_val >= rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BLTUC") ||
             op_name.equals_insensitive("BLTUC64")) {
    if (rs_val < rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGEUC") ||
             op_name.equals_insensitive("BGEUC64")) {
    if ((uint32_t)rs_val >= (uint32_t)rt_val)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BOVC")) {
    if (IsAdd64bitOverflow(rs_val, rt_val))
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BNVC")) {
    if (!IsAdd64bitOverflow(rs_val, rt_val))
      target = pc + offset;
    else
      target = pc + 4;
  }

  Context context;
  context.type = eContextRelativeBranchImmediate;
  context.SetImmediate(current_inst_size + offset);

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

/*
    Emulate below MIPS branch instructions.
    BLTZC, BLEZC, BGEZC, BGTZC, BEQZC, BNEZC : Compact Branches
*/
bool EmulateInstructionMIPS64::Emulate_BXX_2ops_C(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs;
  int64_t offset, pc, target = 0;
  int64_t rs_val;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());
  uint32_t current_inst_size = m_insn_info->get(insn.getOpcode()).getSize();

  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rs, 0, &success);
  if (!success)
    return false;

  if (op_name.equals_insensitive("BLTZC") ||
      op_name.equals_insensitive("BLTZC64")) {
    if (rs_val < 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BLEZC") ||
             op_name.equals_insensitive("BLEZC64")) {
    if (rs_val <= 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGEZC") ||
             op_name.equals_insensitive("BGEZC64")) {
    if (rs_val >= 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BGTZC") ||
             op_name.equals_insensitive("BGTZC64")) {
    if (rs_val > 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BEQZC") ||
             op_name.equals_insensitive("BEQZC64")) {
    if (rs_val == 0)
      target = pc + offset;
    else
      target = pc + 4;
  } else if (op_name.equals_insensitive("BNEZC") ||
             op_name.equals_insensitive("BNEZC64")) {
    if (rs_val != 0)
      target = pc + offset;
    else
      target = pc + 4;
  }

  Context context;
  context.type = eContextRelativeBranchImmediate;
  context.SetImmediate(current_inst_size + offset);

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_J(llvm::MCInst &insn) {
  bool success = false;
  uint64_t offset, pc;

  /*
   * J offset
   *      offset = sign_ext (offset << 2)
   *      PC = PC[63-28] | offset
  */
  offset = insn.getOperand(0).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  /* This is a PC-region branch and not PC-relative */
  pc = (pc & 0xFFFFFFFFF0000000ULL) | offset;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               pc);
}

bool EmulateInstructionMIPS64::Emulate_JAL(llvm::MCInst &insn) {
  bool success = false;
  uint64_t offset, target, pc;

  /*
   * JAL offset
   *      offset = sign_ext (offset << 2)
   *      PC = PC[63-28] | offset
  */
  offset = insn.getOperand(0).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  /* This is a PC-region branch and not PC-relative */
  target = (pc & 0xFFFFFFFFF0000000ULL) | offset;

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 8))
    return false;

  return true;
}

bool EmulateInstructionMIPS64::Emulate_JALR(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs, rt;
  uint64_t pc, rs_val;

  /*
   * JALR rt, rs
   *      GPR[rt] = PC + 8
   *      PC = GPR[rs]
  */
  rt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  rs = m_reg_info->getEncodingValue(insn.getOperand(1).getReg());

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rs_val = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + rs, 0,
                                &success);
  if (!success)
    return false;

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             rs_val))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF,
                             registers::mips64::dwarf::zero + rt, pc + 8))
    return false;

  return true;
}

bool EmulateInstructionMIPS64::Emulate_JIALC(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rt;
  int64_t target, offset, pc, rt_val;

  /*
   * JIALC rt, offset
   *      offset = sign_ext (offset)
   *      PC = GPR[rt] + offset
   *      RA = PC + 4
  */
  rt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  rt_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rt, 0, &success);
  if (!success)
    return false;

  target = rt_val + offset;

  Context context;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                             target))
    return false;

  if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::ra,
                             pc + 4))
    return false;

  return true;
}

bool EmulateInstructionMIPS64::Emulate_JIC(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rt;
  int64_t target, offset, rt_val;

  /*
   * JIC rt, offset
   *      offset = sign_ext (offset)
   *      PC = GPR[rt] + offset
  */
  rt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  rt_val = (int64_t)ReadRegisterUnsigned(eRegisterKindDWARF,
                                         registers::mips64::dwarf::zero + rt, 0, &success);
  if (!success)
    return false;

  target = rt_val + offset;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_JR(llvm::MCInst &insn) {
  bool success = false;
  uint32_t rs;
  uint64_t rs_val;

  /*
   * JR rs
   *      PC = GPR[rs]
  */
  rs = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());

  rs_val = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + rs, 0,
                                &success);
  if (!success)
    return false;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               rs_val);
}

/*
    Emulate Branch on FP True/False
    BC1F, BC1FL :   Branch on FP False (L stands for branch likely)
    BC1T, BC1TL :   Branch on FP True  (L stands for branch likely)
*/
bool EmulateInstructionMIPS64::Emulate_FP_branch(llvm::MCInst &insn) {
  bool success = false;
  uint32_t cc, fcsr;
  int64_t pc, offset, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  /*
   * BC1F cc, offset
   *  condition <- (FPConditionCode(cc) == 0)
   *      if condition then
   *          offset = sign_ext (offset)
   *          PC = PC + offset
  */
  cc = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  fcsr =
      ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::fcsr, 0, &success);
  if (!success)
    return false;

  /* fcsr[23], fcsr[25-31] are vaild condition bits */
  fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

  if (op_name.equals_insensitive("BC1F") ||
      op_name.equals_insensitive("BC1FL")) {
    if ((fcsr & (1 << cc)) == 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BC1T") ||
             op_name.equals_insensitive("BC1TL")) {
    if ((fcsr & (1 << cc)) != 0)
      target = pc + offset;
    else
      target = pc + 8;
  }

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_BC1EQZ(llvm::MCInst &insn) {
  bool success = false;
  uint32_t ft;
  uint64_t ft_val;
  int64_t target, pc, offset;

  /*
   * BC1EQZ ft, offset
   *  condition <- (FPR[ft].bit0 == 0)
   *      if condition then
   *          offset = sign_ext (offset)
   *          PC = PC + 4 + offset
  */
  ft = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  ft_val = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + ft, 0,
                                &success);
  if (!success)
    return false;

  if ((ft_val & 1) == 0)
    target = pc + 4 + offset;
  else
    target = pc + 8;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_BC1NEZ(llvm::MCInst &insn) {
  bool success = false;
  uint32_t ft;
  uint64_t ft_val;
  int64_t target, pc, offset;

  /*
   * BC1NEZ ft, offset
   *  condition <- (FPR[ft].bit0 != 0)
   *      if condition then
   *          offset = sign_ext (offset)
   *          PC = PC + 4 + offset
  */
  ft = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  ft_val = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + ft, 0,
                                &success);
  if (!success)
    return false;

  if ((ft_val & 1) != 0)
    target = pc + 4 + offset;
  else
    target = pc + 8;

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

/*
    Emulate MIPS-3D Branch instructions
    BC1ANY2F, BC1ANY2T  : Branch on Any of Two Floating Point Condition Codes
   False/True
    BC1ANY4F, BC1ANY4T  : Branch on Any of Four Floating Point Condition Codes
   False/True
*/
bool EmulateInstructionMIPS64::Emulate_3D_branch(llvm::MCInst &insn) {
  bool success = false;
  uint32_t cc, fcsr;
  int64_t pc, offset, target = 0;
  llvm::StringRef op_name = m_insn_info->getName(insn.getOpcode());

  cc = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  offset = insn.getOperand(1).getImm();

  pc = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0,
                            &success);
  if (!success)
    return false;

  fcsr = (uint32_t)ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::fcsr,
                                        0, &success);
  if (!success)
    return false;

  /* fcsr[23], fcsr[25-31] are vaild condition bits */
  fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

  if (op_name.equals_insensitive("BC1ANY2F")) {
    /* if any one bit is 0 */
    if (((fcsr >> cc) & 3) != 3)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BC1ANY2T")) {
    /* if any one bit is 1 */
    if (((fcsr >> cc) & 3) != 0)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BC1ANY4F")) {
    /* if any one bit is 0 */
    if (((fcsr >> cc) & 0xf) != 0xf)
      target = pc + offset;
    else
      target = pc + 8;
  } else if (op_name.equals_insensitive("BC1ANY4T")) {
    /* if any one bit is 1 */
    if (((fcsr >> cc) & 0xf) != 0)
      target = pc + offset;
    else
      target = pc + 8;
  }

  Context context;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_BNZB(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 1, true);
}

bool EmulateInstructionMIPS64::Emulate_BNZH(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 2, true);
}

bool EmulateInstructionMIPS64::Emulate_BNZW(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 4, true);
}

bool EmulateInstructionMIPS64::Emulate_BNZD(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 8, true);
}

bool EmulateInstructionMIPS64::Emulate_BZB(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 1, false);
}

bool EmulateInstructionMIPS64::Emulate_BZH(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 2, false);
}

bool EmulateInstructionMIPS64::Emulate_BZW(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 4, false);
}

bool EmulateInstructionMIPS64::Emulate_BZD(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_DF(insn, 8, false);
}

bool EmulateInstructionMIPS64::Emulate_MSA_Branch_DF(llvm::MCInst &insn,
                                                     int element_byte_size,
                                                     bool bnz) {
  bool success = false, branch_hit = true;
  int64_t target = 0;
  RegisterValue reg_value;
  const uint8_t *ptr = nullptr;

  uint32_t wt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  int64_t offset = insn.getOperand(1).getImm();

  int64_t pc =
      ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0, &success);
  if (!success)
    return false;

  if (ReadRegister(eRegisterKindDWARF, registers::mips64::dwarf::w0 + wt,
                   reg_value))
    ptr = (const uint8_t *)reg_value.GetBytes();
  else
    return false;

  for (int i = 0; i < 16 / element_byte_size; i++) {
    switch (element_byte_size) {
    case 1:
      if ((*ptr == 0 && bnz) || (*ptr != 0 && !bnz))
        branch_hit = false;
      break;
    case 2:
      if ((*(const uint16_t *)ptr == 0 && bnz) ||
          (*(const uint16_t *)ptr != 0 && !bnz))
        branch_hit = false;
      break;
    case 4:
      if ((*(const uint32_t *)ptr == 0 && bnz) ||
          (*(const uint32_t *)ptr != 0 && !bnz))
        branch_hit = false;
      break;
    case 8:
      if ((*(const uint64_t *)ptr == 0 && bnz) ||
          (*(const uint64_t *)ptr != 0 && !bnz))
        branch_hit = false;
      break;
    }
    if (!branch_hit)
      break;
    ptr = ptr + element_byte_size;
  }

  if (branch_hit)
    target = pc + offset;
  else
    target = pc + 8;

  Context context;
  context.type = eContextRelativeBranchImmediate;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_BNZV(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_V(insn, true);
}

bool EmulateInstructionMIPS64::Emulate_BZV(llvm::MCInst &insn) {
  return Emulate_MSA_Branch_V(insn, false);
}

bool EmulateInstructionMIPS64::Emulate_MSA_Branch_V(llvm::MCInst &insn,
                                                    bool bnz) {
  bool success = false;
  int64_t target = 0;
  llvm::APInt wr_val = llvm::APInt::getZero(128);
  llvm::APInt fail_value = llvm::APInt::getMaxValue(128);
  llvm::APInt zero_value = llvm::APInt::getZero(128);
  RegisterValue reg_value;

  uint32_t wt = m_reg_info->getEncodingValue(insn.getOperand(0).getReg());
  int64_t offset = insn.getOperand(1).getImm();

  int64_t pc =
      ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::pc, 0, &success);
  if (!success)
    return false;

  if (ReadRegister(eRegisterKindDWARF, registers::mips64::dwarf::w0 + wt,
                   reg_value))
    wr_val = reg_value.GetAsUInt128(fail_value);
  else
    return false;

  if ((llvm::APInt::isSameValue(zero_value, wr_val) && !bnz) ||
      (!llvm::APInt::isSameValue(zero_value, wr_val) && bnz))
    target = pc + offset;
  else
    target = pc + 8;

  Context context;
  context.type = eContextRelativeBranchImmediate;

  return WriteRegisterUnsigned(context, eRegisterKindDWARF, registers::mips64::dwarf::pc,
                               target);
}

bool EmulateInstructionMIPS64::Emulate_LDST_Imm(llvm::MCInst &insn) {
  bool success = false;
  uint32_t base;
  int64_t imm, address;
  Context bad_vaddr_context;

  uint32_t num_operands = insn.getNumOperands();
  base =
      m_reg_info->getEncodingValue(insn.getOperand(num_operands - 2).getReg());
  imm = insn.getOperand(num_operands - 1).getImm();

  if (!GetRegisterInfo(eRegisterKindDWARF,
                       registers::mips64::dwarf::zero + base))
    return false;

  /* read base register */
  address = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + base, 0,
                                 &success);
  if (!success)
    return false;

  /* destination address */
  address = address + imm;

  /* Set the bad_vaddr register with base address used in the instruction */
  bad_vaddr_context.type = eContextInvalid;
  WriteRegisterUnsigned(bad_vaddr_context, eRegisterKindDWARF, registers::mips64::dwarf::bad,
                        address);

  return true;
}

bool EmulateInstructionMIPS64::Emulate_LDST_Reg(llvm::MCInst &insn) {
  bool success = false;
  uint32_t base, index;
  int64_t address, index_address;
  Context bad_vaddr_context;

  uint32_t num_operands = insn.getNumOperands();
  base =
      m_reg_info->getEncodingValue(insn.getOperand(num_operands - 2).getReg());
  index =
      m_reg_info->getEncodingValue(insn.getOperand(num_operands - 1).getReg());

  if (!GetRegisterInfo(eRegisterKindDWARF,
                       registers::mips64::dwarf::zero + base))
    return false;

  if (!GetRegisterInfo(eRegisterKindDWARF,
                       registers::mips64::dwarf::zero + index))
    return false;

  /* read base register */
  address = ReadRegisterUnsigned(eRegisterKindDWARF, registers::mips64::dwarf::zero + base, 0,
                                 &success);
  if (!success)
    return false;

  /* read index register */
  index_address = ReadRegisterUnsigned(eRegisterKindDWARF,
                                       registers::mips64::dwarf::zero + index, 0, &success);
  if (!success)
    return false;

  /* destination address */
  address = address + index_address;

  /* Set the bad_vaddr register with base address used in the instruction */
  bad_vaddr_context.type = eContextInvalid;
  WriteRegisterUnsigned(bad_vaddr_context, eRegisterKindDWARF, registers::mips64::dwarf::bad,
                        address);

  return true;
}
