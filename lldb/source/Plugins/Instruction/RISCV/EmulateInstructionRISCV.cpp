//===-- EmulateInstructionRISCV.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionRISCV.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"
#include "Plugins/Process/Utility/lldb-riscv-register-enums.h"
#include "RISCVCInstructions.h"
#include "RISCVInstructions.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/OptionValueArray.h"
#include "lldb/Interpreter/OptionValueDictionary.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(EmulateInstructionRISCV, InstructionRISCV)

namespace lldb_private {

/// Returns all values wrapped in Optional, or None if any of the values is
/// None.
template <typename... Ts>
static llvm::Optional<std::tuple<Ts...>> zipOpt(llvm::Optional<Ts> &&...ts) {
  if ((ts.has_value() && ...))
    return llvm::Optional<std::tuple<Ts...>>(
        std::make_tuple(std::move(*ts)...));
  else
    return llvm::None;
}

// The funct3 is the type of compare in B<CMP> instructions.
// funct3 means "3-bits function selector", which RISC-V ISA uses as minor
// opcode. It reuses the major opcode encoding space.
constexpr uint32_t BEQ = 0b000;
constexpr uint32_t BNE = 0b001;
constexpr uint32_t BLT = 0b100;
constexpr uint32_t BGE = 0b101;
constexpr uint32_t BLTU = 0b110;
constexpr uint32_t BGEU = 0b111;

// used in decoder
constexpr int32_t SignExt(uint32_t imm) { return int32_t(imm); }

// used in executor
template <typename T>
constexpr std::enable_if_t<sizeof(T) <= 4, uint64_t> SextW(T value) {
  return uint64_t(int64_t(int32_t(value)));
}

// used in executor
template <typename T> constexpr uint64_t ZextD(T value) {
  return uint64_t(value);
}

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

constexpr uint32_t DecodeSImm(uint32_t inst) {
  return (uint64_t(int64_t(int32_t(inst & 0xFE000000)) >> 20)) // imm[11:5]
         | ((inst & 0xF80) >> 7);                              // imm[4:0]
}

constexpr uint32_t DecodeUImm(uint32_t inst) {
  return SextW(inst & 0xFFFFF000); // imm[31:12]
}

static uint32_t GPREncodingToLLDB(uint32_t reg_encode) {
  if (reg_encode == 0)
    return gpr_x0_riscv;
  if (reg_encode >= 1 && reg_encode <= 31)
    return gpr_x1_riscv + reg_encode - 1;
  return LLDB_INVALID_REGNUM;
}

bool Rd::Write(EmulateInstructionRISCV &emulator, uint64_t value) {
  uint32_t lldb_reg = GPREncodingToLLDB(rd);
  EmulateInstruction::Context ctx;
  ctx.type = EmulateInstruction::eContextRegisterStore;
  ctx.SetNoArgs();
  RegisterValue registerValue;
  registerValue.SetUInt64(value);
  return emulator.WriteRegister(ctx, eRegisterKindLLDB, lldb_reg,
                                registerValue);
}

llvm::Optional<uint64_t> Rs::Read(EmulateInstructionRISCV &emulator) {
  uint32_t lldbReg = GPREncodingToLLDB(rs);
  RegisterValue value;
  return emulator.ReadRegister(eRegisterKindLLDB, lldbReg, value)
             ? llvm::Optional<uint64_t>(value.GetAsUInt64())
             : llvm::None;
}

llvm::Optional<int32_t> Rs::ReadI32(EmulateInstructionRISCV &emulator) {
  return Read(emulator).transform(
      [](uint64_t value) { return int32_t(uint32_t(value)); });
}

llvm::Optional<int64_t> Rs::ReadI64(EmulateInstructionRISCV &emulator) {
  return Read(emulator).transform(
      [](uint64_t value) { return int64_t(value); });
}

llvm::Optional<uint32_t> Rs::ReadU32(EmulateInstructionRISCV &emulator) {
  return Read(emulator).transform(
      [](uint64_t value) { return uint32_t(value); });
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

template <typename T>
constexpr bool is_load =
    std::is_same_v<T, LB> || std::is_same_v<T, LH> || std::is_same_v<T, LW> ||
    std::is_same_v<T, LD> || std::is_same_v<T, LBU> || std::is_same_v<T, LHU> ||
    std::is_same_v<T, LWU>;

template <typename T>
constexpr bool is_store = std::is_same_v<T, SB> || std::is_same_v<T, SH> ||
                          std::is_same_v<T, SW> || std::is_same_v<T, SD>;

template <typename T>
constexpr bool is_amo_add =
    std::is_same_v<T, AMOADD_W> || std::is_same_v<T, AMOADD_D>;

template <typename T>
constexpr bool is_amo_bit_op =
    std::is_same_v<T, AMOXOR_W> || std::is_same_v<T, AMOXOR_D> ||
    std::is_same_v<T, AMOAND_W> || std::is_same_v<T, AMOAND_D> ||
    std::is_same_v<T, AMOOR_W> || std::is_same_v<T, AMOOR_D>;

template <typename T>
constexpr bool is_amo_swap =
    std::is_same_v<T, AMOSWAP_W> || std::is_same_v<T, AMOSWAP_D>;

template <typename T>
constexpr bool is_amo_cmp =
    std::is_same_v<T, AMOMIN_W> || std::is_same_v<T, AMOMIN_D> ||
    std::is_same_v<T, AMOMAX_W> || std::is_same_v<T, AMOMAX_D> ||
    std::is_same_v<T, AMOMINU_W> || std::is_same_v<T, AMOMINU_D> ||
    std::is_same_v<T, AMOMAXU_W> || std::is_same_v<T, AMOMAXU_D>;

template <typename I>
static std::enable_if_t<is_load<I> || is_store<I>, llvm::Optional<uint64_t>>
LoadStoreAddr(EmulateInstructionRISCV &emulator, I inst) {
  return inst.rs1.Read(emulator).transform(
      [&](uint64_t rs1) { return rs1 + uint64_t(SignExt(inst.imm)); });
}

// Read T from memory, then load its sign-extended value m_emu to register.
template <typename I, typename T, typename E>
static std::enable_if_t<is_load<I>, bool>
Load(EmulateInstructionRISCV &emulator, I inst, uint64_t (*extend)(E)) {
  auto addr = LoadStoreAddr(emulator, inst);
  if (!addr)
    return false;
  return emulator.ReadMem<T>(*addr)
      .transform([&](T t) { return inst.rd.Write(emulator, extend(E(t))); })
      .value_or(false);
}

template <typename I, typename T>
static std::enable_if_t<is_store<I>, bool>
Store(EmulateInstructionRISCV &emulator, I inst) {
  auto addr = LoadStoreAddr(emulator, inst);
  if (!addr)
    return false;
  return inst.rs2.Read(emulator)
      .transform([&](uint64_t rs2) { return emulator.WriteMem<T>(*addr, rs2); })
      .value_or(false);
}

template <typename I>
static std::enable_if_t<is_amo_add<I> || is_amo_bit_op<I> || is_amo_swap<I> ||
                            is_amo_cmp<I>,
                        llvm::Optional<uint64_t>>
AtomicAddr(EmulateInstructionRISCV &emulator, I inst, unsigned int align) {
  return inst.rs1.Read(emulator)
      .transform([&](uint64_t rs1) {
        return rs1 % align == 0 ? llvm::Optional<uint64_t>(rs1) : llvm::None;
      })
      .value_or(llvm::None);
}

template <typename I, typename T>
static std::enable_if_t<is_amo_swap<I>, bool>
AtomicSwap(EmulateInstructionRISCV &emulator, I inst, int align,
           uint64_t (*extend)(T)) {
  auto addr = AtomicAddr(emulator, inst, align);
  if (!addr)
    return false;
  return zipOpt(emulator.ReadMem<T>(*addr), inst.rs2.Read(emulator))
      .transform([&](auto &&tup) {
        auto [tmp, rs2] = tup;
        return emulator.WriteMem<T>(*addr, T(rs2)) &&
               inst.rd.Write(emulator, extend(tmp));
      })
      .value_or(false);
}

template <typename I, typename T>
static std::enable_if_t<is_amo_add<I>, bool>
AtomicADD(EmulateInstructionRISCV &emulator, I inst, int align,
          uint64_t (*extend)(T)) {
  auto addr = AtomicAddr(emulator, inst, align);
  if (!addr)
    return false;
  return zipOpt(emulator.ReadMem<T>(*addr), inst.rs2.Read(emulator))
      .transform([&](auto &&tup) {
        auto [tmp, rs2] = tup;
        return emulator.WriteMem<T>(*addr, T(tmp + rs2)) &&
               inst.rd.Write(emulator, extend(tmp));
      })
      .value_or(false);
}

template <typename I, typename T>
static std::enable_if_t<is_amo_bit_op<I>, bool>
AtomicBitOperate(EmulateInstructionRISCV &emulator, I inst, int align,
                 uint64_t (*extend)(T), T (*operate)(T, T)) {
  auto addr = AtomicAddr(emulator, inst, align);
  if (!addr)
    return false;
  return zipOpt(emulator.ReadMem<T>(*addr), inst.rs2.Read(emulator))
      .transform([&](auto &&tup) {
        auto [value, rs2] = tup;
        return emulator.WriteMem<T>(*addr, operate(value, T(rs2))) &&
               inst.rd.Write(emulator, extend(value));
      })
      .value_or(false);
}

template <typename I, typename T>
static std::enable_if_t<is_amo_cmp<I>, bool>
AtomicCmp(EmulateInstructionRISCV &emulator, I inst, int align,
          uint64_t (*extend)(T), T (*cmp)(T, T)) {
  auto addr = AtomicAddr(emulator, inst, align);
  if (!addr)
    return false;
  return zipOpt(emulator.ReadMem<T>(*addr), inst.rs2.Read(emulator))
      .transform([&](auto &&tup) {
        auto [value, rs2] = tup;
        return emulator.WriteMem<T>(*addr, cmp(value, T(rs2))) &&
               inst.rd.Write(emulator, extend(value));
      })
      .value_or(false);
}

bool AtomicSequence(EmulateInstructionRISCV &emulator) {
  // The atomic sequence is always 4 instructions long:
  // example:
  //   110cc:	100427af          	lr.w	a5,(s0)
  //   110d0:	00079663          	bnez	a5,110dc
  //   110d4:	1ce426af          	sc.w.aq	a3,a4,(s0)
  //   110d8:	fe069ae3          	bnez	a3,110cc
  //   110dc:   ........          	<next instruction>
  const auto pc = emulator.ReadPC();
  if (!pc)
    return false;
  auto current_pc = pc.value();
  const auto entry_pc = current_pc;

  // The first instruction should be LR.W or LR.D
  auto inst = emulator.ReadInstructionAt(current_pc);
  if (!inst || (!std::holds_alternative<LR_W>(inst->decoded) &&
                !std::holds_alternative<LR_D>(inst->decoded)))
    return false;

  // The second instruction should be BNE to exit address
  inst = emulator.ReadInstructionAt(current_pc += 4);
  if (!inst || !std::holds_alternative<B>(inst->decoded))
    return false;
  auto bne_exit = std::get<B>(inst->decoded);
  if (bne_exit.funct3 != BNE)
    return false;
  // save the exit address to check later
  const auto exit_pc = current_pc + SextW(bne_exit.imm);

  // The third instruction should be SC.W or SC.D
  inst = emulator.ReadInstructionAt(current_pc += 4);
  if (!inst || (!std::holds_alternative<SC_W>(inst->decoded) &&
                !std::holds_alternative<SC_D>(inst->decoded)))
    return false;

  // The fourth instruction should be BNE to entry address
  inst = emulator.ReadInstructionAt(current_pc += 4);
  if (!inst || !std::holds_alternative<B>(inst->decoded))
    return false;
  auto bne_start = std::get<B>(inst->decoded);
  if (bne_start.funct3 != BNE)
    return false;
  if (entry_pc != current_pc + SextW(bne_start.imm))
    return false;

  current_pc += 4;
  // check the exit address and jump to it
  return exit_pc == current_pc && emulator.WritePC(current_pc);
}

template <typename T> static RISCVInst DecodeUType(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, DecodeUImm(inst)};
}

template <typename T> static RISCVInst DecodeJType(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, DecodeJImm(inst)};
}

template <typename T> static RISCVInst DecodeIType(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, Rs{DecodeRS1(inst)}, DecodeIImm(inst)};
}

template <typename T> static RISCVInst DecodeBType(uint32_t inst) {
  return T{Rs{DecodeRS1(inst)}, Rs{DecodeRS2(inst)}, DecodeBImm(inst),
           (inst & 0x7000) >> 12};
}

template <typename T> static RISCVInst DecodeSType(uint32_t inst) {
  return T{Rs{DecodeRS1(inst)}, Rs{DecodeRS2(inst)}, DecodeSImm(inst)};
}

template <typename T> static RISCVInst DecodeRType(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, Rs{DecodeRS1(inst)}, Rs{DecodeRS2(inst)}};
}

template <typename T> static RISCVInst DecodeRShamtType(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, Rs{DecodeRS1(inst)}, DecodeRS2(inst)};
}

template <typename T> static RISCVInst DecodeRRS1Type(uint32_t inst) {
  return T{Rd{DecodeRD(inst)}, Rs{DecodeRS1(inst)}};
}

static const InstrPattern PATTERNS[] = {
    // RV32I & RV64I (The base integer ISA) //
    {"LUI", 0x7F, 0x37, DecodeUType<LUI>},
    {"AUIPC", 0x7F, 0x17, DecodeUType<AUIPC>},
    {"JAL", 0x7F, 0x6F, DecodeJType<JAL>},
    {"JALR", 0x707F, 0x67, DecodeIType<JALR>},
    {"B", 0x7F, 0x63, DecodeBType<B>},
    {"LB", 0x707F, 0x3, DecodeIType<LB>},
    {"LH", 0x707F, 0x1003, DecodeIType<LH>},
    {"LW", 0x707F, 0x2003, DecodeIType<LW>},
    {"LBU", 0x707F, 0x4003, DecodeIType<LBU>},
    {"LHU", 0x707F, 0x5003, DecodeIType<LHU>},
    {"SB", 0x707F, 0x23, DecodeSType<SB>},
    {"SH", 0x707F, 0x1023, DecodeSType<SH>},
    {"SW", 0x707F, 0x2023, DecodeSType<SW>},
    {"ADDI", 0x707F, 0x13, DecodeIType<ADDI>},
    {"SLTI", 0x707F, 0x2013, DecodeIType<SLTI>},
    {"SLTIU", 0x707F, 0x3013, DecodeIType<SLTIU>},
    {"XORI", 0x707F, 0x4013, DecodeIType<XORI>},
    {"ORI", 0x707F, 0x6013, DecodeIType<ORI>},
    {"ANDI", 0x707F, 0x7013, DecodeIType<ANDI>},
    {"SLLI", 0xF800707F, 0x1013, DecodeRShamtType<SLLI>},
    {"SRLI", 0xF800707F, 0x5013, DecodeRShamtType<SRLI>},
    {"SRAI", 0xF800707F, 0x40005013, DecodeRShamtType<SRAI>},
    {"ADD", 0xFE00707F, 0x33, DecodeRType<ADD>},
    {"SUB", 0xFE00707F, 0x40000033, DecodeRType<SUB>},
    {"SLL", 0xFE00707F, 0x1033, DecodeRType<SLL>},
    {"SLT", 0xFE00707F, 0x2033, DecodeRType<SLT>},
    {"SLTU", 0xFE00707F, 0x3033, DecodeRType<SLTU>},
    {"XOR", 0xFE00707F, 0x4033, DecodeRType<XOR>},
    {"SRL", 0xFE00707F, 0x5033, DecodeRType<SRL>},
    {"SRA", 0xFE00707F, 0x40005033, DecodeRType<SRA>},
    {"OR", 0xFE00707F, 0x6033, DecodeRType<OR>},
    {"AND", 0xFE00707F, 0x7033, DecodeRType<AND>},
    {"LWU", 0x707F, 0x6003, DecodeIType<LWU>},
    {"LD", 0x707F, 0x3003, DecodeIType<LD>},
    {"SD", 0x707F, 0x3023, DecodeSType<SD>},
    {"ADDIW", 0x707F, 0x1B, DecodeIType<ADDIW>},
    {"SLLIW", 0xFE00707F, 0x101B, DecodeRShamtType<SLLIW>},
    {"SRLIW", 0xFE00707F, 0x501B, DecodeRShamtType<SRLIW>},
    {"SRAIW", 0xFE00707F, 0x4000501B, DecodeRShamtType<SRAIW>},
    {"ADDW", 0xFE00707F, 0x3B, DecodeRType<ADDW>},
    {"SUBW", 0xFE00707F, 0x4000003B, DecodeRType<SUBW>},
    {"SLLW", 0xFE00707F, 0x103B, DecodeRType<SLLW>},
    {"SRLW", 0xFE00707F, 0x503B, DecodeRType<SRLW>},
    {"SRAW", 0xFE00707F, 0x4000503B, DecodeRType<SRAW>},

    // RV32M & RV64M (The integer multiplication and division extension) //
    {"MUL", 0xFE00707F, 0x2000033, DecodeRType<MUL>},
    {"MULH", 0xFE00707F, 0x2001033, DecodeRType<MULH>},
    {"MULHSU", 0xFE00707F, 0x2002033, DecodeRType<MULHSU>},
    {"MULHU", 0xFE00707F, 0x2003033, DecodeRType<MULHU>},
    {"DIV", 0xFE00707F, 0x2004033, DecodeRType<DIV>},
    {"DIVU", 0xFE00707F, 0x2005033, DecodeRType<DIVU>},
    {"REM", 0xFE00707F, 0x2006033, DecodeRType<REM>},
    {"REMU", 0xFE00707F, 0x2007033, DecodeRType<REMU>},
    {"MULW", 0xFE00707F, 0x200003B, DecodeRType<MULW>},
    {"DIVW", 0xFE00707F, 0x200403B, DecodeRType<DIVW>},
    {"DIVUW", 0xFE00707F, 0x200503B, DecodeRType<DIVUW>},
    {"REMW", 0xFE00707F, 0x200603B, DecodeRType<REMW>},
    {"REMUW", 0xFE00707F, 0x200703B, DecodeRType<REMUW>},

    // RV32A & RV64A (The standard atomic instruction extension) //
    {"LR_W", 0xF9F0707F, 0x1000202F, DecodeRRS1Type<LR_W>},
    {"LR_D", 0xF9F0707F, 0x1000302F, DecodeRRS1Type<LR_D>},
    {"SC_W", 0xF800707F, 0x1800202F, DecodeRType<SC_W>},
    {"SC_D", 0xF800707F, 0x1800302F, DecodeRType<SC_D>},
    {"AMOSWAP_W", 0xF800707F, 0x800202F, DecodeRType<AMOSWAP_W>},
    {"AMOADD_W", 0xF800707F, 0x202F, DecodeRType<AMOADD_W>},
    {"AMOXOR_W", 0xF800707F, 0x2000202F, DecodeRType<AMOXOR_W>},
    {"AMOAND_W", 0xF800707F, 0x6000202F, DecodeRType<AMOAND_W>},
    {"AMOOR_W", 0xF800707F, 0x4000202F, DecodeRType<AMOOR_W>},
    {"AMOMIN_W", 0xF800707F, 0x8000202F, DecodeRType<AMOMIN_W>},
    {"AMOMAX_W", 0xF800707F, 0xA000202F, DecodeRType<AMOMAX_W>},
    {"AMOMINU_W", 0xF800707F, 0xC000202F, DecodeRType<AMOMINU_W>},
    {"AMOMAXU_W", 0xF800707F, 0xE000202F, DecodeRType<AMOMAXU_W>},
    {"AMOSWAP_D", 0xF800707F, 0x800302F, DecodeRType<AMOSWAP_D>},
    {"AMOADD_D", 0xF800707F, 0x302F, DecodeRType<AMOADD_D>},
    {"AMOXOR_D", 0xF800707F, 0x2000302F, DecodeRType<AMOXOR_D>},
    {"AMOAND_D", 0xF800707F, 0x6000302F, DecodeRType<AMOAND_D>},
    {"AMOOR_D", 0xF800707F, 0x4000302F, DecodeRType<AMOOR_D>},
    {"AMOMIN_D", 0xF800707F, 0x8000302F, DecodeRType<AMOMIN_D>},
    {"AMOMAX_D", 0xF800707F, 0xA000302F, DecodeRType<AMOMAX_D>},
    {"AMOMINU_D", 0xF800707F, 0xC000302F, DecodeRType<AMOMINU_D>},
    {"AMOMAXU_D", 0xF800707F, 0xE000302F, DecodeRType<AMOMAXU_D>},

    // RVC (Compressed Instructions) //
    {"C_LWSP", 0xE003, 0x4002, DecodeC_LWSP},
    {"C_LDSP", 0xE003, 0x6002, DecodeC_LDSP},
    {"C_SWSP", 0xE003, 0xC002, DecodeC_SWSP},
    {"C_SDSP", 0xE003, 0xE002, DecodeC_SDSP},
    {"C_LW", 0xE003, 0x4000, DecodeC_LW},
    {"C_LD", 0xE003, 0x6000, DecodeC_LD},
    {"C_SW", 0xE003, 0xC000, DecodeC_SW},
    {"C_SD", 0xE003, 0xE000, DecodeC_SD},
    {"C_J", 0xE003, 0xA001, DecodeC_J},
    {"C_JR", 0xF07F, 0x8002, DecodeC_JR},
    {"C_JALR", 0xF07F, 0x9002, DecodeC_JALR},
    {"C_BNEZ", 0xE003, 0xE001, DecodeC_BNEZ},
    {"C_BEQZ", 0xE003, 0xC001, DecodeC_BEQZ},
    {"C_LI", 0xE003, 0x4001, DecodeC_LI},
    {"C_LUI_ADDI16SP", 0xE003, 0x6001, DecodeC_LUI_ADDI16SP},
    {"C_ADDI", 0xE003, 0x1, DecodeC_ADDI},
    {"C_ADDIW", 0xE003, 0x2001, DecodeC_ADDIW},
    {"C_ADDI4SPN", 0xE003, 0x0, DecodeC_ADDI4SPN},
    {"C_SLLI", 0xE003, 0x2, DecodeC_SLLI},
    {"C_SRLI", 0xEC03, 0x8001, DecodeC_SRLI},
    {"C_SRAI", 0xEC03, 0x8401, DecodeC_SRAI},
    {"C_ANDI", 0xEC03, 0x8801, DecodeC_ANDI},
    {"C_MV", 0xF003, 0x8002, DecodeC_MV},
    {"C_ADD", 0xF003, 0x9002, DecodeC_ADD},
    {"C_AND", 0xFC63, 0x8C61, DecodeC_AND},
    {"C_OR", 0xFC63, 0x8C41, DecodeC_OR},
    {"C_XOR", 0xFC63, 0x8C21, DecodeC_XOR},
    {"C_SUB", 0xFC63, 0x8C01, DecodeC_SUB},
    {"C_SUBW", 0xFC63, 0x9C01, DecodeC_SUBW},
    {"C_ADDW", 0xFC63, 0x9C21, DecodeC_ADDW},
};

llvm::Optional<DecodeResult> EmulateInstructionRISCV::Decode(uint32_t inst) {
  Log *log = GetLog(LLDBLog::Unwind);

  uint16_t try_rvc = uint16_t(inst & 0x0000ffff);
  // check whether the compressed encode could be valid
  uint16_t mask = try_rvc & 0b11;
  bool is_rvc = try_rvc != 0 && mask != 3;

  for (const InstrPattern &pat : PATTERNS) {
    if ((inst & pat.type_mask) == pat.eigen) {
      LLDB_LOGF(log, "EmulateInstructionRISCV::%s: inst(%x at %llx) was decoded to %s",
                __FUNCTION__, inst, m_addr, pat.name);
      auto decoded = is_rvc ? pat.decode(try_rvc) : pat.decode(inst);
      return DecodeResult{decoded, inst, is_rvc, pat};
    }
  }
  LLDB_LOGF(log, "EmulateInstructionRISCV::%s: inst(0x%x) was unsupported",
            __FUNCTION__, inst);
  return llvm::None;
}

class Executor {
  EmulateInstructionRISCV &m_emu;
  bool m_ignore_cond;
  bool m_is_rvc;

public:
  // also used in EvaluateInstruction()
  static uint64_t size(bool is_rvc) { return is_rvc ? 2 : 4; }

private:
  uint64_t delta() { return size(m_is_rvc); }

public:
  Executor(EmulateInstructionRISCV &emulator, bool ignoreCond, bool is_rvc)
      : m_emu(emulator), m_ignore_cond(ignoreCond), m_is_rvc(is_rvc) {}

  bool operator()(LUI inst) { return inst.rd.Write(m_emu, SignExt(inst.imm)); }
  bool operator()(AUIPC inst) {
    return m_emu.ReadPC()
        .transform([&](uint64_t pc) {
          return inst.rd.Write(m_emu, SignExt(inst.imm) + pc);
        })
        .value_or(false);
  }
  bool operator()(JAL inst) {
    return m_emu.ReadPC()
        .transform([&](uint64_t pc) {
          return inst.rd.Write(m_emu, pc + delta()) &&
                 m_emu.WritePC(SignExt(inst.imm) + pc);
        })
        .value_or(false);
  }
  bool operator()(JALR inst) {
    return zipOpt(m_emu.ReadPC(), inst.rs1.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [pc, rs1] = tup;
          return inst.rd.Write(m_emu, pc + delta()) &&
                 m_emu.WritePC((SignExt(inst.imm) + rs1) & ~1);
        })
        .value_or(false);
  }
  bool operator()(B inst) {
    return zipOpt(m_emu.ReadPC(), inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [pc, rs1, rs2] = tup;
          if (m_ignore_cond || CompareB(rs1, rs2, inst.funct3))
            return m_emu.WritePC(SignExt(inst.imm) + pc);
          return true;
        })
        .value_or(false);
  }
  bool operator()(LB inst) {
    return Load<LB, uint8_t, int8_t>(m_emu, inst, SextW);
  }
  bool operator()(LH inst) {
    return Load<LH, uint16_t, int16_t>(m_emu, inst, SextW);
  }
  bool operator()(LW inst) {
    return Load<LW, uint32_t, int32_t>(m_emu, inst, SextW);
  }
  bool operator()(LBU inst) {
    return Load<LBU, uint8_t, uint8_t>(m_emu, inst, ZextD);
  }
  bool operator()(LHU inst) {
    return Load<LHU, uint16_t, uint16_t>(m_emu, inst, ZextD);
  }
  bool operator()(SB inst) { return Store<SB, uint8_t>(m_emu, inst); }
  bool operator()(SH inst) { return Store<SH, uint16_t>(m_emu, inst); }
  bool operator()(SW inst) { return Store<SW, uint32_t>(m_emu, inst); }
  bool operator()(ADDI inst) {
    return inst.rs1.ReadI64(m_emu)
        .transform([&](int64_t rs1) {
          return inst.rd.Write(m_emu, rs1 + int64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(SLTI inst) {
    return inst.rs1.ReadI64(m_emu)
        .transform([&](int64_t rs1) {
          return inst.rd.Write(m_emu, rs1 < int64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(SLTIU inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 < uint64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(XORI inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 ^ uint64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(ORI inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 | uint64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(ANDI inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 & uint64_t(SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(ADD inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 + rs2);
        })
        .value_or(false);
  }
  bool operator()(SUB inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 - rs2);
        })
        .value_or(false);
  }
  bool operator()(SLL inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 << (rs2 & 0b111111));
        })
        .value_or(false);
  }
  bool operator()(SLT inst) {
    return zipOpt(inst.rs1.ReadI64(m_emu), inst.rs2.ReadI64(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 < rs2);
        })
        .value_or(false);
  }
  bool operator()(SLTU inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 < rs2);
        })
        .value_or(false);
  }
  bool operator()(XOR inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 ^ rs2);
        })
        .value_or(false);
  }
  bool operator()(SRL inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 >> (rs2 & 0b111111));
        })
        .value_or(false);
  }
  bool operator()(SRA inst) {
    return zipOpt(inst.rs1.ReadI64(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 >> (rs2 & 0b111111));
        })
        .value_or(false);
  }
  bool operator()(OR inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 | rs2);
        })
        .value_or(false);
  }
  bool operator()(AND inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 & rs2);
        })
        .value_or(false);
  }
  bool operator()(LWU inst) {
    return Load<LWU, uint32_t, uint32_t>(m_emu, inst, ZextD);
  }
  bool operator()(LD inst) {
    return Load<LD, uint64_t, uint64_t>(m_emu, inst, ZextD);
  }
  bool operator()(SD inst) { return Store<SD, uint64_t>(m_emu, inst); }
  bool operator()(SLLI inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 << inst.shamt);
        })
        .value_or(false);
  }
  bool operator()(SRLI inst) {
    return inst.rs1.Read(m_emu)
        .transform([&](uint64_t rs1) {
          return inst.rd.Write(m_emu, rs1 >> inst.shamt);
        })
        .value_or(false);
  }
  bool operator()(SRAI inst) {
    return inst.rs1.ReadI64(m_emu)
        .transform([&](int64_t rs1) {
          return inst.rd.Write(m_emu, rs1 >> inst.shamt);
        })
        .value_or(false);
  }
  bool operator()(ADDIW inst) {
    return inst.rs1.ReadI32(m_emu)
        .transform([&](int32_t rs1) {
          return inst.rd.Write(m_emu, SextW(rs1 + SignExt(inst.imm)));
        })
        .value_or(false);
  }
  bool operator()(SLLIW inst) {
    return inst.rs1.ReadU32(m_emu)
        .transform([&](uint32_t rs1) {
          return inst.rd.Write(m_emu, SextW(rs1 << inst.shamt));
        })
        .value_or(false);
  }
  bool operator()(SRLIW inst) {
    return inst.rs1.ReadU32(m_emu)
        .transform([&](uint32_t rs1) {
          return inst.rd.Write(m_emu, SextW(rs1 >> inst.shamt));
        })
        .value_or(false);
  }
  bool operator()(SRAIW inst) {
    return inst.rs1.ReadI32(m_emu)
        .transform([&](int32_t rs1) {
          return inst.rd.Write(m_emu, SextW(rs1 >> inst.shamt));
        })
        .value_or(false);
  }
  bool operator()(ADDW inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(uint32_t(rs1 + rs2)));
        })
        .value_or(false);
  }
  bool operator()(SUBW inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(uint32_t(rs1 - rs2)));
        })
        .value_or(false);
  }
  bool operator()(SLLW inst) {
    return zipOpt(inst.rs1.ReadU32(m_emu), inst.rs2.ReadU32(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(rs1 << (rs2 & 0b11111)));
        })
        .value_or(false);
  }
  bool operator()(SRLW inst) {
    return zipOpt(inst.rs1.ReadU32(m_emu), inst.rs2.ReadU32(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(rs1 >> (rs2 & 0b11111)));
        })
        .value_or(false);
  }
  bool operator()(SRAW inst) {
    return zipOpt(inst.rs1.ReadI32(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(rs1 >> (rs2 & 0b11111)));
        })
        .value_or(false);
  }
  // RV32M & RV64M (Integer Multiplication and Division Extension) //
  bool operator()(MUL inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, rs1 * rs2);
        })
        .value_or(false);
  }
  bool operator()(MULH inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          // signed * signed
          auto mul = llvm::APInt(128, rs1, true) * llvm::APInt(128, rs2, true);
          return inst.rd.Write(m_emu, mul.ashr(64).trunc(64).getZExtValue());
        })
        .value_or(false);
  }
  bool operator()(MULHSU inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          // signed * unsigned
          auto mul = llvm::APInt(128, rs1, true).zext(128) *
                     llvm::APInt(128, rs2, false);
          return inst.rd.Write(m_emu, mul.lshr(64).trunc(64).getZExtValue());
        })
        .value_or(false);
  }
  bool operator()(MULHU inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          // unsigned * unsigned
          auto mul =
              llvm::APInt(128, rs1, false) * llvm::APInt(128, rs2, false);
          return inst.rd.Write(m_emu, mul.lshr(64).trunc(64).getZExtValue());
        })
        .value_or(false);
  }
  bool operator()(DIV inst) {
    return zipOpt(inst.rs1.ReadI64(m_emu), inst.rs2.ReadI64(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, UINT64_MAX);

          if (dividend == INT64_MIN && divisor == -1)
            return inst.rd.Write(m_emu, dividend);

          return inst.rd.Write(m_emu, dividend / divisor);
        })
        .value_or(false);
  }
  bool operator()(DIVU inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, UINT64_MAX);

          return inst.rd.Write(m_emu, dividend / divisor);
        })
        .value_or(false);
  }
  bool operator()(REM inst) {
    return zipOpt(inst.rs1.ReadI64(m_emu), inst.rs2.ReadI64(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, dividend);

          if (dividend == INT64_MIN && divisor == -1)
            return inst.rd.Write(m_emu, 0);

          return inst.rd.Write(m_emu, dividend % divisor);
        })
        .value_or(false);
  }
  bool operator()(REMU inst) {
    return zipOpt(inst.rs1.Read(m_emu), inst.rs2.Read(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, dividend);

          return inst.rd.Write(m_emu, dividend % divisor);
        })
        .value_or(false);
  }
  bool operator()(MULW inst) {
    return zipOpt(inst.rs1.ReadI32(m_emu), inst.rs2.ReadI32(m_emu))
        .transform([&](auto &&tup) {
          auto [rs1, rs2] = tup;
          return inst.rd.Write(m_emu, SextW(rs1 * rs2));
        })
        .value_or(false);
  }
  bool operator()(DIVW inst) {
    return zipOpt(inst.rs1.ReadI32(m_emu), inst.rs2.ReadI32(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, UINT64_MAX);

          if (dividend == INT32_MIN && divisor == -1)
            return inst.rd.Write(m_emu, SextW(dividend));

          return inst.rd.Write(m_emu, SextW(dividend / divisor));
        })
        .value_or(false);
  }
  bool operator()(DIVUW inst) {
    return zipOpt(inst.rs1.ReadU32(m_emu), inst.rs2.ReadU32(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, UINT64_MAX);

          return inst.rd.Write(m_emu, SextW(dividend / divisor));
        })
        .value_or(false);
  }
  bool operator()(REMW inst) {
    return zipOpt(inst.rs1.ReadI32(m_emu), inst.rs2.ReadI32(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, SextW(dividend));

          if (dividend == INT32_MIN && divisor == -1)
            return inst.rd.Write(m_emu, 0);

          return inst.rd.Write(m_emu, SextW(dividend % divisor));
        })
        .value_or(false);
  }
  bool operator()(REMUW inst) {
    return zipOpt(inst.rs1.ReadU32(m_emu), inst.rs2.ReadU32(m_emu))
        .transform([&](auto &&tup) {
          auto [dividend, divisor] = tup;

          if (divisor == 0)
            return inst.rd.Write(m_emu, SextW(dividend));

          return inst.rd.Write(m_emu, SextW(dividend % divisor));
        })
        .value_or(false);
  }
  // RV32A & RV64A (The standard atomic instruction extension) //
  bool operator()(LR_W) { return AtomicSequence(m_emu); }
  bool operator()(LR_D) { return AtomicSequence(m_emu); }
  bool operator()(SC_W) {
    llvm_unreachable("should be handled in AtomicSequence");
  }
  bool operator()(SC_D) {
    llvm_unreachable("should be handled in AtomicSequence");
  }
  bool operator()(AMOSWAP_W inst) {
    return AtomicSwap<AMOSWAP_W, uint32_t>(m_emu, inst, 4, SextW);
  }
  bool operator()(AMOADD_W inst) {
    return AtomicADD<AMOADD_W, uint32_t>(m_emu, inst, 4, SextW);
  }
  bool operator()(AMOXOR_W inst) {
    return AtomicBitOperate<AMOXOR_W, uint32_t>(
        m_emu, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a ^ b; });
  }
  bool operator()(AMOAND_W inst) {
    return AtomicBitOperate<AMOAND_W, uint32_t>(
        m_emu, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a & b; });
  }
  bool operator()(AMOOR_W inst) {
    return AtomicBitOperate<AMOOR_W, uint32_t>(
        m_emu, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a | b; });
  }
  bool operator()(AMOMIN_W inst) {
    return AtomicCmp<AMOMIN_W, uint32_t>(
        m_emu, inst, 4, SextW, [](uint32_t a, uint32_t b) {
          return uint32_t(std::min(int32_t(a), int32_t(b)));
        });
  }
  bool operator()(AMOMAX_W inst) {
    return AtomicCmp<AMOMAX_W, uint32_t>(
        m_emu, inst, 4, SextW, [](uint32_t a, uint32_t b) {
          return uint32_t(std::max(int32_t(a), int32_t(b)));
        });
  }
  bool operator()(AMOMINU_W inst) {
    return AtomicCmp<AMOMINU_W, uint32_t>(
        m_emu, inst, 4, SextW,
        [](uint32_t a, uint32_t b) { return std::min(a, b); });
  }
  bool operator()(AMOMAXU_W inst) {
    return AtomicCmp<AMOMAXU_W, uint32_t>(
        m_emu, inst, 4, SextW,
        [](uint32_t a, uint32_t b) { return std::max(a, b); });
  }
  bool operator()(AMOSWAP_D inst) {
    return AtomicSwap<AMOSWAP_D, uint64_t>(m_emu, inst, 8, ZextD);
  }
  bool operator()(AMOADD_D inst) {
    return AtomicADD<AMOADD_D, uint64_t>(m_emu, inst, 8, ZextD);
  }
  bool operator()(AMOXOR_D inst) {
    return AtomicBitOperate<AMOXOR_D, uint64_t>(
        m_emu, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a ^ b; });
  }
  bool operator()(AMOAND_D inst) {
    return AtomicBitOperate<AMOAND_D, uint64_t>(
        m_emu, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a & b; });
  }
  bool operator()(AMOOR_D inst) {
    return AtomicBitOperate<AMOOR_D, uint64_t>(
        m_emu, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a | b; });
  }
  bool operator()(AMOMIN_D inst) {
    return AtomicCmp<AMOMIN_D, uint64_t>(
        m_emu, inst, 8, ZextD, [](uint64_t a, uint64_t b) {
          return uint64_t(std::min(int64_t(a), int64_t(b)));
        });
  }
  bool operator()(AMOMAX_D inst) {
    return AtomicCmp<AMOMAX_D, uint64_t>(
        m_emu, inst, 8, ZextD, [](uint64_t a, uint64_t b) {
          return uint64_t(std::max(int64_t(a), int64_t(b)));
        });
  }
  bool operator()(AMOMINU_D inst) {
    return AtomicCmp<AMOMINU_D, uint64_t>(
        m_emu, inst, 8, ZextD,
        [](uint64_t a, uint64_t b) { return std::min(a, b); });
  }
  bool operator()(AMOMAXU_D inst) {
    return AtomicCmp<AMOMAXU_D, uint64_t>(
        m_emu, inst, 8, ZextD,
        [](uint64_t a, uint64_t b) { return std::max(a, b); });
  }
  bool operator()(INVALID inst) { return false; }
  bool operator()(RESERVED inst) { return false; }
  bool operator()(EBREAK inst) { return false; }
  bool operator()(HINT inst) { return true; }
  bool operator()(NOP inst) { return true; }
};

bool EmulateInstructionRISCV::Execute(DecodeResult inst, bool ignore_cond) {
  return std::visit(Executor(*this, ignore_cond, inst.is_rvc), inst.decoded);
}

bool EmulateInstructionRISCV::EvaluateInstruction(uint32_t options) {
  bool increase_pc = options & eEmulateInstructionOptionAutoAdvancePC;
  bool ignore_cond = options & eEmulateInstructionOptionIgnoreConditions;

  if (!increase_pc)
    return Execute(m_decoded, ignore_cond);

  auto old_pc = ReadPC();
  if (!old_pc)
    return false;

  bool success = Execute(m_decoded, ignore_cond);
  if (!success)
    return false;

  auto new_pc = ReadPC();
  if (!new_pc)
    return false;

  // If the pc is not updated during execution, we do it here.
  return new_pc != old_pc ||
         WritePC(*old_pc + Executor::size(m_decoded.is_rvc));
}

llvm::Optional<DecodeResult>
EmulateInstructionRISCV::ReadInstructionAt(lldb::addr_t addr) {
  return ReadMem<uint32_t>(addr)
      .transform([&](uint32_t inst) { return Decode(inst); })
      .value_or(llvm::None);
}

bool EmulateInstructionRISCV::ReadInstruction() {
  auto addr = ReadPC();
  m_addr = addr.value_or(LLDB_INVALID_ADDRESS);
  if (!addr)
    return false;
  auto inst = ReadInstructionAt(*addr);
  if (!inst)
    return false;
  m_decoded = *inst;
  if (inst->is_rvc)
    m_opcode.SetOpcode16(inst->inst, GetByteOrder());
  else
    m_opcode.SetOpcode32(inst->inst, GetByteOrder());
  return true;
}

llvm::Optional<lldb::addr_t> EmulateInstructionRISCV::ReadPC() {
  bool success = false;
  auto addr = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC,
                                   LLDB_INVALID_ADDRESS, &success);
  return success ? llvm::Optional<lldb::addr_t>(addr) : llvm::None;
}

bool EmulateInstructionRISCV::WritePC(lldb::addr_t pc) {
  EmulateInstruction::Context ctx;
  ctx.type = eContextAdvancePC;
  ctx.SetNoArgs();
  return WriteRegisterUnsigned(ctx, eRegisterKindGeneric,
                               LLDB_REGNUM_GENERIC_PC, pc);
}

llvm::Optional<RegisterInfo>
EmulateInstructionRISCV::GetRegisterInfo(lldb::RegisterKind reg_kind,
                                         uint32_t reg_index) {
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
    return {};

  return array[reg_index];
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
