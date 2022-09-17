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
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(EmulateInstructionRISCV, InstructionRISCV)

namespace lldb_private {

// The funct3 is the type of compare in B<CMP> instructions.
// funct3 means "3-bits function selector", which RISC-V ISA uses as minor
// opcode. It reuses the major opcode encoding space.
constexpr uint32_t BEQ = 0b000;
constexpr uint32_t BNE = 0b001;
constexpr uint32_t BLT = 0b100;
constexpr uint32_t BGE = 0b101;
constexpr uint32_t BLTU = 0b110;
constexpr uint32_t BGEU = 0b111;

constexpr uint32_t DecodeSHAMT5(uint32_t inst) { return DecodeRS2(inst); }
constexpr uint32_t DecodeSHAMT7(uint32_t inst) {
  return (inst & 0x7F00000) >> 20;
}
constexpr uint32_t DecodeFunct3(uint32_t inst) { return (inst & 0x7000) >> 12; }

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
  return (uint64_t(int64_t(int32_t(inst & 0xFE00000)) >> 20)) // imm[11:5]
         | ((inst & 0xF80) >> 7);                             // imm[4:0]
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

static bool ReadRegister(EmulateInstructionRISCV &emulator, uint32_t reg_encode,
                         RegisterValue &value) {
  uint32_t lldb_reg = GPREncodingToLLDB(reg_encode);
  return emulator.ReadRegister(eRegisterKindLLDB, lldb_reg, value);
}

static bool WriteRegister(EmulateInstructionRISCV &emulator,
                          uint32_t reg_encode, const RegisterValue &value) {
  uint32_t lldb_reg = GPREncodingToLLDB(reg_encode);
  EmulateInstruction::Context ctx;
  ctx.type = EmulateInstruction::eContextRegisterStore;
  ctx.SetNoArgs();
  return emulator.WriteRegister(ctx, eRegisterKindLLDB, lldb_reg, value);
}

static bool ExecJAL(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  bool success = false;
  int64_t offset = SignExt(DecodeJImm(inst));
  int64_t pc = emulator.ReadPC(success);
  return success && emulator.WritePC(pc + offset) &&
         WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(uint64_t(pc + 4)));
}

static bool ExecJALR(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int64_t offset = SignExt(DecodeIImm(inst));
  RegisterValue value;
  if (!ReadRegister(emulator, DecodeRS1(inst), value))
    return false;
  bool success = false;
  int64_t pc = emulator.ReadPC(success);
  int64_t rs1 = int64_t(value.GetAsUInt64());
  // JALR clears the bottom bit. According to riscv-spec:
  // "The JALR instruction now clears the lowest bit of the calculated target
  // address, to simplify hardware and to allow auxiliary information to be
  // stored in function pointers."
  return emulator.WritePC((rs1 + offset) & ~1) &&
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

struct Wrapped {
  RegisterValue value;

  template <typename T>
  [[nodiscard]] std::enable_if_t<std::is_unsigned_v<T>, T> trunc() const {
    return T(value.GetAsUInt64());
  }

  template <typename T> T sext() const = delete;
};

template <> int32_t Wrapped::sext<int32_t>() const {
  return int32_t(trunc<uint32_t>());
}

template <> int64_t Wrapped::sext<int64_t>() const {
  return int64_t(trunc<uint64_t>());
}

template <bool hasRS2> struct RSRegs {
private:
  Wrapped rs1_value;

public:
  bool valid;

  Wrapped &rs1() { return rs1_value; }
};

template <> struct RSRegs<true> : RSRegs<false> {
private:
  Wrapped rs2_value;

public:
  Wrapped &rs2() { return rs2_value; }
};

RSRegs<true> readRS1RS2(EmulateInstructionRISCV &emulator, uint32_t inst) {
  RSRegs<true> value{};
  value.valid = ReadRegister(emulator, DecodeRS1(inst), value.rs1().value) &&
                ReadRegister(emulator, DecodeRS2(inst), value.rs2().value);
  return value;
}

RSRegs<false> readRS1(EmulateInstructionRISCV &emulator, uint32_t inst) {
  RSRegs<false> value{};
  value.valid = ReadRegister(emulator, DecodeRS1(inst), value.rs1().value);
  return value;
}

static bool ExecB(EmulateInstructionRISCV &emulator, uint32_t inst,
                  bool ignore_cond) {
  bool success = false;
  uint64_t pc = emulator.ReadPC(success);
  if (!success)
    return false;

  uint64_t offset = SignExt(DecodeBImm(inst));
  uint64_t target = pc + offset;
  if (ignore_cond)
    return emulator.WritePC(target);

  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint32_t funct3 = DecodeFunct3(inst);
  if (CompareB(rs.rs1().trunc<uint64_t>(), rs.rs2().trunc<uint64_t>(), funct3))
    return emulator.WritePC(target);

  return true;
}

static bool ExecLUI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  uint32_t imm = DecodeUImm(inst);
  RegisterValue value;
  value.SetUInt64(SignExt(imm));
  return WriteRegister(emulator, DecodeRD(inst), value);
}

static bool ExecAUIPC(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  uint32_t imm = DecodeUImm(inst);
  RegisterValue value;
  bool success = false;
  value.SetUInt64(SignExt(imm) + emulator.ReadPC(success));
  return success && WriteRegister(emulator, DecodeRD(inst), value);
}

template <typename T>
static std::enable_if_t<std::is_integral_v<T>, llvm::Optional<T>>
ReadMem(EmulateInstructionRISCV &emulator, uint64_t addr) {
  EmulateInstructionRISCV::Context ctx;
  ctx.type = EmulateInstruction::eContextRegisterLoad;
  ctx.SetNoArgs();
  bool success = false;
  T result = emulator.ReadMemoryUnsigned(ctx, addr, sizeof(T), T(), &success);
  if (!success)
    return {}; // aka return false
  return result;
}

template <typename T>
static bool WriteMem(EmulateInstructionRISCV &emulator, uint64_t addr,
                     RegisterValue value) {
  EmulateInstructionRISCV::Context ctx;
  ctx.type = EmulateInstruction::eContextRegisterStore;
  ctx.SetNoArgs();
  return emulator.WriteMemoryUnsigned(ctx, addr, value.GetAsUInt64(),
                                      sizeof(T));
}

static uint64_t LoadStoreAddr(EmulateInstructionRISCV &emulator,
                              uint32_t inst) {
  int32_t imm = SignExt(DecodeSImm(inst));
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return LLDB_INVALID_ADDRESS;
  uint64_t addr = rs.rs1().trunc<uint64_t>() + uint64_t(imm);
  return addr;
}

// Read T from memory, then load its sign-extended value E to register.
template <typename T, typename E>
static bool Load(EmulateInstructionRISCV &emulator, uint32_t inst,
                 uint64_t (*extend)(E)) {
  uint64_t addr = LoadStoreAddr(emulator, inst);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;
  E value = E(ReadMem<T>(emulator, addr).value());
  if (!value)
    return false;
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(extend(value)));
}

template <typename T>
static bool Store(EmulateInstructionRISCV &emulator, uint32_t inst) {
  uint64_t addr = LoadStoreAddr(emulator, inst);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;
  RegisterValue value;
  if (!ReadRegister(emulator, DecodeRS2(inst), value))
    return false;
  return WriteMem<T>(emulator, addr, value);
}

// RV32I & RV64I (The base integer ISA) //
static bool ExecLB(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint8_t, int8_t>(emulator, inst, SextW);
}

static bool ExecLH(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint16_t, int16_t>(emulator, inst, SextW);
}

static bool ExecLW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint32_t, int32_t>(emulator, inst, SextW);
}

static bool ExecLD(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint64_t, uint64_t>(emulator, inst, ZextD);
}

static bool ExecLBU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint8_t, uint8_t>(emulator, inst, ZextD);
}

static bool ExecLHU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint16_t, uint16_t>(emulator, inst, ZextD);
}

static bool ExecLWU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Load<uint32_t, uint32_t>(emulator, inst, ZextD);
}

static bool ExecSB(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Store<uint8_t>(emulator, inst);
}

static bool ExecSH(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Store<uint16_t>(emulator, inst);
}

static bool ExecSW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Store<uint32_t>(emulator, inst);
}

static bool ExecSD(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return Store<uint64_t>(emulator, inst);
}

static bool ExecADDI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().sext<int64_t>() + int64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLTI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().sext<int64_t>() < int64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLTIU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() < uint64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecXORI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() ^ uint64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecORI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() | uint64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecANDI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() & uint64_t(imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLLI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() << DecodeSHAMT7(inst);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRLI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() >> DecodeSHAMT7(inst);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRAI(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().sext<int64_t>() >> DecodeSHAMT7(inst);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecADD(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() + rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSUB(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() - rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLL(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>()
                    << (rs.rs2().trunc<uint64_t>() & 0b111111);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLT(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().sext<int64_t>() < rs.rs2().sext<int64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLTU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() < rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecXOR(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() ^ rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRL(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result =
      rs.rs1().trunc<uint64_t>() >> (rs.rs2().trunc<uint64_t>() & 0b111111);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRA(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result =
      rs.rs1().sext<int64_t>() >> (rs.rs2().trunc<uint64_t>() & 0b111111);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecOR(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() | rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecAND(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() & rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecADDIW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  int32_t imm = SignExt(DecodeIImm(inst));

  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().sext<int32_t>() + imm);
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLLIW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().trunc<uint32_t>() << DecodeSHAMT5(inst));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRLIW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().trunc<uint32_t>() >> DecodeSHAMT5(inst));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRAIW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().sext<int32_t>() >> DecodeSHAMT5(inst));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecADDW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result =
      SextW(uint32_t(rs.rs1().trunc<uint64_t>() + rs.rs2().trunc<uint64_t>()));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSUBW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result =
      SextW(uint32_t(rs.rs1().trunc<uint64_t>() - rs.rs2().trunc<uint64_t>()));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSLLW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().trunc<uint32_t>()
                          << (rs.rs2().trunc<uint32_t>() & 0b11111));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRLW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().trunc<uint32_t>() >>
                          (rs.rs2().trunc<uint32_t>() & 0b11111));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecSRAW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().sext<int32_t>() >>
                          (rs.rs2().trunc<uint64_t>() & 0b111111));
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

// RV32M & RV64M (The standard integer multiplication and division extension) //
static bool ExecMUL(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = rs.rs1().trunc<uint64_t>() * rs.rs2().trunc<uint64_t>();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecMULH(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  llvm::APInt mul = llvm::APInt(128, rs.rs1().trunc<uint64_t>(), true) *
                    llvm::APInt(128, rs.rs2().trunc<uint64_t>(), true);

  // signed * signed
  auto result = mul.ashr(64).trunc(64).getZExtValue();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecMULHSU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  // signed * unsigned
  llvm::APInt mul =
      llvm::APInt(128, rs.rs1().trunc<uint64_t>(), true).zext(128) *
      llvm::APInt(128, rs.rs2().trunc<uint64_t>(), false);

  auto result = mul.lshr(64).trunc(64).getZExtValue();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecMULHU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  // unsigned * unsigned
  llvm::APInt mul = llvm::APInt(128, rs.rs1().trunc<uint64_t>(), false) *
                    llvm::APInt(128, rs.rs2().trunc<uint64_t>(), false);
  auto result = mul.lshr(64).trunc(64).getZExtValue();
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecDIV(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  int64_t dividend = rs.rs1().sext<int64_t>();
  int64_t divisor = rs.rs2().sext<int64_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(UINT64_MAX));

  if (dividend == INT64_MIN && divisor == -1)
    return WriteRegister(emulator, DecodeRD(inst),
                         RegisterValue(uint64_t(dividend)));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(uint64_t(dividend / divisor)));
}

static bool ExecDIVU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t dividend = rs.rs1().trunc<uint64_t>();
  uint64_t divisor = rs.rs2().trunc<uint64_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(UINT64_MAX));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(dividend / divisor));
}

static bool ExecREM(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  int64_t dividend = rs.rs1().sext<int64_t>();
  int64_t divisor = rs.rs2().sext<int64_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst),
                         RegisterValue(uint64_t(dividend)));

  if (dividend == INT64_MIN && divisor == -1)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(uint64_t(0)));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(uint64_t(dividend % divisor)));
}

static bool ExecREMU(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t dividend = rs.rs1().trunc<uint64_t>();
  uint64_t divisor = rs.rs2().trunc<uint64_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(dividend));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(dividend % divisor));
}

static bool ExecMULW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint64_t result = SextW(rs.rs1().sext<int32_t>() * rs.rs2().sext<int32_t>());
  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(result));
}

static bool ExecDIVW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  int32_t dividend = rs.rs1().sext<int32_t>();
  int32_t divisor = rs.rs2().sext<int32_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(UINT64_MAX));

  if (dividend == INT32_MIN && divisor == -1)
    return WriteRegister(emulator, DecodeRD(inst),
                         RegisterValue(SextW(dividend)));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(SextW(dividend / divisor)));
}

static bool ExecDIVUW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint32_t dividend = rs.rs1().trunc<uint32_t>();
  uint32_t divisor = rs.rs2().trunc<uint32_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(UINT64_MAX));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(SextW(dividend / divisor)));
}

static bool ExecREMW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  int32_t dividend = rs.rs1().sext<int32_t>();
  int32_t divisor = rs.rs2().sext<int32_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst),
                         RegisterValue(SextW(dividend)));

  if (dividend == INT32_MIN && divisor == -1)
    return WriteRegister(emulator, DecodeRD(inst), RegisterValue(uint64_t(0)));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(SextW(dividend % divisor)));
}

static bool ExecREMUW(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  auto rs = readRS1RS2(emulator, inst);
  if (!rs.valid)
    return false;

  uint32_t dividend = rs.rs1().trunc<uint32_t>();
  uint32_t divisor = rs.rs2().trunc<uint32_t>();

  if (divisor == 0)
    return WriteRegister(emulator, DecodeRD(inst),
                         RegisterValue(SextW(dividend)));

  return WriteRegister(emulator, DecodeRD(inst),
                       RegisterValue(SextW(dividend % divisor)));
}

// RV32A & RV64A (The standard atomic instruction extension) //
static uint64_t AtomicAddr(EmulateInstructionRISCV &emulator, uint32_t reg,
                           unsigned int align) {
  RegisterValue value;
  if (!ReadRegister(emulator, reg, value))
    return LLDB_INVALID_ADDRESS;

  uint64_t addr = value.GetAsUInt64();
  return addr % align == 0 ? addr : LLDB_INVALID_ADDRESS;
}

template <typename T>
static bool AtomicSwap(EmulateInstructionRISCV &emulator, uint32_t inst,
                       int align, uint64_t (*extend)(T)) {
  RegisterValue rs2;
  if (!ReadRegister(emulator, DecodeRS2(inst), rs2))
    return false;

  uint64_t addr = AtomicAddr(emulator, DecodeRS1(inst), align);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  T tmp = ReadMem<T>(emulator, addr).value();
  if (!tmp)
    return false;

  bool success =
      WriteMem<T>(emulator, addr, RegisterValue(T(rs2.GetAsUInt64())));
  if (!success)
    return false;

  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(extend(tmp)));
}

template <typename T>
static bool AtomicADD(EmulateInstructionRISCV &emulator, uint32_t inst,
                      int align, uint64_t (*extend)(T)) {
  RegisterValue rs2;
  if (!ReadRegister(emulator, DecodeRS2(inst), rs2))
    return false;

  uint64_t addr = AtomicAddr(emulator, DecodeRS1(inst), align);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  T value = ReadMem<T>(emulator, addr).value();
  if (!value)
    return false;

  bool success =
      WriteMem<T>(emulator, addr, RegisterValue(value + T(rs2.GetAsUInt64())));
  if (!success)
    return false;

  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(extend(value)));
}

template <typename T>
static bool AtomicBitOperate(EmulateInstructionRISCV &emulator, uint32_t inst,
                             int align, uint64_t (*extend)(T),
                             T (*operate)(T, T)) {
  RegisterValue rs2;
  if (!ReadRegister(emulator, DecodeRS2(inst), rs2))
    return false;

  uint64_t addr = AtomicAddr(emulator, DecodeRS1(inst), align);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  T value = ReadMem<T>(emulator, addr).value();
  if (!value)
    return false;

  bool success = WriteMem<T>(
      emulator, addr, RegisterValue(operate(value, T(rs2.GetAsUInt64()))));
  if (!success)
    return false;

  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(extend(value)));
}

template <typename T>
static bool AtomicCmp(EmulateInstructionRISCV &emulator, uint32_t inst,
                      int align, uint64_t (*extend)(T), T (*cmp)(T, T)) {
  RegisterValue rs2;
  if (!ReadRegister(emulator, DecodeRS2(inst), rs2))
    return false;

  uint64_t addr = AtomicAddr(emulator, DecodeRS1(inst), align);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  T value = ReadMem<T>(emulator, addr).value();
  if (!value)
    return false;

  bool success = WriteMem<T>(emulator, addr,
                             RegisterValue(cmp(value, T(rs2.GetAsUInt64()))));
  if (!success)
    return false;

  return WriteRegister(emulator, DecodeRD(inst), RegisterValue(extend(value)));
}

// 00010 aq[1] rl[1] 00000 rs1[5] 010 rd[5] 0101111 LR.D
// 00010 aq[1] rl[1] 00000 rs1[5] 011 rd[5] 0101111 LR.W
static bool IsLR(uint32_t inst) {
  return (inst & 0xF9F0707F) == 0x1000202F || (inst & 0xF9F0707F) == 0x1000302F;
}

// 00011 aq[1] rl[1] rs2[5] rs1[5] 010 rd[5] 0101111 SC.W
// 00011 aq[1] rl[1] rs2[5] rs1[5] 011 rd[5] 0101111 SC.D
static bool IsSC(uint32_t inst) {
  return (inst & 0xF800707F) == 0x1800202F || (inst & 0xF800707F) == 0x1800302F;
}

// imm[7] rs2[5] rs1[5] 001 imm[5] 1100011 BNE
static bool IsBNE(uint32_t inst) { return (inst & 0x707F) == 0x1063; }

static bool ExecAtomicSequence(EmulateInstructionRISCV &emulator, uint32_t inst,
                               bool) {
  // The atomic sequence is always 4 instructions long:
  // example:
  //   110cc:	100427af          	lr.w	a5,(s0)
  //   110d0:	00079663          	bnez	a5,110dc
  //   110d4:	1ce426af          	sc.w.aq	a3,a4,(s0)
  //   110d8:	fe069ae3          	bnez	a3,110cc
  //   110dc:   ........          	<next instruction>
  bool success = false;
  llvm::Optional<lldb::addr_t> pc = emulator.ReadPC(success);
  if (!success || !pc)
    return false;
  lldb::addr_t current_pc = pc.value();
  lldb::addr_t start_pc = current_pc;
  llvm::Optional<uint32_t> value = inst;

  // inst must be LR
  if (!value || !IsLR(value.value()))
    return false;
  value = ReadMem<uint32_t>(emulator, current_pc += 4).value();

  // inst must be BNE to exit
  if (!value || !IsBNE(value.value()))
    return false;
  auto exit_pc = current_pc + SextW(DecodeBImm(value.value()));
  value = ReadMem<uint32_t>(emulator, current_pc += 4).value();

  // inst must be SC
  if (!value || !IsSC(value.value()))
    return false;
  value = ReadMem<uint32_t>(emulator, current_pc += 4).value();

  // inst must be BNE to restart
  if (!value || !IsBNE(value.value()))
    return false;
  if (current_pc + SextW(DecodeBImm(value.value())) != start_pc)
    return false;
  current_pc += 4;

  if (exit_pc != current_pc)
    return false;

  return emulator.WritePC(current_pc);
}

static bool ExecLR_W(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return ExecAtomicSequence(emulator, inst, false);
}

static bool ExecAMOSWAP_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicSwap<uint32_t>(emulator, inst, 4, SextW);
}

static bool ExecAMOADD_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicADD<uint32_t>(emulator, inst, 4, SextW);
}

static bool ExecAMOXOR_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicBitOperate<uint32_t>(
      emulator, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a ^ b; });
}

static bool ExecAMOAND_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicBitOperate<uint32_t>(
      emulator, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a & b; });
}

static bool ExecAMOOR_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                        bool) {
  return AtomicBitOperate<uint32_t>(
      emulator, inst, 4, SextW, [](uint32_t a, uint32_t b) { return a | b; });
}

static bool ExecAMOMIN_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicCmp<uint32_t>(
      emulator, inst, 4, SextW, [](uint32_t a, uint32_t b) {
        return uint32_t(std::min(int32_t(a), int32_t(b)));
      });
}

static bool ExecAMOMAX_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicCmp<uint32_t>(
      emulator, inst, 4, SextW, [](uint32_t a, uint32_t b) {
        return uint32_t(std::max(int32_t(a), int32_t(b)));
      });
}

static bool ExecAMOMINU_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicCmp<uint32_t>(
      emulator, inst, 4, SextW,
      [](uint32_t a, uint32_t b) { return std::min(a, b); });
}

static bool ExecAMOMAXU_W(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicCmp<uint32_t>(
      emulator, inst, 4, SextW,
      [](uint32_t a, uint32_t b) { return std::max(a, b); });
}

static bool ExecLR_D(EmulateInstructionRISCV &emulator, uint32_t inst, bool) {
  return ExecAtomicSequence(emulator, inst, false);
}

static bool ExecAMOSWAP_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicSwap<uint64_t>(emulator, inst, 8, ZextD);
}

static bool ExecAMOADD_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicADD<uint64_t>(emulator, inst, 8, ZextD);
}

static bool ExecAMOXOR_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicBitOperate<uint64_t>(
      emulator, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a ^ b; });
}

static bool ExecAMOAND_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicBitOperate<uint64_t>(
      emulator, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a & b; });
}

static bool ExecAMOOR_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                        bool) {
  return AtomicBitOperate<uint64_t>(
      emulator, inst, 8, ZextD, [](uint64_t a, uint64_t b) { return a | b; });
}

static bool ExecAMOMIN_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicCmp<uint64_t>(
      emulator, inst, 8, ZextD, [](uint64_t a, uint64_t b) {
        return uint64_t(std::min(int64_t(a), int64_t(b)));
      });
}

static bool ExecAMOMAX_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                         bool) {
  return AtomicCmp<uint64_t>(
      emulator, inst, 8, ZextD, [](uint64_t a, uint64_t b) {
        return uint64_t(std::max(int64_t(a), int64_t(b)));
      });
}

static bool ExecAMOMINU_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicCmp<uint64_t>(
      emulator, inst, 8, ZextD,
      [](uint64_t a, uint64_t b) { return std::min(a, b); });
}

static bool ExecAMOMAXU_D(EmulateInstructionRISCV &emulator, uint32_t inst,
                          bool) {
  return AtomicCmp<uint64_t>(
      emulator, inst, 8, ZextD,
      [](uint64_t a, uint64_t b) { return std::max(a, b); });
}

static const InstrPattern PATTERNS[] = {
    // RV32I & RV64I (The base integer ISA) //
    {"LUI", 0x7F, 0x37, ExecLUI},
    {"AUIPC", 0x7F, 0x17, ExecAUIPC},
    {"JAL", 0x7F, 0x6F, ExecJAL},
    {"JALR", 0x707F, 0x67, ExecJALR},
    {"B<CMP>", 0x7F, 0x63, ExecB},
    {"LB", 0x707F, 0x3, ExecLB},
    {"LH", 0x707F, 0x1003, ExecLH},
    {"LW", 0x707F, 0x2003, ExecLW},
    {"LBU", 0x707F, 0x4003, ExecLBU},
    {"LHU", 0x707F, 0x5003, ExecLHU},
    {"SB", 0x707F, 0x23, ExecSB},
    {"SH", 0x707F, 0x1023, ExecSH},
    {"SW", 0x707F, 0x2023, ExecSW},
    {"ADDI", 0x707F, 0x13, ExecADDI},
    {"SLTI", 0x707F, 0x2013, ExecSLTI},
    {"SLTIU", 0x707F, 0x3013, ExecSLTIU},
    {"XORI", 0x707F, 0x4013, ExecXORI},
    {"ORI", 0x707F, 0x6013, ExecORI},
    {"ANDI", 0x707F, 0x7013, ExecANDI},
    {"SLLI", 0xF800707F, 0x1013, ExecSLLI},
    {"SRLI", 0xF800707F, 0x5013, ExecSRLI},
    {"SRAI", 0xF800707F, 0x40005013, ExecSRAI},
    {"ADD", 0xFE00707F, 0x33, ExecADD},
    {"SUB", 0xFE00707F, 0x40000033, ExecSUB},
    {"SLL", 0xFE00707F, 0x1033, ExecSLL},
    {"SLT", 0xFE00707F, 0x2033, ExecSLT},
    {"SLTU", 0xFE00707F, 0x3033, ExecSLTU},
    {"XOR", 0xFE00707F, 0x4033, ExecXOR},
    {"SRL", 0xFE00707F, 0x5033, ExecSRL},
    {"SRA", 0xFE00707F, 0x40005033, ExecSRA},
    {"OR", 0xFE00707F, 0x6033, ExecOR},
    {"AND", 0xFE00707F, 0x7033, ExecAND},
    {"LWU", 0x707F, 0x6003, ExecLWU},
    {"LD", 0x707F, 0x3003, ExecLD},
    {"SD", 0x707F, 0x3023, ExecSD},
    {"ADDIW", 0x707F, 0x1B, ExecADDIW},
    {"SLLIW", 0xFE00707F, 0x101B, ExecSLLIW},
    {"SRLIW", 0xFE00707F, 0x501B, ExecSRLIW},
    {"SRAIW", 0xFE00707F, 0x4000501B, ExecSRAIW},
    {"ADDW", 0xFE00707F, 0x3B, ExecADDW},
    {"SUBW", 0xFE00707F, 0x4000003B, ExecSUBW},
    {"SLLW", 0xFE00707F, 0x103B, ExecSLLW},
    {"SRLW", 0xFE00707F, 0x503B, ExecSRLW},
    {"SRAW", 0xFE00707F, 0x4000503B, ExecSRAW},

    // RV32M & RV64M (The integer multiplication and division extension) //
    {"MUL", 0xFE00707F, 0x2000033, ExecMUL},
    {"MULH", 0xFE00707F, 0x2001033, ExecMULH},
    {"MULHSU", 0xFE00707F, 0x2002033, ExecMULHSU},
    {"MULHU", 0xFE00707F, 0x2003033, ExecMULHU},
    {"DIV", 0xFE00707F, 0x2004033, ExecDIV},
    {"DIVU", 0xFE00707F, 0x2005033, ExecDIVU},
    {"REM", 0xFE00707F, 0x2006033, ExecREM},
    {"REMU", 0xFE00707F, 0x2007033, ExecREMU},
    {"MULW", 0xFE00707F, 0x200003B, ExecMULW},
    {"DIVW", 0xFE00707F, 0x200403B, ExecDIVW},
    {"DIVUW", 0xFE00707F, 0x200503B, ExecDIVUW},
    {"REMW", 0xFE00707F, 0x200603B, ExecREMW},
    {"REMUW", 0xFE00707F, 0x200703B, ExecREMUW},

    // RV32A & RV64A (The standard atomic instruction extension) //
    {"LR_W", 0xF9F0707F, 0x1000202F, ExecLR_W},
    {"AMOSWAP_W", 0xF800707F, 0x800202F, ExecAMOSWAP_W},
    {"AMOADD_W", 0xF800707F, 0x202F, ExecAMOADD_W},
    {"AMOXOR_W", 0xF800707F, 0x2000202F, ExecAMOXOR_W},
    {"AMOAND_W", 0xF800707F, 0x6000202F, ExecAMOAND_W},
    {"AMOOR_W", 0xF800707F, 0x4000202F, ExecAMOOR_W},
    {"AMOMIN_W", 0xF800707F, 0x8000202F, ExecAMOMIN_W},
    {"AMOMAX_W", 0xF800707F, 0xA000202F, ExecAMOMAX_W},
    {"AMOMINU_W", 0xF800707F, 0xC000202F, ExecAMOMINU_W},
    {"AMOMAXU_W", 0xF800707F, 0xE000202F, ExecAMOMAXU_W},
    {"LR_D", 0xF9F0707F, 0x1000302F, ExecLR_D},
    {"AMOSWAP_D", 0xF800707F, 0x800302F, ExecAMOSWAP_D},
    {"AMOADD_D", 0xF800707F, 0x302F, ExecAMOADD_D},
    {"AMOXOR_D", 0xF800707F, 0x2000302F, ExecAMOXOR_D},
    {"AMOAND_D", 0xF800707F, 0x6000302F, ExecAMOAND_D},
    {"AMOOR_D", 0xF800707F, 0x4000302F, ExecAMOOR_D},
    {"AMOMIN_D", 0xF800707F, 0x8000302F, ExecAMOMIN_D},
    {"AMOMAX_D", 0xF800707F, 0xA000302F, ExecAMOMAX_D},
    {"AMOMINU_D", 0xF800707F, 0xC000302F, ExecAMOMINU_D},
    {"AMOMAXU_D", 0xF800707F, 0xE000302F, ExecAMOMAXU_D},
};

const InstrPattern *EmulateInstructionRISCV::Decode(uint32_t inst) {
  for (const InstrPattern &pat : PATTERNS) {
    if ((inst & pat.type_mask) == pat.eigen)
      return &pat;
  }
  return nullptr;
}

/// This function only determines the next instruction address for software
/// single stepping by emulating instructions
bool EmulateInstructionRISCV::DecodeAndExecute(uint32_t inst,
                                               bool ignore_cond) {
  Log *log = GetLog(LLDBLog::Unwind);
  const InstrPattern *pattern = this->Decode(inst);
  if (pattern) {
    LLDB_LOGF(log, "EmulateInstructionRISCV::%s: inst(%x) was decoded to %s",
              __FUNCTION__, inst, pattern->name);
    return pattern->exec(*this, inst, ignore_cond);
  }

  LLDB_LOGF(log, "EmulateInstructionRISCV::%s: inst(0x%x) was unsupported",
            __FUNCTION__, inst);
  return false;
}

bool EmulateInstructionRISCV::EvaluateInstruction(uint32_t options) {
  uint32_t inst_size = m_opcode.GetByteSize();
  uint32_t inst = m_opcode.GetOpcode32();
  bool increase_pc = options & eEmulateInstructionOptionAutoAdvancePC;
  bool ignore_cond = options & eEmulateInstructionOptionIgnoreConditions;
  bool success = false;

  lldb::addr_t old_pc = LLDB_INVALID_ADDRESS;
  if (increase_pc) {
    old_pc = ReadPC(success);
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
    lldb::addr_t new_pc = ReadPC(success);
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
  m_addr = ReadPC(success);
  if (!success) {
    m_addr = LLDB_INVALID_ADDRESS;
    return false;
  }

  Context ctx;
  ctx.type = eContextReadOpcode;
  ctx.SetNoArgs();
  uint32_t inst = uint32_t(ReadMemoryUnsigned(ctx, m_addr, 4, 0, &success));
  uint16_t try_rvc = uint16_t(inst & 0x0000ffff);
  // check whether the compressed encode could be valid
  uint16_t mask = try_rvc & 0b11;
  if (try_rvc != 0 && mask != 3)
    m_opcode.SetOpcode16(try_rvc, GetByteOrder());
  else
    m_opcode.SetOpcode32(inst, GetByteOrder());

  return true;
}

lldb::addr_t EmulateInstructionRISCV::ReadPC(bool &success) {
  return ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC,
                              LLDB_INVALID_ADDRESS, &success);
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
