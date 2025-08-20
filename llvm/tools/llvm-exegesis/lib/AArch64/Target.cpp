//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "AArch64.h"
#include "AArch64RegisterInfo.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/prctl.h> // For PR_PAC_* constants
#ifndef PR_PAC_APIAKEY
#define PR_PAC_APIAKEY (1UL << 0)
#endif
#ifndef PR_PAC_APIBKEY
#define PR_PAC_APIBKEY (1UL << 1)
#endif
#ifndef PR_PAC_APDAKEY
#define PR_PAC_APDAKEY (1UL << 2)
#endif
#ifndef PR_PAC_APDBKEY
#define PR_PAC_APDBKEY (1UL << 3)
#endif
#endif

#define GET_AVAILABLE_OPCODE_CHECKER
#include "AArch64GenInstrInfo.inc"

namespace llvm {
namespace exegesis {

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 32:
    return AArch64::MOVi32imm;
  case 64:
    return AArch64::MOVi64imm;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an immediate value into a register.
static MCInst loadImmediate(MCRegister Reg, unsigned RegBitWidth,
                            const APInt &Value) {
  assert(Value.getBitWidth() <= RegBitWidth &&
         "Value must fit in the Register");
  return MCInstBuilder(getLoadImmediateOpcode(RegBitWidth))
      .addReg(Reg)
      .addImm(Value.getZExtValue());
}

static MCInst loadZPRImmediate(MCRegister Reg, unsigned RegBitWidth,
                               const APInt &Value) {
  assert(Value.getZExtValue() < (1 << 7) &&
         "Value must be in the range of the immediate opcode");
  return MCInstBuilder(AArch64::DUP_ZI_D)
      .addReg(Reg)
      .addImm(Value.getZExtValue())
      .addImm(0);
}

static MCInst loadPPRImmediate(MCRegister Reg, unsigned RegBitWidth,
                               const APInt &Value) {
  // For PPR, we typically use PTRUE instruction to set predicate registers
  return MCInstBuilder(AArch64::PTRUE_B)
      .addReg(Reg)
      .addImm(31); // All lanes true for 16 bits
}

// Generates instructions to load an immediate value into an FPCR register.
static std::vector<MCInst>
loadFPCRImmediate(MCRegister Reg, unsigned RegBitWidth, const APInt &Value) {
  MCRegister TempReg = AArch64::X8;
  MCInst LoadImm = MCInstBuilder(AArch64::MOVi64imm).addReg(TempReg).addImm(0);
  MCInst MoveToFPCR =
      MCInstBuilder(AArch64::MSR).addImm(AArch64SysReg::FPCR).addReg(TempReg);
  return {LoadImm, MoveToFPCR};
}

// Fetch base-instruction to load an FP immediate value into a register.
static unsigned getLoadFPImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 16:
    return AArch64::FMOVH0; // FMOVHi;
  case 32:
    return AArch64::FMOVS0; // FMOVSi;
  case 64:
    return AArch64::MOVID; // FMOVDi;
  case 128:
    return AArch64::MOVIv2d_ns;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an FP immediate value into a register.
static MCInst loadFPImmediate(MCRegister Reg, unsigned RegBitWidth,
                              const APInt &Value) {
  assert(Value.getZExtValue() == 0 && "Expected initialisation value 0");
  MCInst Instructions =
      MCInstBuilder(getLoadFPImmediateOpcode(RegBitWidth)).addReg(Reg);
  if (RegBitWidth >= 64)
    Instructions.addOperand(MCOperand::createImm(Value.getZExtValue()));
  return Instructions;
}

#include "AArch64GenExegesis.inc"

namespace {

class ExegesisAArch64Target : public ExegesisTarget {
public:
  ExegesisAArch64Target()
      : ExegesisTarget(AArch64CpuPfmCounters, AArch64_MC::isOpcodeAvailable) {}

  Error randomizeTargetMCOperand(const Instruction &Instr, const Variable &Var,
                                 MCOperand &AssignedValue,
                                 const BitVector &ForbiddenRegs) const override;

private:
  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, MCRegister Reg,
                               const APInt &Value) const override {
    if (AArch64::GPR32RegClass.contains(Reg))
      return {loadImmediate(Reg, 32, Value)};
    if (AArch64::GPR64RegClass.contains(Reg))
      return {loadImmediate(Reg, 64, Value)};
    if (AArch64::PPRRegClass.contains(Reg))
      return {loadPPRImmediate(Reg, 16, Value)};
    if (AArch64::FPR8RegClass.contains(Reg))
      return {loadFPImmediate(Reg - AArch64::B0 + AArch64::D0, 64, Value)};
    if (AArch64::FPR16RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 16, Value)};
    if (AArch64::FPR32RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 32, Value)};
    if (AArch64::FPR64RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 64, Value)};
    if (AArch64::FPR128RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 128, Value)};
    if (AArch64::ZPRRegClass.contains(Reg))
      return {loadZPRImmediate(Reg, 128, Value)};
    if (Reg == AArch64::FPCR)
      return {loadFPCRImmediate(Reg, 32, Value)};

    errs() << "setRegTo is not implemented, results will be unreliable\n";
    return {};
  }

  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::aarch64 || Arch == Triple::aarch64_be;
  }

  void addTargetSpecificPasses(PassManagerBase &PM) const override {
    // Function return is a pseudo-instruction that needs to be expanded
    PM.add(createAArch64ExpandPseudoPass());
  }
};

Error ExegesisAArch64Target::randomizeTargetMCOperand(
    const Instruction &Instr, const Variable &Var, MCOperand &AssignedValue,
    const BitVector &ForbiddenRegs) const {
  const Operand &Op = Instr.getPrimaryOperand(Var);
  const auto OperandType = Op.getExplicitOperandInfo().OperandType;
  //  FIXME: Implement opcode-specific immediate value handling for system
  //  instructions:
  //   - MRS/MSR: Use valid system register encodings (e.g., NZCV, FPCR, FPSR)
  //   - MSRpstatesvcrImm1: Use valid PSTATE field encodings (e.g., SPSel,
  //   DAIFSet)
  //   - SYSLxt/SYSxt: Use valid system instruction encodings with proper
  //   CRn/CRm/op values
  //   - UDF: Use valid undefined instruction immediate ranges (0-65535)
  //   Currently defaulting to immediate value 0, which may cause invalid
  //   encodings or unreliable benchmark results for these system-level
  //   instructions.
  switch (OperandType) {
  case MCOI::OperandType::OPERAND_UNKNOWN: {
    AssignedValue = MCOperand::createImm(0);
    return Error::success();
  }
  // MSL (Masking Shift Left) imm operand for 32-bit splatted SIMD constants
  // Correspond to AArch64InstructionSelector::tryAdvSIMDModImm321s()
  case llvm::AArch64::OPERAND_MSL_SHIFT: {
    unsigned Opcode = Instr.getOpcode();
    switch (Opcode) {
    case AArch64::MOVIv2s_msl:
    case AArch64::MVNIv2s_msl:
      // Type 7: Pattern 0x00 0x00 abcdefgh 0xFF 0x00 0x00 abcdefgh 0xFF
      // Creates 2-element 32-bit vector with 8-bit imm at positions [15:8] &
      // [47:40] Shift value 264 (0x108) for Type 7 pattern encoding Corresponds
      // to AArch64_AM::encodeAdvSIMDModImmType7()
      AssignedValue = MCOperand::createImm(264);
      return Error::success();
    case AArch64::MOVIv4s_msl:
    case AArch64::MVNIv4s_msl:
      // Type 8: Pattern 0x00 abcdefgh 0xFF 0xFF 0x00 abcdefgh 0xFF 0xFF
      // Creates 4-element 32-bit vector with 8-bit imm at positions [23:16] &
      // [55:48] Shift value 272 (0x110) for Type 8 pattern encoding Corresponds
      // to AArch64_AM::encodeAdvSIMDModImmType8()
      AssignedValue = MCOperand::createImm(272);
      return Error::success();
    default:
      return make_error<Failure>(
          Twine("Unsupported MSL shift opcode: ").concat(Twine(Opcode)));
    }
  }
  case MCOI::OperandType::OPERAND_PCREL:
  case MCOI::OperandType::OPERAND_FIRST_TARGET:
    AssignedValue = MCOperand::createImm(0);
    return Error::success();
  default:
    break;
  }

  return make_error<Failure>(
      Twine("Unimplemented operand type: MCOI::OperandType:")
          .concat(Twine(static_cast<int>(OperandType))));
}

} // namespace

static ExegesisTarget *getTheExegesisAArch64Target() {
  static ExegesisAArch64Target Target;
  return &Target;
}

void InitializeAArch64ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisAArch64Target());
}

} // namespace exegesis
} // namespace llvm
