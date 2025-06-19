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
  MCRegister TempReg = AArch64::X16;
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

static void generateRegisterStackPush(unsigned int RegToPush,
                                      std::vector<MCInst> &GeneratedCode,
                                      int imm = -16) {
  // STR [X|W]t, [SP, #simm]!: SP is decremented by default 16 bytes
  //                           before the store to maintain 16-bytes alignment.
  if (AArch64::GPR64RegClass.contains(RegToPush)) {
    GeneratedCode.push_back(MCInstBuilder(AArch64::STRXpre)
                                .addReg(AArch64::SP)
                                .addReg(RegToPush)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  } else if (AArch64::GPR32RegClass.contains(RegToPush)) {
    GeneratedCode.push_back(MCInstBuilder(AArch64::STRWpre)
                                .addReg(AArch64::SP)
                                .addReg(RegToPush)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  } else {
    llvm_unreachable("Unsupported register class for stack push");
  }
}

static void generateRegisterStackPop(unsigned int RegToPopTo,
                                     std::vector<MCInst> &GeneratedCode,
                                     int imm = 16) {
  // LDR Xt, [SP], #simm: SP is incremented by default 16 bytes after the load.
  if (AArch64::GPR64RegClass.contains(RegToPopTo)) {
    GeneratedCode.push_back(MCInstBuilder(AArch64::LDRXpost)
                                .addReg(AArch64::SP)
                                .addReg(RegToPopTo)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  } else if (AArch64::GPR32RegClass.contains(RegToPopTo)) {
    GeneratedCode.push_back(MCInstBuilder(AArch64::LDRWpost)
                                .addReg(AArch64::SP)
                                .addReg(RegToPopTo)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  } else {
    llvm_unreachable("Unsupported register class for stack pop");
  }
}

void generateSysCall(long SyscallNumber, std::vector<MCInst> &GeneratedCode) {
  GeneratedCode.push_back(
      loadImmediate(AArch64::X8, 64, APInt(64, SyscallNumber)));
  GeneratedCode.push_back(MCInstBuilder(AArch64::SVC).addImm(0));
}

/// Functions to save/restore system call registers
#ifdef __linux__
constexpr std::array<unsigned, 6> SyscallArgumentRegisters{
    AArch64::X0, AArch64::X1, AArch64::X2,
    AArch64::X3, AArch64::X4, AArch64::X5,
};

static void saveSysCallRegisters(std::vector<MCInst> &GeneratedCode,
                                 unsigned ArgumentCount) {
  // AArch64 Linux typically uses X0-X5 for the first 6 arguments.
  // Some syscalls can take up to 8 arguments in X0-X7.
  assert(ArgumentCount <= 6 &&
         "This implementation saves up to 6 argument registers (X0-X5)");
  // generateRegisterStackPush(ArgumentRegisters::TempRegister, GeneratedCode);
  // Preserve X8 (used for the syscall number/return value).
  generateRegisterStackPush(AArch64::X8, GeneratedCode);
  // Preserve the registers used to pass arguments to the system call.
  for (unsigned I = 0; I < ArgumentCount; ++I) {
    generateRegisterStackPush(SyscallArgumentRegisters[I], GeneratedCode);
  }
}

static void restoreSysCallRegisters(std::vector<MCInst> &GeneratedCode,
                                    unsigned ArgumentCount) {
  assert(ArgumentCount <= 6 &&
         "This implementation restores up to 6 argument registers (X0-X5)");
  // Restore argument registers, in opposite order of the way they are saved.
  for (int I = ArgumentCount - 1; I >= 0; --I) {
    generateRegisterStackPop(SyscallArgumentRegisters[I], GeneratedCode);
  }
  generateRegisterStackPop(AArch64::X8, GeneratedCode);
  // generateRegisterStackPop(ArgumentRegisters::TempRegister, GeneratedCode);
}
#endif // __linux__
#include "AArch64GenExegesis.inc"

namespace {

class ExegesisAArch64Target : public ExegesisTarget {
public:
  ExegesisAArch64Target()
      : ExegesisTarget(AArch64CpuPfmCounters, AArch64_MC::isOpcodeAvailable) {}

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
