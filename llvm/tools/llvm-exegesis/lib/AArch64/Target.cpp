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
#include <linux/prctl.h> // For PR_PAC_* constants
#include <sys/prctl.h>
#ifndef PR_PAC_SET_ENABLED_KEYS
#define PR_PAC_SET_ENABLED_KEYS 60
#endif
#ifndef PR_PAC_GET_ENABLED_KEYS
#define PR_PAC_GET_ENABLED_KEYS 61
#endif
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

bool isPointerAuth(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;

  // FIXME: Pointer Authentication instructions.
  // We would like to measure these instructions, but they can behave
  // differently on different platforms, and maybe the snippets need to look
  // different for these instructions,
  // Platform-specific handling:  On Linux, we disable authentication, may
  // interfere with measurements. On non-Linux platforms, disable opcodes for
  // now.
  case AArch64::AUTDA:
  case AArch64::AUTDB:
  case AArch64::AUTDZA:
  case AArch64::AUTDZB:
  case AArch64::AUTIA:
  case AArch64::AUTIA1716:
  case AArch64::AUTIASP:
  case AArch64::AUTIAZ:
  case AArch64::AUTIB:
  case AArch64::AUTIB1716:
  case AArch64::AUTIBSP:
  case AArch64::AUTIBZ:
  case AArch64::AUTIZA:
  case AArch64::AUTIZB:
    return true;
  }
}

bool isLoadTagMultiple(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;

  // Load tag multiple instruction
  case AArch64::LDGM:
    return true;
  }
}

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

  const char *getIgnoredOpcodeReasonOrNull(const LLVMState &State,
                                           unsigned Opcode) const override {
    if (const char *Reason =
            ExegesisTarget::getIgnoredOpcodeReasonOrNull(State, Opcode))
      return Reason;

    if (isPointerAuth(Opcode)) {
#if defined(__aarch64__) && defined(__linux__)
      // Disable all PAC keys. Note that while we expect the measurements to
      // be the same with PAC keys disabled, they could potentially be lower
      // since authentication checks are bypassed.
      if (prctl(PR_PAC_SET_ENABLED_KEYS,
                PR_PAC_APIAKEY | PR_PAC_APIBKEY | PR_PAC_APDAKEY |
                    PR_PAC_APDBKEY, // all keys
                0,                  // disable all
                0, 0) < 0) {
        return "Failed to disable PAC keys";
      }
#else
      return "Unsupported opcode: isPointerAuth";
#endif
    }

    if (isLoadTagMultiple(Opcode))
      return "Unsupported opcode: load tag multiple";

    return nullptr;
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
