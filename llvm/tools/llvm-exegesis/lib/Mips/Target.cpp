//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "../Latency.h"
#include "Mips.h"
#include "MipsRegisterInfo.h"

namespace llvm {
namespace exegesis {

#include "MipsGenExegesis.inc"

namespace {
class ExegesisMipsTarget : public ExegesisTarget {
public:
  ExegesisMipsTarget() : ExegesisTarget(MipsCpuPfmCounters) {}

private:
  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, unsigned Reg,
                               const APInt &Value) const override;
  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::mips || Arch == Triple::mipsel ||
           Arch == Triple::mips64 || Arch == Triple::mips64el;
  }
};
} // end anonymous namespace

// Generates instructions to load an immediate value into a register.
static std::vector<MCInst> loadImmediate(unsigned Reg, bool IsGPR32,
                                         const APInt &Value) {
  unsigned ZeroReg;
  unsigned ORi, LUi, SLL;
  if (IsGPR32) {
    ZeroReg = Mips::ZERO;
    ORi = Mips::ORi;
    SLL = Mips::SLL;
    LUi = Mips::LUi;
  } else {
    ZeroReg = Mips::ZERO_64;
    ORi = Mips::ORi64;
    SLL = Mips::SLL64_64;
    LUi = Mips::LUi64;
  }

  if (Value.isIntN(16)) {
    return {MCInstBuilder(ORi)
        .addReg(Reg)
        .addReg(ZeroReg)
        .addImm(Value.getZExtValue())};
  }

  std::vector<MCInst> Instructions;
  if (Value.isIntN(32)) {
    const uint16_t HiBits = Value.getHiBits(16).getZExtValue();
    if (!IsGPR32 && Value.getActiveBits() == 32) {
      // Expand to an ORi instead of a LUi to avoid sign-extending into the
      // upper 32 bits.
      Instructions.push_back(
          MCInstBuilder(ORi)
              .addReg(Reg)
              .addReg(ZeroReg)
              .addImm(HiBits));
      Instructions.push_back(
          MCInstBuilder(SLL)
              .addReg(Reg)
              .addReg(Reg)
              .addImm(16));
    } else {
      Instructions.push_back(
          MCInstBuilder(LUi)
              .addReg(Reg)
              .addImm(HiBits));
    }

    const uint16_t LoBits = Value.getLoBits(16).getZExtValue();
    if (LoBits) {
      Instructions.push_back(
          MCInstBuilder(ORi)
          .addReg(Reg)
          .addReg(ZeroReg)
          .addImm(LoBits));
    }

    return Instructions;
  }

  llvm_unreachable("Not implemented for values wider than 32 bits");
}

std::vector<MCInst> ExegesisMipsTarget::setRegTo(const MCSubtargetInfo &STI,
                                                 unsigned Reg,
                                                 const APInt &Value) const {
  if (Mips::GPR32RegClass.contains(Reg))
    return loadImmediate(Reg, true, Value);
  if (Mips::GPR64RegClass.contains(Reg))
    return loadImmediate(Reg, false, Value);
  errs() << "setRegTo is not implemented, results will be unreliable\n";
  return {};
}

static ExegesisTarget *getTheExegesisMipsTarget() {
  static ExegesisMipsTarget Target;
  return &Target;
}

void InitializeMipsExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisMipsTarget());
}

} // namespace exegesis
} // namespace llvm
