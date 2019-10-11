//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The PowerPC ExegesisTarget.
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "../Latency.h"
#include "PPC.h"
#include "PPCRegisterInfo.h"

namespace llvm {
namespace exegesis {

#include "PPCGenExegesis.inc"

namespace {
class ExegesisPowerPCTarget : public ExegesisTarget {
public:
  ExegesisPowerPCTarget() : ExegesisTarget(PPCCpuPfmCounters) {}

private:
  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, unsigned Reg,
                               const APInt &Value) const override;
  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::ppc64le;
  }
};
} // end anonymous namespace

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 32:
    return PPC::LI;
  case 64:
    return PPC::LI8;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an immediate value into a register.
static MCInst loadImmediate(unsigned Reg, unsigned RegBitWidth,
                            const APInt &Value) {
  if (Value.getBitWidth() > RegBitWidth)
    llvm_unreachable("Value must fit in the Register");
  return MCInstBuilder(getLoadImmediateOpcode(RegBitWidth))
      .addReg(Reg)
      .addImm(Value.getZExtValue());
}

std::vector<MCInst> ExegesisPowerPCTarget::setRegTo(const MCSubtargetInfo &STI,
                                                    unsigned Reg,
                                                    const APInt &Value) const {
  if (PPC::GPRCRegClass.contains(Reg))
    return {loadImmediate(Reg, 32, Value)};
  if (PPC::G8RCRegClass.contains(Reg))
    return {loadImmediate(Reg, 64, Value)};
  errs() << "setRegTo is not implemented, results will be unreliable\n";
  return {};
}

static ExegesisTarget *getTheExegesisPowerPCTarget() {
  static ExegesisPowerPCTarget Target;
  return &Target;
}

void InitializePowerPCExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisPowerPCTarget());
}

} // namespace exegesis
} // namespace llvm
