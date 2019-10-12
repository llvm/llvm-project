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

// Generates instruction to load an immediate value into a register.
static MCInst loadImmediate(unsigned Reg, unsigned RegBitWidth,
                            const APInt &Value) {
  if (Value.getActiveBits() > 16)
    llvm_unreachable("Not implemented for Values wider than 16 bits");
  if (Value.getBitWidth() > RegBitWidth)
    llvm_unreachable("Value must fit in the Register");
  return MCInstBuilder(Mips::ORi)
      .addReg(Reg)
      .addReg(Mips::ZERO)
      .addImm(Value.getZExtValue());
}

std::vector<MCInst> ExegesisMipsTarget::setRegTo(const MCSubtargetInfo &STI,
                                                 unsigned Reg,
                                                 const APInt &Value) const {
  if (Mips::GPR32RegClass.contains(Reg))
    return {loadImmediate(Reg, 32, Value)};
  if (Mips::GPR64RegClass.contains(Reg))
    return {loadImmediate(Reg, 64, Value)};
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
