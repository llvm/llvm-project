//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "../Latency.h"
#include "AArch64.h"
#include "AArch64RegisterInfo.h"

namespace llvm {
namespace exegesis {

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 32:
    return llvm::AArch64::MOVi32imm;
  case 64:
    return llvm::AArch64::MOVi64imm;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an immediate value into a register.
static llvm::MCInst loadImmediate(unsigned Reg, unsigned RegBitWidth,
                                  const llvm::APInt &Value) {
  if (Value.getBitWidth() > RegBitWidth)
    llvm_unreachable("Value must fit in the Register");
  return llvm::MCInstBuilder(getLoadImmediateOpcode(RegBitWidth))
      .addReg(Reg)
      .addImm(Value.getZExtValue());
}

#include "AArch64GenExegesis.inc"

namespace {

class ExegesisAArch64Target : public ExegesisTarget {
public:
  ExegesisAArch64Target() : ExegesisTarget(AArch64CpuPfmCounters) {}

private:
  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     unsigned Reg,
                                     const llvm::APInt &Value) const override {
    if (llvm::AArch64::GPR32RegClass.contains(Reg))
      return {loadImmediate(Reg, 32, Value)};
    if (llvm::AArch64::GPR64RegClass.contains(Reg))
      return {loadImmediate(Reg, 64, Value)};
    llvm::errs() << "setRegTo is not implemented, results will be unreliable\n";
    return {};
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::aarch64 || Arch == llvm::Triple::aarch64_be;
  }

  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override {
    // Function return is a pseudo-instruction that needs to be expanded
    PM.add(llvm::createAArch64ExpandPseudoPass());
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
