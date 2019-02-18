//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     unsigned Reg,
                                     const llvm::APInt &Value) const override;
  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::ppc64le;
  }
};
} // end anonymous namespace

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 32:
    return llvm::PPC::LI;
  case 64:
    return llvm::PPC::LI8;
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

std::vector<llvm::MCInst>
ExegesisPowerPCTarget::setRegTo(const llvm::MCSubtargetInfo &STI, unsigned Reg,
                                const llvm::APInt &Value) const {
  if (llvm::PPC::GPRCRegClass.contains(Reg))
    return {loadImmediate(Reg, 32, Value)};
  if (llvm::PPC::G8RCRegClass.contains(Reg))
    return {loadImmediate(Reg, 64, Value)};
  llvm::errs() << "setRegTo is not implemented, results will be unreliable\n";
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
