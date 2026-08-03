//===-- PISAVerifyTypes.cpp - modify function signatures ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mixing of scalar/integer/float types causes issues with extended LLT
// (scalar == integer, scalar == float, but integer != float), especially
// with combiner, e.g. folding multiple COPY into illegal instruction.
//
// We disallow usage of ANY_SCALAR types when extended LLT is on.
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "pisa-verify-types"
#define DEBUG_NAME "PISA verify types"

static cl::opt<bool> EnableVerifyTypes("pisa-verify-extended-types",
                                       cl::desc("Enable PISA verify types"),
#ifndef NDEBUG
                                       cl::init(true),
#else  // NDEBUG
                                       cl::init(false),
#endif // NDEBUG
                                       cl::Hidden);

namespace {

class PISAVerifyTypes : public MachineFunctionPass {
public:
  static char ID;

  PISAVerifyTypes() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char PISAVerifyTypes::ID = 0;
INITIALIZE_PASS(PISAVerifyTypes, DEBUG_TYPE, DEBUG_NAME, false, false)

bool PISAVerifyTypes::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableVerifyTypes)
    return false;

  // TODO: remove next 2 lines after extended LLT is enabled
  if (!LLT::getUseExtended())
    return false;

  assert(LLT::getUseExtended() &&
         "PISAVerifyTypes only works with extended LLT");
  // verify that we do not have any scalar types
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (!MI.isPreISelOpcode())
        continue;
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        auto &MO = MI.getOperand(I);
        if (!MO.isReg())
          continue;
        auto RegTy = MF.getRegInfo().getType(MO.getReg());
        if (RegTy.getScalarType().getKind() == LLT::Kind::ANY_SCALAR)
          MI.emitGenericError("use of scalar types in " +
                              MI.getMF()->getName() +
                              " not supported with extendedLLT");
      }
    }
  }
  return false;
}

MachineFunctionPass *llvm::createPISAVerifyTypesPass() {
  return new PISAVerifyTypes();
}
