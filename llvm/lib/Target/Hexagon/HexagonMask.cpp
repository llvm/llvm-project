//===-- HexagonMask.cpp - replace const ext tfri with mask ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mask"

#include "HexagonSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace llvm {
FunctionPass *createHexagonMask();
void initializeHexagonMaskPass(PassRegistry &);

class HexagonMask : public MachineFunctionPass {
public:
  static char ID;
  HexagonMask() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeHexagonMaskPass(Registry);
  }

  StringRef getPassName() const override {
    return "Hexagon replace const ext tfri with mask";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const HexagonInstrInfo *HII;
  void replaceConstExtTransferImmWithMask(MachineFunction &MF);
};

char HexagonMask::ID = 0;

void HexagonMask::replaceConstExtTransferImmWithMask(MachineFunction &MF) {
  for (auto &MBB : MF) {
    for (auto &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != Hexagon::A2_tfrsi)
        continue;

      const MachineOperand &Op0 = MI.getOperand(0);
      const MachineOperand &Op1 = MI.getOperand(1);
      if (!Op1.isImm())
        continue;
      int32_t V = Op1.getImm();
      if (isInt<16>(V))
        continue;

      unsigned Idx, Len;
      if (!isShiftedMask_32(V, Idx, Len))
        continue;
      if (!isUInt<5>(Idx) || !isUInt<5>(Len))
        continue;

      BuildMI(MBB, MI, MI.getDebugLoc(), HII->get(Hexagon::S2_mask),
              Op0.getReg())
          .addImm(Len)
          .addImm(Idx);
      MBB.erase(MI);
    }
  }
}

bool HexagonMask::runOnMachineFunction(MachineFunction &MF) {
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  HII = HST.getInstrInfo();
  const Function &F = MF.getFunction();

  if (!F.hasFnAttribute(Attribute::OptimizeForSize))
    return false;
  // Mask instruction is available only from v66
  if (!HST.hasV66Ops())
    return false;
  // The mask instruction available in v66 can be used to generate values in
  // registers using 2 immediates Eg. to form 0x07fffffc in R0, you would write
  // "R0 = mask(#25,#2)" Since it is a single-word instruction, it takes less
  // code size than a constant-extended transfer at Os
  replaceConstExtTransferImmWithMask(MF);

  return true;
}

} // namespace llvm

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

INITIALIZE_PASS(HexagonMask, "hexagon-mask", "Hexagon mask", false, false)

FunctionPass *llvm::createHexagonMask() { return new HexagonMask(); }
