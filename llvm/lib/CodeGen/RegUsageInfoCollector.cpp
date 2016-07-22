//===-- RegUsageInfoCollector.cpp - Register Usage Information Collector --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This pass is required to take advantage of the interprocedural register
/// allocation infrastructure.
///
/// This pass is simple MachineFunction pass which collects register usage
/// details by iterating through each physical registers and checking
/// MRI::isPhysRegUsed() then creates a RegMask based on this details.
/// The pass then stores this RegMask in PhysicalRegisterUsageInfo.cpp
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterUsageInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"

using namespace llvm;

#define DEBUG_TYPE "ip-regalloc"

STATISTIC(NumCSROpt,
          "Number of functions optimized for callee saved registers");

namespace llvm {
void initializeRegUsageInfoCollectorPass(PassRegistry &);
}

namespace {
class RegUsageInfoCollector : public MachineFunctionPass {
public:
  RegUsageInfoCollector() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeRegUsageInfoCollectorPass(Registry);
  }

  const char *getPassName() const override {
    return "Register Usage Information Collector Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;
};
} // end of anonymous namespace

char RegUsageInfoCollector::ID = 0;

INITIALIZE_PASS_BEGIN(RegUsageInfoCollector, "RegUsageInfoCollector",
                      "Register Usage Information Collector", false, false)
INITIALIZE_PASS_DEPENDENCY(PhysicalRegisterUsageInfo)
INITIALIZE_PASS_END(RegUsageInfoCollector, "RegUsageInfoCollector",
                    "Register Usage Information Collector", false, false)

FunctionPass *llvm::createRegUsageInfoCollector() {
  return new RegUsageInfoCollector();
}

void RegUsageInfoCollector::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<PhysicalRegisterUsageInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool RegUsageInfoCollector::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetMachine &TM = MF.getTarget();

  DEBUG(dbgs() << " -------------------- " << getPassName()
               << " -------------------- \n");
  DEBUG(dbgs() << "Function Name : " << MF.getName() << "\n");

  std::vector<uint32_t> RegMask;

  // Compute the size of the bit vector to represent all the registers.
  // The bit vector is broken into 32-bit chunks, thus takes the ceil of
  // the number of registers divided by 32 for the size.
  unsigned RegMaskSize = (TRI->getNumRegs() + 31) / 32;
  RegMask.resize(RegMaskSize, 0xFFFFFFFF);

  const Function *F = MF.getFunction();

  PhysicalRegisterUsageInfo *PRUI = &getAnalysis<PhysicalRegisterUsageInfo>();

  PRUI->setTargetMachine(&TM);

  DEBUG(dbgs() << "Clobbered Registers: ");

  for (unsigned PReg = 1, PRegE = TRI->getNumRegs(); PReg < PRegE; ++PReg)
    if (MRI->isPhysRegModified(PReg, true))
      RegMask[PReg / 32] &= ~(1u << PReg % 32);

  if (!TargetFrameLowering::isSafeForNoCSROpt(F)) {
    const uint32_t *CallPreservedMask =
        TRI->getCallPreservedMask(MF, F->getCallingConv());
    // Set callee saved register as preserved.
    for (unsigned i = 0; i < RegMaskSize; ++i)
      RegMask[i] = RegMask[i] | CallPreservedMask[i];
  } else {
    ++NumCSROpt;
    DEBUG(dbgs() << MF.getName()
                 << " function optimized for not having CSR.\n");
  }

  for (unsigned PReg = 1, PRegE = TRI->getNumRegs(); PReg < PRegE; ++PReg)
    if (MachineOperand::clobbersPhysReg(&(RegMask[0]), PReg))
      DEBUG(dbgs() << TRI->getName(PReg) << " ");

  DEBUG(dbgs() << " \n----------------------------------------\n");

  PRUI->storeUpdateRegUsageInfo(F, std::move(RegMask));

  return false;
}
