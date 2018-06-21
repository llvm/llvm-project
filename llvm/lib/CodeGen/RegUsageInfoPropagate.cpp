//=--- RegUsageInfoPropagate.cpp - Register Usage Informartion Propagation --=//
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
/// This pass iterates through MachineInstrs in a given MachineFunction and at
/// each callsite queries RegisterUsageInfo for RegMask (calculated based on
/// actual register allocation) of the callee function, if the RegMask detail
/// is available then this pass will update the RegMask of the call instruction.
/// This updated RegMask will be used by the register allocator while allocating
/// the current MachineFunction.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterUsageInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <map>
#include <string>

namespace llvm {
void initializeRegUsageInfoPropagationPassPass(PassRegistry &);
}

using namespace llvm;

#define DEBUG_TYPE "ip-regalloc"

#define RUIP_NAME "Register Usage Information Propagation"

namespace {
class RegUsageInfoPropagationPass : public MachineFunctionPass {

public:
  RegUsageInfoPropagationPass() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeRegUsageInfoPropagationPassPass(Registry);
  }

  StringRef getPassName() const override { return RUIP_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  static char ID;

private:
  static void setRegMask(MachineInstr &MI, const uint32_t *RegMask) {
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask())
        MO.setRegMask(RegMask);
    }
  }
};
} // end of anonymous namespace
char RegUsageInfoPropagationPass::ID = 0;

INITIALIZE_PASS_BEGIN(RegUsageInfoPropagationPass, "reg-usage-propagation",
                      RUIP_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(PhysicalRegisterUsageInfo)
INITIALIZE_PASS_END(RegUsageInfoPropagationPass, "reg-usage-propagation",
                    RUIP_NAME, false, false)

FunctionPass *llvm::createRegUsageInfoPropPass() {
  return new RegUsageInfoPropagationPass();
}

void RegUsageInfoPropagationPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<PhysicalRegisterUsageInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Assumes call instructions have a single reference to a function.
static const Function *findCalledFunction(const Module &M, MachineInstr &MI) {
  for (MachineOperand &MO : MI.operands()) {
    if (MO.isGlobal())
      return dyn_cast<Function>(MO.getGlobal());

    if (MO.isSymbol())
      return M.getFunction(MO.getSymbolName());
  }

  return nullptr;
}

bool RegUsageInfoPropagationPass::runOnMachineFunction(MachineFunction &MF) {
  const Module *M = MF.getFunction().getParent();
  PhysicalRegisterUsageInfo *PRUI = &getAnalysis<PhysicalRegisterUsageInfo>();

  DEBUG(dbgs() << " ++++++++++++++++++++ " << getPassName()
               << " ++++++++++++++++++++  \n");
  DEBUG(dbgs() << "MachineFunction : " << MF.getName() << "\n");

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasCalls() && !MFI.hasTailCall())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.isCall())
        continue;
      DEBUG(dbgs()
            << "Call Instruction Before Register Usage Info Propagation : \n");
      DEBUG(dbgs() << MI << "\n");

      auto UpdateRegMask = [&](const Function *F) {
        const auto *RegMask = PRUI->getRegUsageInfo(F);
        if (!RegMask)
          return;
        setRegMask(MI, &(*RegMask)[0]);
        Changed = true;
      };

      if (const Function *F = findCalledFunction(*M, MI)) {
        UpdateRegMask(F);
      } else {
        DEBUG(dbgs() << "Failed to find call target function\n");
      }

      DEBUG(dbgs() << "Call Instruction After Register Usage Info Propagation : "
            << MI << '\n');
    }
  }

  DEBUG(dbgs() << " +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                  "++++++ \n");
  return Changed;
}
