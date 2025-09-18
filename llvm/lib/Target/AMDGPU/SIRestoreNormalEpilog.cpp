#include "SICustomBranchBundles.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "si-restore-normal-epilog"

namespace
{

class SIRestoreNormalEpilogLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIRestoreNormalEpilogLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    hoist_unrelated_copies(MF);
    normalize_ir_post_phi_elimination(MF);
    return true;
  }

  StringRef getPassName() const override {
    return "SI Restore Normal Epilog Post PHI Elimination";
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }

};

} // namespace

INITIALIZE_PASS(SIRestoreNormalEpilogLegacy, DEBUG_TYPE,
                "SI restore normal epilog", false, false)

char SIRestoreNormalEpilogLegacy::ID;
char &llvm::SIRestoreNormalEpilogLegacyID = SIRestoreNormalEpilogLegacy::ID;
