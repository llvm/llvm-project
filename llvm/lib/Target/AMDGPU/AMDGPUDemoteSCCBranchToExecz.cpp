#include "llvm/CodeGen/MachineFunctionPass.h"

#include "AMDGPU.h"
#include "AMDGPUDemoteSCCBranchToExecz.h"

using namespace llvm;

namespace {
#define DEBUG_TYPE "amdgpu-demote-scc-to-execz"
const char PassName[] = "AMDGPU if conversion";

class AMDGPUDemoteSCCBranchToExecz {
public:
  AMDGPUDemoteSCCBranchToExecz() = default;

  bool run() { return false; }
};

class AMDGPUDemoteSCCBranchToExeczLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUDemoteSCCBranchToExeczLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    AMDGPUDemoteSCCBranchToExecz IfCvt{};
    return IfCvt.run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PassName; }
};

char AMDGPUDemoteSCCBranchToExeczLegacy::ID = 0;

} // namespace

PreservedAnalyses llvm::AMDGPUDemoteSCCBranchToExeczPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  AMDGPUDemoteSCCBranchToExecz IfCvt{};
  if (!IfCvt.run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

char &llvm::AMDGPUDemoteSCCBranchToExeczLegacyID =
    AMDGPUDemoteSCCBranchToExeczLegacy::ID;
INITIALIZE_PASS_BEGIN(AMDGPUDemoteSCCBranchToExeczLegacy, DEBUG_TYPE, PassName,
                      false, false)
INITIALIZE_PASS_END(AMDGPUDemoteSCCBranchToExeczLegacy, DEBUG_TYPE, PassName,
                    false, false)
