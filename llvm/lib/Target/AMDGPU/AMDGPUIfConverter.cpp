#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineBranchProbabilityInfo.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/SSAIfConv.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSchedule.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/InitializePasses.h>

#include "AMDGPU.h"

using namespace llvm;

namespace {
#define DEBUG_TYPE "amdgpu-if-cvt"
const char PassName[] = "AMDGPU if conversion";

class AMDGPUIfConverter : public MachineFunctionPass {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  TargetSchedModel SchedModel;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineBranchProbabilityInfo *MBPI = nullptr;
  MachineLoopInfo *Loops = nullptr;

  static constexpr unsigned BlockInstrLimit = 30;
  static constexpr bool Stress = false;
  SSAIfConv IfConv{DEBUG_TYPE, BlockInstrLimit, Stress};

public:
  static char ID;

  AMDGPUIfConverter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool tryConvertIf(MachineBasicBlock *);

  StringRef getPassName() const override { return PassName; }
};

char AMDGPUIfConverter::ID = 0;

void AMDGPUIfConverter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool AMDGPUIfConverter::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  TII = STI.getInstrInfo();
  TRI = STI.getRegisterInfo();
  MRI = &MF.getRegInfo();
  SchedModel.init(&STI);
  DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  bool Changed = false;
  IfConv.runOnMachineFunction(MF);

  for (auto *DomNode : post_order(DomTree))
    if (tryConvertIf(DomNode->getBlock()))
      Changed = true;

  return Changed;
}

bool AMDGPUIfConverter::tryConvertIf(MachineBasicBlock *MBB) { return false; }

} // namespace

char &llvm::AMDGPUIfConverterID = AMDGPUIfConverter::ID;
INITIALIZE_PASS_BEGIN(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)