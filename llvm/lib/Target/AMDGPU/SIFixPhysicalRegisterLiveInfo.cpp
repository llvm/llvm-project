#include "SIFixPhysicalRegisterLiveInfo.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-physical-register-live-info"

namespace {

struct SIFixPhysicalRegisterLiveInfo {
  MachineDominatorTree *MDT;
  SIFixPhysicalRegisterLiveInfo(MachineDominatorTree *MDT) : MDT(MDT) {}
  void getAnalysisUsage(AnalysisUsage &AU) const;
  const MachineInstr *definedInBlock(const SIRegisterInfo *TRI,
                                     MachineBasicBlock *MBB, Register Reg);

  bool run(MachineFunction &MF);
};

struct SIFixPhysicalRegisterLiveInfoLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIFixPhysicalRegisterLiveInfoLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Fix Physical Register Live Info";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // end of anonymous namespace

INITIALIZE_PASS_BEGIN(SIFixPhysicalRegisterLiveInfoLegacy, DEBUG_TYPE,
                      "SI Fix Physical Register Live Info", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SIFixPhysicalRegisterLiveInfoLegacy, DEBUG_TYPE,
                    "SI Fix Physical Register Live Info", false, false)

char SIFixPhysicalRegisterLiveInfoLegacy::ID = 0;

char &llvm::SIFixPhysicalRegisterLiveInfoLegacyID =
    SIFixPhysicalRegisterLiveInfoLegacy::ID;

FunctionPass *llvm::createSIFixPhysicalRegisterLiveInfoLegacyPass() {
  return new SIFixPhysicalRegisterLiveInfoLegacy();
}

bool SIFixPhysicalRegisterLiveInfoLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  SIFixPhysicalRegisterLiveInfo Impl(&MDT);
  return Impl.run(MF);
}

PreservedAnalyses
SIFixPhysicalRegisterLiveInfoPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  SIFixPhysicalRegisterLiveInfo Impl(&MDT);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SIFixPhysicalRegisterLiveInfo::run(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (!MRI.isSSA())
    return false;

  bool Changed = false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Check SCC for now.
  for (Register Reg : {AMDGPU::SCC}) {
    for (MachineBasicBlock &MBB : MF) {
      if (!MBB.isLiveIn(Reg))
        continue;

      MachineBasicBlock *UseBB = &MBB;
      MachineBasicBlock *DefBB = nullptr;
      const MachineInstr *DefMI = nullptr;
      // The PHI node is not considered here. Theoritically, if the physical
      // register is live out of a block, the InstrEmitter in the DAG Isel will
      // copy it out.
      MachineDomTreeNode *Node = MDT->getNode(&MBB)->getIDom();
      while (Node != nullptr) {
        DefBB = Node->getBlock();
        DefMI = definedInBlock(TRI, DefBB, Reg);
        if (DefMI != nullptr)
          break;

        Node = Node->getIDom();
      }

      Register NewCopy =
          MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0_XEXECRegClass);
      if (DefMI == nullptr) {
        // kernel arguments
        MachineBasicBlock &EntryBB = MF.front();
        assert(EntryBB.isLiveIn(Reg));
        DebugLoc DL = DefMI->getDebugLoc();

        BuildMI(EntryBB, EntryBB.getFirstTerminator(), DebugLoc(),
                TII->get(TargetOpcode::COPY), NewCopy)
            .addReg(Reg);

      } else {
        DebugLoc DL = DefMI->getDebugLoc();
        // Insert a COPY just SCC instruction which defines it.
        BuildMI(*DefBB,
                std::next(const_cast<MachineInstr *>(DefMI)->getIterator()), DL,
                TII->get(TargetOpcode::COPY), NewCopy)
            .addReg(Reg);
      }

      // Copy back to $scc in the SplitBB
      MachineBasicBlock::iterator InsertPt = UseBB->getFirstNonPHI();
      BuildMI(*UseBB, InsertPt, InsertPt->getDebugLoc(),
              TII->get(TargetOpcode::COPY), Reg)
          .addReg(NewCopy, RegState::Kill);

      // Now remove the scc from the livein of UseBB
      UseBB->removeLiveIn(Reg);
    }
  }

  return Changed;
}

// Returns the iterator where the Physical Reg is defined, so that an COPY out
// will be inserted just after the definition point. Or else, the value could be
// clobbered by the following instructions.
const MachineInstr *SIFixPhysicalRegisterLiveInfo::definedInBlock(
    const SIRegisterInfo *TRI, MachineBasicBlock *MBB, Register Reg) {
  for (MachineBasicBlock::reverse_iterator I = MBB->rbegin(), E = MBB->rend();
       I != E; ++I) {
    if (I->isDebugInstr())
      continue;

    if (I->isCall()) {
      // $scc is caller saved register, so if it is clobbered, we ingore it
      // since the value of $scc is not a useful one.
      continue;
    }

    if (I->definesRegister(Reg, TRI))
      return &*I;
  }

  return nullptr;
}
