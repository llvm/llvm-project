#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include "AMDGPUSSARAUtils.h"

#include <stack>

#include "VRegMaskPair.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  typedef struct {
    Register CurName;
    LaneBitmask PrevMask;
    unsigned PrevSubRegIdx;
    MachineInstr *DefMI;
  } CurVRegInfo;

  using VRegDefStack = std::vector<CurVRegInfo>;

  SetVector<VRegMaskPair> CrossBlockVRegs;
  DenseMap<VRegMaskPair, SmallPtrSet<MachineBasicBlock *, 8>> DefBlocks;
  DenseMap<VRegMaskPair, SmallPtrSet<MachineBasicBlock *, 8>> LiveInBlocks;
  DenseMap<unsigned, SetVector<VRegMaskPair>> PHINodes;
  DenseMap<MachineInstr *, VRegMaskPair> PHIMap;
  DenseSet<unsigned> DefSeen;
  DenseSet<unsigned> Renamed;

  void collectCrossBlockVRegs(MachineFunction &MF);
  void findPHINodesPlacement(const SmallPtrSetImpl<MachineBasicBlock *> &LiveInBlocks,
                          const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                          SmallVectorImpl<MachineBasicBlock *> &PHIBlocks) {
    
    IDFCalculatorBase<MachineBasicBlock, false> IDF(MDT->getBase());

    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    IDF.calculate(PHIBlocks);
  }

  MachineOperand &rewriteUse(MachineOperand &Op, MachineBasicBlock::iterator I,
                             MachineBasicBlock &MBB,
                             DenseMap<unsigned, VRegDefStack> VregNames) {
    // Sub-reg handling:
    // 1. if (UseMask & ~DefMask) != 0 : current Def does not define all used
    // lanes. We should search names stack for the Def that defines missed
    // lanes to construct the REG_SEQUENCE
    // 2. if (UseMask & DefMask) == 0 : current Def defines subregisters of a
    // register which are not used by the current Use. We should search names
    // stack for the corresponding sub-register def. Replace reg.subreg in Use
    // only if VReg.subreg found != current VReg.subreg in use!
    // 3. (UseMask & DefMask) == UseMask just replace the reg if the reg found
    // != current reg in Use. Take care of the subreg in Use. If (DefMask |
    // UseMask) != UseMask, i.e. current Def defines more lanes that is used
    // by the current Use, we need to calculate the corresponding subreg index
    // for the Use. DefinedLanes serves as a result of the expression
    // mentioned above. UndefSubRegs initially is set to UseMask but is
    // updated on each iteration if we are looking for the sub-regs
    // definitions to compose REG_SEQUENCE.
    bool RewriteOp = true;
    unsigned VReg = Op.getReg();
    assert(!VregNames[VReg].empty() &&
           "Error: use does not dominated by definition!\n");
    SmallVector<std::tuple<unsigned, unsigned, unsigned>> RegSeqOps;
    LaneBitmask UseMask = getOperandLaneMask(Op, TRI, MRI);
    dbgs() << "Use mask : " << PrintLaneMask(UseMask) << "\n";
    LaneBitmask UndefSubRegs = UseMask;
    LaneBitmask DefinedLanes = LaneBitmask::getNone();
    unsigned SubRegIdx = AMDGPU::NoRegister;
    dbgs() << "Looking for appropriate definiton...\n";
    Register CurVReg = AMDGPU::NoRegister;
    VRegDefStack VregDefs = VregNames[VReg];
    VRegDefStack::reverse_iterator It = VregDefs.rbegin();
    for (; It != VregDefs.rend(); ++It) {
      CurVRegInfo VRInfo = *It;
      dbgs() << "Def:\n";
      CurVReg = VRInfo.CurName;
      MachineInstr *DefMI = VRInfo.DefMI;
      MachineOperand *DefOp = DefMI->findRegisterDefOperand(CurVReg, TRI);
      const TargetRegisterClass *RC =
          TRI->getRegClassForOperandReg(*MRI, *DefOp);
      dbgs() << "DefMI: " << *DefMI << "\n";
      dbgs() << "Operand: " << *DefOp << "\n";
      LaneBitmask DefMask = VRInfo.PrevMask;
      dbgs() << "Def mask : " << PrintLaneMask(DefMask) << "\n";
      LaneBitmask LanesDefinedyCurrentDef = (UndefSubRegs & DefMask) & UseMask;
      dbgs() << "Lanes defined by current Def: "
             << PrintLaneMask(LanesDefinedyCurrentDef) << "\n";
      DefinedLanes |= LanesDefinedyCurrentDef;
      dbgs() << "Total defined lanes: " << PrintLaneMask(DefinedLanes) << "\n";

      if (LanesDefinedyCurrentDef == UseMask) {
        // All lanes used here are defined by this def.
        if (CurVReg == VReg && Op.getSubReg() == DefOp->getSubReg()) {
          // Need nothing - bail out.
          RewriteOp = false;
          break;
        }
        SubRegIdx = DefOp->getSubReg();
        if ((DefMask & ~UseMask).any()) {
          // Definition defines more lanes then used. Need sub register
          // index;
          SubRegIdx = getSubRegIndexForLaneMask(UseMask, TRI);
        }
        break;
      }

      if (LanesDefinedyCurrentDef.any()) {
        // Current definition defines some of the lanes used here.
        unsigned DstSubReg =
            getSubRegIndexForLaneMask(LanesDefinedyCurrentDef, TRI);
        if (!DstSubReg) {
          SmallVector<unsigned> Idxs =
              getCoveringSubRegsForLaneMask(LanesDefinedyCurrentDef, RC, TRI);
          for (unsigned SubIdx : Idxs) {
            dbgs() << "Matching subreg: " << SubIdx << " : "
                   << PrintLaneMask(TRI->getSubRegIndexLaneMask(SubIdx))
                   << "\n";
            RegSeqOps.push_back({CurVReg, SubIdx, SubIdx});
          }
        } else {
          unsigned SrcSubReg = (DefMask & ~LanesDefinedyCurrentDef).any()
                                   ? DstSubReg
                                   : DefOp->getSubReg();
          RegSeqOps.push_back({CurVReg, SrcSubReg, DstSubReg});
        }
        UndefSubRegs = UseMask & ~DefinedLanes;
        dbgs() << "UndefSubRegs: " << PrintLaneMask(UndefSubRegs) << "\n";
        if (UndefSubRegs.none())
          break;
      } else {
        // The current definition does not define any of the lanes used
        // here. Continue to search for the definition.
        dbgs() << "No lanes defined by this def!\n";
        continue;
      }
    }

    if (UndefSubRegs != UseMask && !UndefSubRegs.none()) {
      // WE haven't found all sub-regs definition. Assume undef.
      // Insert IMPLISIT_DEF

      const TargetRegisterClass *RC = TRI->getRegClassForOperandReg(*MRI, Op);
      SmallVector<unsigned> Idxs =
          getCoveringSubRegsForLaneMask(UndefSubRegs, RC, TRI);
      for (unsigned SubIdx : Idxs) {
        const TargetRegisterClass *SubRC = TRI->getSubRegisterClass(RC, SubIdx);
        Register NewVReg = MRI->createVirtualRegister(SubRC);
        BuildMI(MBB, I, I->getDebugLoc(), TII->get(AMDGPU::IMPLICIT_DEF))
            .addReg(NewVReg, RegState::Define);
        RegSeqOps.push_back({NewVReg, AMDGPU::NoRegister, SubIdx});
      }
    }

    if (!RegSeqOps.empty()) {
      // All subreg defs are found. Insert REG_SEQUENCE.
      auto *RC = TRI->getRegClassForReg(*MRI, VReg);
      CurVReg = MRI->createVirtualRegister(RC);
      auto RS = BuildMI(MBB, I, I->getDebugLoc(), TII->get(AMDGPU::REG_SEQUENCE),
                        CurVReg);
      for (auto O : RegSeqOps) {
        auto [R, SrcSubreg, DstSubreg] = O;
        RS.addReg(R, 0, SrcSubreg);
        RS.addImm(DstSubreg);
      }

      VregNames[VReg].push_back(
          {CurVReg, getFullMaskForRC(*RC, TRI), AMDGPU::NoRegister, RS});
    }

    assert(CurVReg != AMDGPU::NoRegister &&
           "Use is not dominated by definition!\n");

    dbgs() << "Rewriting use: " << Op << " to "
           << printReg(CurVReg, TRI, SubRegIdx, MRI) << "\n";

    if (RewriteOp) {
      Op.setReg(CurVReg);
      Op.setSubReg(SubRegIdx);
    }
    return Op;
  }

  void renameVRegs(MachineBasicBlock &MBB,
                   DenseMap<unsigned, VRegDefStack> VregNames) {
    for (auto &PHI : MBB.phis()) {
      MachineOperand &Op = PHI.getOperand(0);
      Register Res = Op.getReg();
      unsigned SubRegIdx = Op.getSubReg();
      const TargetRegisterClass *RC =
          SubRegIdx ? TRI->getSubRegisterClass(
                          TRI->getRegClassForReg(*MRI, Res), SubRegIdx)
                    : TRI->getRegClassForReg(*MRI, Res);
      Register NewVReg = MRI->createVirtualRegister(RC);
      Op.setReg(NewVReg);
      Op.setSubReg(AMDGPU::NoRegister);
      VregNames[Res].push_back(
          {NewVReg, getFullMaskForRC(*RC, TRI), AMDGPU::NoRegister, &PHI});
      DefSeen.insert(NewVReg);
      Renamed.insert(Res);
    }
    for (auto &I : make_range(MBB.getFirstNonPHI(), MBB.end())) {


      for (auto &Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual() &&
            Renamed.contains(Op.getReg())) {
          Op = rewriteUse(Op, I, MBB, VregNames);
        }
      }

      for (auto &Op : I.defs()) {
        if (Op.getReg().isVirtual()) {
          unsigned VReg = Op.getReg();
          if (DefSeen.contains(VReg)) {
            const TargetRegisterClass *RC =
                TRI->getRegClassForOperandReg(*MRI, Op);
            Register NewVReg = MRI->createVirtualRegister(RC);
            VregNames[VReg].push_back({NewVReg,
                                       getOperandLaneMask(Op, TRI, MRI),
                                       Op.getSubReg(), &I});

            Op.ChangeToRegister(NewVReg, true, false, false, false, false);
            Op.setSubReg(AMDGPU::NoRegister);
            LLVM_DEBUG(dbgs()
                       << "Renaming VReg: " << Register::virtReg2Index(VReg)
                       << " to " << Register::virtReg2Index(NewVReg) << "\n");
            Renamed.insert(VReg);
          } else {
            VregNames[VReg].push_back(
                {VReg, getOperandLaneMask(Op, TRI, MRI), Op.getSubReg(), &I});
            DefSeen.insert(VReg);
          }
        }
      }
    }

    for (auto Succ : successors(&MBB)) {
      for (auto &PHI : Succ->phis()) {
        VRegMaskPair VMP = PHIMap[&PHI];
        // unsigned SubRegIdx = AMDGPU::NoRegister;
        // const TargetRegisterClass *RC =
        //     TRI->getRegClassForReg(*MRI, VMP.getVReg());
        // LaneBitmask FullMask = getFullMaskForRC(*RC, TRI);
        // if (VMP.getLaneMask() != FullMask) {
        //   SubRegIdx = getSubRegIndexForLaneMask(VMP.getLaneMask(), TRI);
        // }
        unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);
        if (VregNames[VMP.getVReg()].empty()) {
          PHI.addOperand(MachineOperand::CreateReg(VMP.getVReg(), false, false,
                                                   false, false, false, false,
                                                   SubRegIdx));
        } else {
          MachineOperand Op =
              MachineOperand::CreateReg(VMP.getVReg(), false, false, false,
                                        false, false, false, SubRegIdx);
          MachineBasicBlock::iterator IP = MBB.getFirstTerminator();
          Op = rewriteUse(Op, IP, MBB, VregNames);
          PHI.addOperand(Op);
        }
        PHI.addOperand(MachineOperand::CreateMBB(&MBB));
      }
    }
    // recurse to the succs in DomTree
    DomTreeNodeBase<MachineBasicBlock> *Node = MDT->getNode(&MBB);
    for (auto *Child : Node->children()) {
      MachineBasicBlock *ChildMBB = Child->getBlock();
      // Process child in the dominator tree
      renameVRegs(*ChildMBB, VregNames);
    }
  }

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

void AMDGPURebuildSSALegacy::collectCrossBlockVRegs(MachineFunction &MF) {
  for (auto &MBB : MF) {
    SetVector<VRegMaskPair> Killed;
    for (auto &I : MBB) {
      for (auto Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          VRegMaskPair VMP(Op, TRI, MRI);
          if (!Killed.contains(VMP))
            CrossBlockVRegs.insert(VMP);
        }
      }
      for (auto Op : I.defs()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          VRegMaskPair VMP(Op, TRI, MRI);
          Killed.insert(VMP);
          DefBlocks[VMP].insert(&MBB);
        }
      }
    }
  }
}

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  CrossBlockVRegs.clear();
  DefBlocks.clear();
  LiveInBlocks.clear();
  PHINodes.clear();
  DefSeen.clear();
  Renamed.clear();

  // Collect all cross-block virtual registers.
  // This includes registers that are live-in to the function, and registers
  // that are defined in multiple blocks.
  // We will insert PHI nodes for these registers.
  collectCrossBlockVRegs(MF);

  LLVM_DEBUG(dbgs() << "##### Virt regs live cross block ##################\n";
             for (auto VMP
                  : CrossBlockVRegs) {
               dbgs() << Register::virtReg2Index(VMP.getVReg()) << " ";
             } dbgs()
             << "\n");

  for (auto VMP : CrossBlockVRegs) {
    SmallVector<MachineBasicBlock *> PHIBlocks;
    for (auto &MBB : MF) {
      LiveRange &LR = LIS->getInterval(VMP.getVReg());
      if (LIS->isLiveInToMBB(LR, &MBB))
        LiveInBlocks[VMP].insert(&MBB);
    }

    LLVM_DEBUG(
        dbgs() << "findPHINodesPlacement input:\nVreg: "
               << Register::virtReg2Index(VMP.getVReg()) << "\n";
        dbgs() << "Def Blocks: \n"; for (auto MBB
                                         : DefBlocks[VMP]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs() << "\nLiveIn Blocks: \n";
        for (auto MBB
             : LiveInBlocks[VMP]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs()
        << "\n");

    findPHINodesPlacement(LiveInBlocks[VMP], DefBlocks[VMP],
                          PHIBlocks);
    LLVM_DEBUG(dbgs() << "\nBlocks to insert PHI nodes:\n";
               for (auto MBB : PHIBlocks) {
                 dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
               } dbgs()
               << "\n");
    for (auto MBB : PHIBlocks) {
      if (!PHINodes[MBB->getNumber()].contains(VMP)) {
        // Insert PHI for VReg. Don't use new VReg here as we'll replace them
        // in renaming phase.
        unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);
        dbgs() << printReg(VMP.getVReg(), TRI, SubRegIdx) << "\n";
        auto PHINode =
            BuildMI(*MBB, MBB->begin(), DebugLoc(), TII->get(TargetOpcode::PHI))
                .addReg(VMP.getVReg(), RegState::Define, SubRegIdx);
        PHINodes[MBB->getNumber()].insert(VMP);
        PHIMap[PHINode] = VMP;
      }
    }
  }

    // Rename virtual registers in the basic block.
  DenseMap<unsigned, VRegDefStack> VregNames;
  renameVRegs(MF.front(), VregNames);
  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
  MF.getProperties().reset(MachineFunctionProperties::Property ::NoPHIs);
  return MRI->isSSA();
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

// Legacy PM registration
FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

PreservedAnalyses
llvm::AMDGPURebuildSSAPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  AMDGPURebuildSSALegacy Impl;
  bool Changed = Impl.runOnMachineFunction(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: We could detect CFG changed.
  auto PA = getMachineFunctionPassPreservedAnalyses();
  return PA;
}

llvm::PassPluginLibraryInfo getAMDGPURebuildSSAPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPURebuildSSA", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "amdgpu-rebuild-ssa") {
                    PM.addPass(AMDGPURebuildSSAPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Expose the pass to LLVM’s pass manager infrastructure
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPURebuildSSAPassPluginInfo();
}