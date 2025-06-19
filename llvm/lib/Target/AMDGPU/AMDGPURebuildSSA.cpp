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

  SetVector<unsigned> CrossBlockVRegs;
  DenseMap<unsigned, SmallPtrSet<MachineBasicBlock *, 8>> DefBlocks;
  DenseMap<unsigned, SmallPtrSet<MachineBasicBlock *, 8>> LiveInBlocks;
  DenseMap<unsigned, SmallSet<unsigned, 4>> PHINodes;
  DenseMap<MachineInstr *, unsigned> PHIMap;
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

  void renameVRegs(MachineBasicBlock &MBB,
                   DenseMap<unsigned, VRegDefStack> VregNames) {
    for (auto &PHI : MBB.phis()) {
      Register Res = PHI.getOperand(0).getReg();
      const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, Res);
      Register NewVReg = MRI->createVirtualRegister(RC);
      PHI.getOperand(0).setReg(NewVReg);
      VregNames[Res].push_back(
          {NewVReg, getFullMaskForRC(*RC, TRI), AMDGPU::NoRegister, &PHI});
      DefSeen.insert(NewVReg);
    }
    for (auto &I : make_range(MBB.getFirstNonPHI(), MBB.end())) {

      // Sub-reg handling:
      // 1. if UseMask > DefMask => search names stack to construct REG_SEQUENCE
      // 2. if UseMask < DefMask => search names stack for the corresponding
      // sub-register def. Replace reg in use only if VReg found != current VReg
      // in use!
      // 3. UseMask == DefMask => just replace the reg if the reg found !=
      // current reg in use
      // DefinedLanes serves as a result of the expression mentioned above.
      // UndefSubRegs initially is set to UseMask but is updated on each
      // iteration if we are looking for the sub-regs definitions to compose
      // REG_SEQUENCE.
      for (auto &Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual() &&
            Renamed.contains(Op.getReg())) {
          bool RewriteOp = true;
          unsigned VReg = Op.getReg();
          assert(!VregNames[VReg].empty() &&
                 "Error: use does not dominated by definition!\n");
          SmallVector<std::tuple<unsigned, unsigned, unsigned>> RegSeqOps;
          LaneBitmask UseMask = getOperandLaneMask(Op, TRI, MRI);
          dbgs() << "Use mask : " << PrintLaneMask(UseMask) << "\n";
          LaneBitmask UndefSubRegs = UseMask;
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
            dbgs() << "DefMI: " << *DefMI << "\n";
            dbgs() << "Operand: " << *DefOp << "\n";
            LaneBitmask DefMask = VRInfo.PrevMask;
            dbgs() << "Def mask : " << PrintLaneMask(DefMask) << "\n";

            LaneBitmask DefinedLanes = (UndefSubRegs & DefMask) & UseMask;
            dbgs() << "Defined lanes: " << PrintLaneMask(DefinedLanes)
                   << "\n";

            if (DefinedLanes == UseMask) {
              // All lanes used here are defined by this def.
              if (CurVReg == VReg && Op.getSubReg() == DefOp->getSubReg()) {
                // Need nothing - bail out.
                RewriteOp = false;
                break;
              }
              SubRegIdx = DefOp->getSubReg();
              if ((DefMask | UseMask) != UseMask) {
                // Definition defines more lanes then used. Need su register
                // index;
                SubRegIdx = getSubRegIndexForLaneMask(UseMask, TRI);
              }
              break;
            }

            if (DefinedLanes.any()) {
              // Current definition defines some of the lanes used here.
              RegSeqOps.push_back({CurVReg, DefOp->getSubReg(), It->PrevSubRegIdx});
              UndefSubRegs = UseMask & ~DefMask;
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

          if (!RegSeqOps.empty()) {
            // All subreg defs are found. Insert REG_SEQUENCE.
            auto *RC = TRI->getRegClassForReg(*MRI, VReg);
            CurVReg = MRI->createVirtualRegister(RC);
            auto RS = BuildMI(MBB, I, I.getDebugLoc(),
                              TII->get(AMDGPU::REG_SEQUENCE), CurVReg);
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

          if (RewriteOp) {
            Op.setReg(CurVReg);
            Op.setSubReg(SubRegIdx);
          }

          dbgs() << "Rewriting use: " << Op << " to " << CurVReg
                 << " with subreg: " << SubRegIdx << "\n";
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
        Register VReg = PHIMap[&PHI];
        if (VregNames[VReg].empty()) {
          PHI.addOperand(MachineOperand::CreateReg(VReg, false, false, false,
                                                   false, false));
        } else {
          CurVRegInfo VRInfo = VregNames[VReg].back();
          MachineInstr *DefMI = VregNames[VReg].back().DefMI;
          MachineOperand *DefOp = DefMI->findRegisterDefOperand(VRInfo.CurName, TRI);
          PHI.addOperand(MachineOperand::CreateReg(VRInfo.CurName, false, false,
                                                   false, false, false, false,
                                                   DefOp->getSubReg()));
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
    SetVector<unsigned> Killed;
    for (auto &I : MBB) {
      for (auto Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual() &&
            !Killed.contains(Op.getReg())) {
          CrossBlockVRegs.insert(Op.getReg());
        }
      }
      for (auto Op : I.defs()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          Killed.insert(Op.getReg());
          DefBlocks[Op.getReg()].insert(&MBB);
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
             for (auto VReg : CrossBlockVRegs) {
               dbgs() << Register::virtReg2Index(VReg) << " ";
             } dbgs()
             << "\n");

  for (auto VReg : CrossBlockVRegs) {
    SmallVector<MachineBasicBlock *> PHIBlocks;
    for (auto &MBB : MF) {
      LiveRange &LR = LIS->getInterval(VReg);
      if (LIS->isLiveInToMBB(LR, &MBB))
        LiveInBlocks[VReg].insert(&MBB);
    }

    LLVM_DEBUG(
        dbgs() << "findPHINodesPlacement input:\nVreg: "
               << Register::virtReg2Index(VReg) << "\n";
        dbgs() << "Def Blocks: \n"; for (auto MBB : DefBlocks[VReg]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs() << "\nLiveIn Blocks: \n";
        for (auto MBB : LiveInBlocks[VReg]) {
          dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
        } dbgs()
        << "\n");

    findPHINodesPlacement(LiveInBlocks[VReg], DefBlocks[VReg], PHIBlocks);
    LLVM_DEBUG(dbgs() << "\nBlocks to insert PHI nodes:\n";
               for (auto MBB : PHIBlocks) {
                 dbgs() << MBB->getName() << "." << MBB->getNumber() << " ";
               } dbgs()
               << "\n");
    for (auto MBB : PHIBlocks) {
      if (!PHINodes[MBB->getNumber()].contains(VReg)) {
        // Insert PHI for VReg. Don't use new VReg here as we'll replace them
        // in renaming phase.
        auto PHINode = BuildMI(*MBB, MBB->begin(), DebugLoc(), TII->get(TargetOpcode::PHI))
            .addReg(VReg, RegState::Define);
        PHINodes[MBB->getNumber()].insert(VReg);
        PHIMap[PHINode] = VReg;
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