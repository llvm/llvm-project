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
  DenseMap<unsigned, VRegDefStack> VregNames;
  DenseSet<unsigned> DefSeen;

  void collectCrossBlockVRegs(MachineFunction &MF);
  void findPHINodesPlacement(const SmallPtrSetImpl<MachineBasicBlock *> &LiveInBlocks,
                          const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                          SmallVectorImpl<MachineBasicBlock *> &PHIBlocks) {
    
    IDFCalculatorBase<MachineBasicBlock, false> IDF(MDT->getBase());

    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    IDF.calculate(PHIBlocks);
  }

  void renameVRegs(MachineBasicBlock &MBB) {
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

      // TODO: Need to support the RENAIMED set to avoid replacing the registers
      // which were not renamed in uses!
      for (auto &Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          unsigned VReg = Op.getReg();
          assert(!VregNames[VReg].empty() &&
                 "Error: use does not dominated by definition!\n");
          CurVRegInfo VRInfo = VregNames[VReg].back();
          unsigned CurVReg = VRInfo.CurName;
          // Does it meet the TODO above ?
          if (CurVReg == VReg)
            continue;
          unsigned DefSubregIdx = VRInfo.PrevSubRegIdx;
          LaneBitmask DefMask = VRInfo.PrevMask;
          MachineInstr *DefMI = VregNames[VReg].back().DefMI;
          MachineOperand *DefOp = DefMI->findRegisterDefOperand(CurVReg,
          TRI);

          // LaneBitmask DefMask = getOperandLaneMask(*DefOp);
          dbgs() << "Def mask : " << PrintLaneMask(DefMask) << "\n";
          LaneBitmask UseMask = getOperandLaneMask(Op, TRI, MRI);
          dbgs() << "Use mask : " << PrintLaneMask(UseMask) << "\n";
          LaneBitmask UndefSubRegs = UseMask & ~DefMask;
          dbgs() << "UndefSubRegs: " << PrintLaneMask(UndefSubRegs) << "\n";

          unsigned SubRegIdx = AMDGPU::NoRegister;
          
          if (UndefSubRegs.any()) {
            // The closest Def defines not all the subregs used here!
            SmallVector<std::tuple<unsigned, unsigned, unsigned>> RegSeqOps;

            RegSeqOps.push_back({CurVReg, DefOp->getSubReg(), DefSubregIdx});

            VRegDefStack VregDefs = VregNames[VReg];

            VRegDefStack::reverse_iterator It = ++VregDefs.rbegin();
            for (; It != VregDefs.rend(); ++It) {
              // auto CurDef = It->CurDefMI;
              auto R = It->CurName;
              // auto CurDefOp = CurDef->findRegisterDefOperand(R, TRI);
              LaneBitmask DefMask = It->PrevMask;
              dbgs() << "Lanes defined for VReg before renaming : "
                     << PrintLaneMask(DefMask) << "\n";
              LaneBitmask CurDefinedBits = DefMask & UndefSubRegs;
              dbgs() << "Defined bits are : " << PrintLaneMask(CurDefinedBits)
                     << "\n";
 
              if (unsigned SubRegIdx = getSubRegIndexForLaneMask(CurDefinedBits, TRI))
                RegSeqOps.push_back({R, SubRegIdx, SubRegIdx});
              // clear subregs for which definition is found
              UndefSubRegs &= ~CurDefinedBits;
              dbgs() << "UndefSubRegs: " << PrintLaneMask(UndefSubRegs) << "\n";
              if (UndefSubRegs.none())
                break;
            }
            // All subreg defs are found. Insert REG_SEQUENCE.
            auto *RC = TRI->getRegClassForOperandReg(*MRI, Op);
            CurVReg = MRI->createVirtualRegister(RC);
            auto RS = BuildMI(MBB, I, I.getDebugLoc(), TII->get(AMDGPU::REG_SEQUENCE),
                    CurVReg);
            for (auto O : RegSeqOps) {
              auto [R, SrcSubreg, DstSubreg] = O;
              RS.addReg(R, 0, SrcSubreg);
              RS.addImm(DstSubreg);
            }
          } else {
            if ((DefMask | UseMask) != UseMask) {
              SubRegIdx = getSubRegIndexForLaneMask(UseMask & DefMask, TRI);
            }
          }
          Op.setReg(CurVReg);
          Op.setSubReg(SubRegIdx);
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
      renameVRegs(*ChildMBB);
    }

    for (auto &I : MBB) {
      for (auto Op : I.defs()) {
        if (Op.getReg().isVirtual()) {
          Register VReg = Op.getReg();
          if (!VregNames[VReg].empty())
            VregNames[VReg].pop_back();
        }
      }
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
  VregNames.clear();
  DefSeen.clear();
 
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
  renameVRegs(MF.front());
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