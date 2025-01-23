

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Timer.h"

#include "AMDGPU.h"

#include "AMDGPUNextUseAnalysis.h"

#define DEBUG_TYPE "amdgpu-next-use"

using namespace llvm;

//namespace {


void NextUseResult::init(const MachineFunction &MF) {
  TG = new TimerGroup("Next Use Analysis",
                      "Compilation Timers for Next Use Analysis");
  T1 = new Timer("Next Use Analysis", "Time spent in analyse()", *TG);
  T2 = new Timer("Next Use Analysis", "Time spent in computeNextUseDistance()",
                 *TG);
  for (auto L : LI->getLoopsInPreorder()) {
    SmallVector<MachineBasicBlock *> Exiting;
    L->getExitingBlocks(Exiting);
    for (auto B : Exiting) {
      for (auto S : successors(B)) {
        if (!L->contains(S)) {
          EdgeWeigths[B->getNumber()] = S->getNumber();
        }
      }
    }
  }
}

void NextUseResult::analyze(const MachineFunction &MF) {
  // Upward-exposed distances are only necessary to convey the data flow from
  // the block to its predecessors. No need to store it beyond the analyze
  // function as the analysis users are only interested in the use distances
  // relatively to the given MI or the given block end.
  DenseMap<unsigned, VRegDistances> UpwardNextUses;
  T1->startTimer();
  bool Changed = true;
  while(Changed) {
    Changed = false;
    for (auto MBB : post_order(&MF)) {
      unsigned MBBNum = MBB->getNumber();
      VRegDistances Curr, Prev;
      if (UpwardNextUses.contains(MBBNum)) {
        Prev = UpwardNextUses[MBBNum];
      }

      LLVM_DEBUG(dbgs() << "\nMerging successors for " << MBB->getName()
                        << "\n";);

      for (auto Succ : successors(MBB)) {
        unsigned SuccNum = Succ->getNumber();

        LLVM_DEBUG(dbgs() << "Merging " << Succ->getName() << "\n");

        if (UpwardNextUses.contains(SuccNum)) {
          VRegDistances SuccDist = UpwardNextUses[SuccNum];
          // Check if the edge from MBB to Succ goes out of the Loop
          unsigned Weight = 0;
          if (EdgeWeigths.contains(MBB->getNumber())) {
            int SuccNum = EdgeWeigths[MBB->getNumber()];
            if (Succ->getNumber() == SuccNum)
              Weight = Infinity;
          }
          LLVM_DEBUG(
            dbgs() << "Curr: ";
            printVregDistances(Curr);
            dbgs() << "Succ: ";
            printVregDistances(SuccDist));
          Curr.merge(SuccDist, Weight);
          LLVM_DEBUG(dbgs() << "Curr after merge: ";
                     printVregDistances(Curr));
        }
      }

      NextUseMap[MBBNum].Bottom = Curr;

      for (auto &MI : make_range(MBB->rbegin(), MBB->rend())) {
        
        for (auto &P : Curr) {
          VRegDistances::SortedRecords Tmp;
          for (auto D : P.second)
            Tmp.insert({D.first, ++D.second});
          P.second = Tmp;
        }

        for (auto &MO : MI.operands()) {
          if (MO.isReg() && MO.getReg().isVirtual()) {
            VRegMaskPair P(MO, *TRI);
            if(MO.isUse()) {
              Curr.insert(P, 0);
              UsedInBlock[MBB->getNumber()].insert(P);
            } else if (MO.isDef()) {
              Curr.clear(P);
            }
          }
        }
        NextUseMap[MBBNum].InstrDist[&MI] = Curr;
        // printVregDistances(Curr);
      }

      UpwardNextUses[MBBNum] = std::move(Curr);

      bool Changed4MBB = (Prev != UpwardNextUses[MBBNum]);

      Changed |= Changed4MBB;
    }
  }
  T1->stopTimer();
  TG->print(llvm::errs());
}

void NextUseResult::getFromSortedRecords(
    const VRegDistances::SortedRecords Dists, LaneBitmask Mask, unsigned &D) {
  LLVM_DEBUG(dbgs() << "Mask : [" << PrintLaneMask(Mask) <<"]\n");
  for (auto P : Dists) {
    // Records are sorted in distance increasing order. So, the first record
    // is for the closest use.
    LaneBitmask UseMask = P.first;
    LLVM_DEBUG(dbgs() << "Used mask : [" << PrintLaneMask(UseMask) << "]\n");
    if ((UseMask & Mask) == UseMask) {
      D = P.second;
      break;
    }
  }
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock::iterator I,
                                   const VRegMaskPair VMP) {
  SmallVector<VRegMaskPair> Result;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].InstrDist.contains(&*I)) {
    VRegDistances Dists = NextUseMap[MBBNum].InstrDist[&*I];
    if (NextUseMap[MBBNum].InstrDist[&*I].contains(VMP.VReg)) {
      VRegDistances::SortedRecords Dists =
          NextUseMap[MBBNum].InstrDist[&*I][VMP.VReg];
      LLVM_DEBUG(dbgs() << "Mask : [" << PrintLaneMask(VMP.LaneMask) << "]\n");
      for (auto P : reverse(Dists)) {
        LaneBitmask UseMask = P.first;
        LLVM_DEBUG(dbgs() << "Used mask : [" << PrintLaneMask(UseMask)
                          << "]\n");
        if ((UseMask & VMP.LaneMask) == UseMask) {
          Result.push_back({VMP.VReg, UseMask});
        }
      }
    }
  }
  return std::move(Result);
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock::iterator I,
                                           const VRegMaskPair VMP) {
  unsigned Dist = Infinity;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].InstrDist.contains(&*I)) {
    VRegDistances Dists = NextUseMap[MBBNum].InstrDist[&*I];
    if (NextUseMap[MBBNum].InstrDist[&*I].contains(VMP.VReg)) {
      // printSortedRecords(Dists[VMP.VReg], VMP.VReg);
      getFromSortedRecords(Dists[VMP.VReg], VMP.LaneMask, Dist);
    }
  }

  return Dist;
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
                                           const VRegMaskPair VMP) {
  unsigned Dist = Infinity;
  unsigned MBBNum = MBB.getNumber();
  if (NextUseMap.contains(MBBNum)) {
    if (NextUseMap[MBBNum].Bottom.contains(VMP.VReg)) {
      getFromSortedRecords(NextUseMap[MBBNum].Bottom[VMP.VReg], VMP.LaneMask,
                           Dist);
    }
  }
  return Dist;
}

AMDGPUNextUseAnalysis::Result
AMDGPUNextUseAnalysis::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  return AMDGPUNextUseAnalysis::Result(MF,
                                       MFAM.getResult<SlotIndexesAnalysis>(MF),
                                       MFAM.getResult<MachineLoopAnalysis>(MF));
}

AnalysisKey AMDGPUNextUseAnalysis::Key;

//} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPUNextUseAnalysisPass",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](MachineFunctionAnalysisManager &MFAM) {
                  MFAM.registerPass([] { return AMDGPUNextUseAnalysis(); });
                });
          }};
}

char AMDGPUNextUseAnalysisWrapper::ID = 0;
char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisWrapper::ID;
INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                      "AMDGPU Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                    "AMDGPU Next Use Analysis", false, false)

bool AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(
    MachineFunction &MF) {
  NU.Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  NU.LI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  NU.MRI = &MF.getRegInfo();
  NU.TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  assert(NU.MRI->isSSA());
  NU.init(MF);
  NU.analyze(MF);
//  LLVM_DEBUG(NU.dump());
  return false;
}

void AMDGPUNextUseAnalysisWrapper::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AMDGPUNextUseAnalysisWrapper::AMDGPUNextUseAnalysisWrapper()
    : MachineFunctionPass(ID) {
  initializeAMDGPUNextUseAnalysisWrapperPass(*PassRegistry::getPassRegistry());
}