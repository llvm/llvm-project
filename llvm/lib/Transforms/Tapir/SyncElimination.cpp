//===- SyncElimination.cpp - Eliminate unnecessary sync calls ----------------===//

#include "llvm/Transforms/Tapir.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/SmallSet.h"

#include <deque>
#include <map>

using namespace llvm;

namespace {

typedef SmallSet<const BasicBlock *, 32> BasicBlockSet;
typedef std::deque<const BasicBlock *> BasicBlockDeque;

struct SyncElimination : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  SyncElimination() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "SyncElimination: Found function: " << F.getName() << "\n";

    bool ChangedAny = false;

    while (true) {
      bool Changed = false;

      for (BasicBlock &block: F) {
        if (isa<SyncInst>(block.getTerminator())) {
          if (processSyncInstBlock(block)) {
            Changed = true;
            ChangedAny = true;
            break;
          }
        }
      }

      if (!Changed) {
        break;
      }
    }

    return ChangedAny;
  }

private:

  // We will explain what Rosetta and Vegas are later. Or rename them.
  // We promise.

  // Rosetta-finding code

  void findRosetta(const BasicBlock &BB, BasicBlockSet &OutputSet) {
    assert(isa<SyncInst>(BB.getTerminator()));

    BasicBlockSet Visited;
    BasicBlockDeque Frontier;
    std::map<const BasicBlock *, int> DetachLevel;

    DetachLevel[&BB] = 0;
    Frontier.push_back(&BB);
    OutputSet.insert(&BB);

    while (!Frontier.empty()) {
      const BasicBlock *Current = Frontier.front();
      Frontier.pop_front();

      for (const BasicBlock *Pred: predecessors(Current)) {
        // TODO@jiahao: Investigate potential issues with continue edges here.

        if (Visited.count(Pred) > 0) {
          continue;
        }

        if (isa<SyncInst>(Pred->getTerminator())) {
          continue;
        }

        Visited.insert(Pred);

        DetachLevel[Pred] = DetachLevel[Current];

        if (isa<ReattachInst>(Pred->getTerminator())) {
          DetachLevel[Pred] ++;
        } else if (isa<DetachInst>(Pred->getTerminator())) {
          DetachLevel[Pred] --;
        }

        if (DetachLevel[Pred] > 0) {
          OutputSet.insert(Pred);
        }

        if (DetachLevel[Pred] >= 0) {
          Frontier.push_back(Pred);
        }
      }
    }
  }

  // Vegas-finding code
  //
  // We run BFS starting from the sync block, following all foward edges, and stop a branch whenever
  // we hit another sync block.

  void findVegas(const BasicBlock &BB, BasicBlockSet &OutputSet) {
    assert(isa<SyncInst>(BB.getTerminator()));

    BasicBlockSet Visited;
    BasicBlockDeque Frontier;

    Frontier.push_back(&BB);

    while (!Frontier.empty()) {
      const BasicBlock *Current = Frontier.front();
      Frontier.pop_front();

      for (const BasicBlock *Succ: successors(Current)) {
        if (Visited.count(Succ) > 0) {
          continue;
        }

        Visited.insert(Succ);
        OutputSet.insert(Succ);

        // We need to include blocks whose terminator is another sync.
        // Therefore we still insert the block into OutputSet in this case.
        // However we do not search any further past the sync block.
        if (!isa<SyncInst>(Succ->getTerminator())) {
          Frontier.push_back(Succ);
        }
      }
    }
  }

  bool willMod(const ModRefInfo &Info) {
    return (Info == MRI_Mod || Info == MRI_ModRef);
  }

  bool instTouchesMemory(const Instruction &Inst) {
    return Inst.getOpcode() == Instruction::Load ||
           Inst.getOpcode() == Instruction::Store ||
           Inst.getOpcode() == Instruction::VAArg ||
           Inst.getOpcode() == Instruction::AtomicCmpXchg ||
           Inst.getOpcode() == Instruction::AtomicRMW;
  }

  // FIXME: we can do better
  void checkBlowUp(const Instruction &Inst) {
    if (isa<FenceInst>(Inst)) {
      errs() << Inst << "\n";
      llvm_unreachable("BOOOOOOOOOOOOOOOOOOOOOOOOM! not supported (yet)");
    }
  }

  bool isSyncEliminationLegal(const BasicBlockSet &RosettaSet, const BasicBlockSet &VegasSet) {
    AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

    for (const BasicBlock *RBB : RosettaSet) {
      for (const Instruction &RI : *RBB) {
        checkBlowUp(RI);

        if (RI.getOpcode() == Instruction::Sync) {
          continue;
        }

        for (const BasicBlock *VBB : VegasSet) {
          for (const Instruction &VI : *VBB) {
            checkBlowUp(VI);

            if (VI.getOpcode() == Instruction::Sync) {
              continue;
            }

            ImmutableCallSite RC(&RI), VC(&VI);

            if (!!RC) {
              // If RI is a call/invoke
              if (instTouchesMemory(VI) &&
                  AA->getModRefInfo(const_cast<Instruction *>(&VI), RC) != MRI_NoModRef) {
                errs() << "SyncElimination:     Conflict found between " << RI << " and " << VI << "\n";
                return false;
              }
            } else if (!!VC) {
              // If VI is a call/invoke
              if (instTouchesMemory(RI) &&
                  AA->getModRefInfo(const_cast<Instruction *>(&RI), VC) != MRI_NoModRef) {
                errs() << "SyncElimination:     Conflict found between " << RI << " and " << VI << "\n";
                return false;
              }
            } else {
              if (!instTouchesMemory(VI) || !instTouchesMemory(RI)) {
                continue;
              }

              // If neither instruction is a call/invoke
              MemoryLocation VML = MemoryLocation::get(&VI);
              MemoryLocation RML = MemoryLocation::get(&RI);

              if (AA->alias(RML, VML) && (willMod(AA->getModRefInfo(&RI, RML)) || willMod(AA->getModRefInfo(&VI, VML)))) {
                // If the two memory location can potentially be aliasing each other, and
                // at least one instruction modifies its memory location.
                errs() << "SyncElimination:     Conflict found between " << RI << " and " << VI << "\n";
                return false;
              }
            }
          }
        }
      }
    }

    return true;
  }

  bool processSyncInstBlock(BasicBlock &BB) {
    errs() << "SyncElimination: Found sync block: " << BB.getName() << "\n";

    BasicBlockSet RosettaSet, VegasSet;

    findRosetta(BB, RosettaSet);
    findVegas(BB, VegasSet);

    errs() << "SyncElimination:     Blocks found in the Rosetta set: " << "\n";
    for (const BasicBlock *BB: RosettaSet) {
      errs() << "SyncElimination:         " + BB->getName() << "\n";
    }

    errs() << "SyncElimination:     Blocks found in the Vegas set: " << "\n";
    for (const BasicBlock *BB: VegasSet) {
      errs() << "SyncElimination:         " + BB->getName() << "\n";
    }

    if (isSyncEliminationLegal(RosettaSet, VegasSet)) {
      SyncInst *Sync = dyn_cast<SyncInst>(BB.getTerminator());
      assert(Sync != NULL);
      BasicBlock* suc = Sync->getSuccessor(0);
      IRBuilder<> Builder(Sync);
      Builder.CreateBr(suc);
      Sync->eraseFromParent();
      errs() << "SyncElimination:     A sync is removed. " << "\n";
      return true;
    }

    return false;
  }
};

}

char SyncElimination::ID = 0;
static RegisterPass<SyncElimination> X("sync-elimination", "Do sync-elimination's pass", false, false);

// Public interface to the SyncElimination pass
FunctionPass *llvm::createSyncEliminationPass() {
  return new SyncElimination();
}
