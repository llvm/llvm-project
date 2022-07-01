/*===- Scev.cpp -Creates and Simplifies Recurrences for ‘Expressions involving
Induction Variables’ Algorithm:
1. Get ScalarEvolution object.
2. Use getSCEV for the pointer operands
3. Take the scev pointer base
4. Subtract scev with scev pointer base to get the SCEVAddRecExpr(DiffVal).
eg:{8,+,16}<nuw><nsw><%for.cond>
5. This SCEVAddRecExpr will contain the required indices and Extract it. eg : 8
6. Store the index and corresponding Store instruction in StoreInsts map.
7. Sorting the Offset vector values.
8. Get the BB of store instruction and using that get the terminator
instruction.
9. Move all store instructions one by one before terminator instruction.
===-------------------------------------------------------------------------------------------===*/

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "hello"

namespace {
// Scev - The second implementation with getAnalysisUsage implemented.
struct Scev : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  Scev() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {

    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    SCEV *ScevVal, *BasePtr, *DiffVal, *GetEle, *TempPtr = nullptr;
    SmallVector<int> OffSet;
    llvm::DenseMap<int, Instruction *> StoreInsts;

    int Value = 0;

    // Store the index and corresponding Store instruction in StoreInsts map
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          if (auto *Gep =
                  dyn_cast<GetElementPtrInst>(Store->getPointerOperand())) {
            ScevVal = const_cast<SCEV *>(SE.getSCEV(Gep));
            if ((BasePtr = const_cast<SCEV *>(SE.getPointerBase(ScevVal)))) {
              if (TempPtr == nullptr)
                TempPtr = BasePtr;
              else if (TempPtr != BasePtr) {
                LLVM_DEBUG(dbgs()
                           << "\nBasePointers are not same, stopping the pass");
                continue;
              }
              DiffVal = const_cast<SCEV *>(SE.getMinusSCEV(ScevVal, BasePtr));
              // Get the index of scev
              if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(DiffVal)) {
                if ((GetEle = const_cast<SCEV *>(AddRec->getStart()))) {
                  if (SCEVConstant *BConst = dyn_cast<SCEVConstant>(GetEle)) {
                    ConstantInt *CI = BConst->getValue();
                    Value = CI->getSExtValue();
                  }
                  OffSet.push_back(Value);
                  StoreInsts[Value] = &I;
                }
              }
            }
          }
        }
      }
    }

    // Sorting the Offset vector values
    std::sort(OffSet.begin(), OffSet.end());

    // Get the BB of store instruction and using that get the terminator
    // instruction
    BasicBlock *StoreInstBB = StoreInsts[OffSet[0]]->getParent();
    Instruction *LastInst = StoreInstBB->getTerminator();

    // Move all store instructions one by one before terminator instruction
    if (OffSet.size() != 0) {
      for (auto V = OffSet.begin(), E = OffSet.end(); V != E; V = V + 1) {
        StoreInsts[*V]->moveBefore(LastInst);
      }
    }
    return false;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};
} // namespace

char Scev::ID = 0;
static RegisterPass<Scev>
    X("scev", "Scev Implementation Pass (with getAnalysisUsage implemented)");