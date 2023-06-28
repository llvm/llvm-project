//===- LivenessAnalysis.cpp ------------------===//
//
// A liveness analysis for LLVM IR.
//
// This is based on the algorithm shown in Chapter 10 of the book:
//
//   Modern Compiler Implementation in Java (2nd edition)
//   by Andrew W. Appel

#include "llvm/Transforms/Yk/LivenessAnalysis.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

/// Wrapper to make `std::set_difference` more concise.
///
/// Store the difference between `S1` and `S2` into `Into`.
void vset_difference(const std::set<Value *> &S1, const std::set<Value *> &S2,
                     std::set<Value *> &Into) {
  std::set_difference(S1.begin(), S1.end(), S2.begin(), S2.end(),
                      std::inserter(Into, Into.begin()));
}

/// Wrapper to make `std::set_union` more concise.
///
/// Store the union of `S1` and `S2` into `Into`.
void vset_union(const std::set<Value *> &S1, const std::set<Value *> &S2,
                std::set<Value *> &Into) {
  std::set_union(S1.begin(), S1.end(), S2.begin(), S2.end(),
                 std::inserter(Into, Into.begin()));
}

namespace llvm {

std::set<Instruction *>
LivenessAnalysis::getSuccessorInstructions(Instruction *I) {
  Instruction *Term = I->getParent()->getTerminator();
  std::set<Instruction *> SuccInsts;
  if (I != Term) {
    // Non-terminating instruction: the sole successor instruction is the
    // next instruction in the block.
    SuccInsts.insert(I->getNextNode());
  } else {
    // Terminating instruction: successor instructions are the first
    // instructions of all successor blocks.
    for (unsigned SuccIdx = 0; SuccIdx < Term->getNumSuccessors(); SuccIdx++)
      SuccInsts.insert(&*Term->getSuccessor(SuccIdx)->begin());
  }
  return SuccInsts;
}

bool LivenessAnalysis::updateValueSet(std::set<Value *> &S,
                                      const std::set<Value *> &R) {
  const bool Changed = (S != R);
  S = R;
  return Changed;
}

LivenessAnalysis::LivenessAnalysis(Function *Func) {
  // Compute defs and uses for each instruction.
  std::map<Instruction *, std::set<Value *>> Defs;
  std::map<Instruction *, std::set<Value *>> Uses;

  // Create a domniator tree so we can later sort the live variables.
  DT = DominatorTree(*Func);

  for (BasicBlock &BB : *Func) {
    for (Instruction &I : BB) {
      // Record what this instruction defines.
      if (!I.getType()->isVoidTy())
        Defs[&I].insert(cast<Value>(&I));

      // Record what this instruction uses.
      //
      // In order to track the operands of PHI nodes we need to be a bit crafty
      // as otherwise we end up with live values in blocks where they actually
      // don't exist. To avoid this, we need to mark as live any PHI operand
      // from one block at the end of all other blocks. This leads to those
      // values being killed immediately after entering the block in the
      // backwards pass and thus the value never being live there.
      //
      // The book doesn't cover this quirk, as it explains liveness for
      // non-SSA form, and thus doesn't need to worry about Phi nodes.
      if (isa<PHINode>(I)) {
        PHINode *P = cast<PHINode>(&I);
        for (unsigned IVC = 0; IVC < P->getNumIncomingValues(); IVC++) {
          BasicBlock *IBB = P->getIncomingBlock(IVC);
          Value *IV = P->getIncomingValue(IVC);
          if (isa<Constant>(IV)) {
            continue;
          }
          // For each block that isn't the incoming block create a Def for this
          // value. This means when we do the backwards liveness pass this
          // value is immediately killed in the block.
          for (auto *BBB = P->block_begin(); BBB != P->block_end(); BBB++) {
            if (*BBB != IBB) {
              Instruction *Last = &((*BBB)->back());
              Defs[Last].insert(IV);
            }
          }
          Uses[&I].insert(IV);
        }
        continue;
      }

      for (auto *U = I.op_begin(); U < I.op_end(); U++)
        if ((!isa<Constant>(U)) && (!isa<BasicBlock>(U)) &&
            (!isa<MetadataAsValue>(U)) && (!isa<InlineAsm>(U)))
          Uses[&I].insert(*U);
    }
  }

  // Normally the live range of a variable starts at the instruction that
  // defines it. No instruction defines the function's arguments, but it is
  // important that we don't report them dead. We make function arguments live
  // at the start of the function by adding them into the `In` set of the first
  // instruction.
  Instruction *FirstInst = &*Func->getEntryBlock().begin();
  for (auto &Arg : Func->args())
    In[FirstInst].insert(&Arg);

  // Compute the live sets for each instruction.
  //
  // This is the fixed-point of the following data-flow equations (page 206
  // in the book referenced above):
  //
  //    in[I] = use[I] ∪ (out[I] - def[I])
  //
  //    out[I] =       ∪
  //             (S in succ[I])    in[S]
  //
  // Note that only the `In` map is kept after this constructor ends, so
  // only `In` is a field.
  std::map<Instruction *, std::set<Value *>> Out;
  bool Changed;
  do {
    Changed = false;
    // As the book explains, fixed-points are reached quicker if we process
    // control flow in "approximately reverse direction" and if we compute
    // `out[I]` before `in[I]`.
    //
    // Because the alrogithm works by propagating liveness from use sites
    // backwards to def sites (where liveness is killed), by working
    // backwards we are able to propagate long runs of liveness in one
    // iteration of the algorithm.
    for (BasicBlock *BB : post_order(&*Func)) {
      for (BasicBlock::reverse_iterator II = BB->rbegin(); II != BB->rend();
           II++) {
        Instruction *I = &*II;

        // Update out[I].
        const std::set<Instruction *> SuccInsts = getSuccessorInstructions(I);
        std::set<Value *> NewOut;
        for (Instruction *SI : SuccInsts) {
          NewOut.insert(In[SI].begin(), In[SI].end());
        }
        Changed |= updateValueSet(Out[I], NewOut);

        // Update in[I].
        std::set<Value *> OutMinusDef;
        vset_difference(Out[I], Defs[I], OutMinusDef);

        std::set<Value *> NewIn;
        vset_union(Uses[I], OutMinusDef, NewIn);
        Changed |= updateValueSet(In[I], NewIn);
      }
    }
  } while (Changed); // Until a fixed-point.
}

std::vector<Value *> LivenessAnalysis::getLiveVarsBefore(Instruction *I) {
  std::set<Value *> Res = In[I];
  // Sort the live variables by order of appearance using a dominator tree. The
  // order is important for frame construction during deoptimisation: since live
  // variables may reference other live variables they need to be proceesed in
  // the order they appear in the module.
  std::vector<Value *> Sorted(Res.begin(), Res.end());
  std::sort(Sorted.begin(), Sorted.end(), [this](Value *A, Value *B) {
    if (isa<Instruction>(B))
      return DT.dominates(A, cast<Instruction>(B));
    return false;
  });
  return Sorted;
}

} // namespace llvm
