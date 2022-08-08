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
/// Find the successor instructions of the specified instruction.
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

/// Replaces the value set behind the pointer `S` with the value set `R` and
/// returns whether the set behind `S` changed.
bool LivenessAnalysis::updateValueSet(std::set<Value *> *S,
                                      const std::set<Value *> R) {
  const bool Changed = (*S != R);
  *S = R;
  return Changed;
}

LivenessAnalysis::LivenessAnalysis(Function *Func) {
  // Compute defs and uses for each instruction.
  std::map<Instruction *, std::set<Value *>> Defs;
  std::map<Instruction *, std::set<Value *>> Uses;
  for (BasicBlock &BB : *Func) {
    for (Instruction &I : BB) {
      // Record what this instruction defines.
      if (!I.getType()->isVoidTy())
        Defs[&I].insert(cast<Value>(&I));

      // Record what this instruction uses.
      //
      // Note that Phi nodes are special and must be skipped. If we consider
      // their operands as uses, then Phi nodes in loops may use variables
      // before they are defined, and this messes with the algorithm.
      //
      // The book doesn't cover this quirk, as it explains liveness for
      // non-SSA form, and thus doesn't need to worry about Phi nodes.
      if (isa<PHINode>(I))
        continue;

      for (auto *U = I.op_begin(); U < I.op_end(); U++)
        if ((!isa<Constant>(U)) && (!isa<BasicBlock>(U)) &&
            (!isa<MetadataAsValue>(U)) && (!isa<InlineAsm>(U)))
          Uses[&I].insert(*U);
    }
  }

  // A function implicitly defines its arguments.
  //
  // To propagate the arguments properly we pretend that the first instruction
  // in the entry block defines the arguments.
  Instruction *FirstInst = &*Func->getEntryBlock().begin();
  for (auto &Arg : Func->args())
    Defs[FirstInst].insert(&Arg);

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
        Changed |= updateValueSet(&Out[I], std::move(NewOut));

        // Update in[I].
        std::set<Value *> OutMinusDef;
        vset_difference(Out[I], Defs[I], OutMinusDef);

        std::set<Value *> NewIn;
        vset_union(Uses[I], OutMinusDef, NewIn);
        Changed |= updateValueSet(&In[I], std::move(NewIn));
      }
    }
  } while (Changed); // Until a fixed-point.
}

/// Returns the set of live variables immediately before the specified
/// instruction.
std::set<Value *> LivenessAnalysis::getLiveVarsBefore(Instruction *I) {
  return In[I];
}

} // namespace llvm
