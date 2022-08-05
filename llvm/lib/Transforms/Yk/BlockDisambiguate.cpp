//===- BlockDisambiguate.cpp - Unambiguous block mapping for yk ----===//
//
// This pass ensures that yk is able to unambiguously map machine blocks back
// to LLVM IR blocks. Specifically it does two separate, but related, things:
//
//  - Inserts blocks to disambiguate intra-function branching.
//  - Inserts blocks to disambiguate returning from direct recursion.
//
// Intra-function branch disambiguation
// ------------------------------------
//
// In the JIT runtime, the mapping stage converts the *machine* basic blocks of
// a trace back to high-level basic blocks (the ones in LLVM IR). A problem
// arises when the mapper encounters two consecutive machine basic block
// entries that map back to the same high-level block. If after mapping machine
// blocks, the high-level trace contains the sequence `[bbA, bbA]`, then it may
// not clear if:
//
//  - the program executed the block `bbA` twice, or
//  - `bbA` executed only once, but is composed of more than one machine block.
//
// This pass disambiguates the two cases by adding extra "disambiguation
// blocks" for high-level IR blocks that have an edge straight back to
// themselves. The new blocks make it clear when execution has left and
// re-entered the same high-level block.
//
// For a concrete example, suppose that we have a high-level LLVM IR basic
// block, `bbA`, as shown in Fig. 1a.
//
//                                       ┌──►│
//                                       │   ▼
//         ┌──►│                         │ ┌────┐   ┌────┐
//         │   ▼                         │ │bb.0│──►│bb.1│
//         │ ┌───┐                       │ └────┘   └────┘
//         │ │bbA│                       │   │        │
//         │ └───┘                       │   ▼        ▼
//         └───┤                         │ ┌────┐   ┌────┐
//             ▼                         │ │bb.3│◄──│bb.2│
//                                       │ └────┘   └────┘
//                                       └───┤
//                                           ▼
//
//         (Fig. 1a)                         (Fig. 1b)
//  High-level LLVM IR block.          Lowered machine blocks.
//
// Now suppose that during code-gen `bbA` lowers to the four machine blocks, as
// shown in Fig. 1b. This is entirely possible: any given LLVM instruction can
// lower to any number of machine blocks and there can be arbitrary control flow
// between them [0].
//
// Let's look at two ways that execution can enter the machine blocks of `bbA`
// and flow through them before exiting to the machine blocks elsewhere:
//
//  - `[bb.0, bb.3, bb.0, bb.3]`
//    i.e. `bbA` is executed twice.
//
//  - `[bb.0, bb.1, bb.2, bb.3]`
//    i.e. `bbA` is executed once, taking a longer path.
//
// Since `bb.0` through `bb.3` all belong to the high-level block `bbA`, a
// naive mapping back to high-level IR would give a trace of `[bbA, bbA, bbA,
// bbA]` for both of the above paths, and we have no way of knowing whether
// `bbA` executed once or twice.
//
// The pass implemented in this file resolves this ambiguity by ensuring that
// no high-level IR block can branch straight back to itself. With this
// property in-place the mapper can safely assume that repeated consecutive
// entries for the same high-level block, means that execution is within the
// confines of the same high-level block, and that the block is not being
// re-executed.
//
// For our worked example, this pass would change the high-level IR as shown in
// Fig. 2a.
//
// ```
//                                            ┌──►│
//                                            │   ▼
//                                            │ ┌────┐   ┌────┐
//         ┌──►│                              │ │bb.0│──►│bb.1│
//         │   ▼                              │ └────┘   └────┘
//         │ ┌───┐                            │   │        │
//         │ │bbA│                            │   ▼        ▼
//         │ └───┘                            │ ┌────┐   ┌────┐
//         │   │                              │ │bb.3│◄──│bb.2│
//         │   ▼                              │ └────┘   └────┘
//         │ ┌───┐                            │   │ │
//         └─│bbB│  disambiguation            │   ▼ └───┐
//           └───┘      block                 │ ┌────┐  │
//             │                              └─│bb.4│  │
//             ▼                                └────┘  │
//                                                      ▼
//
//        (Fig. 2a)                            (Fig. 2b)
//    High-level blocks after              Machine blocks after
//     disambiguation pass.                disambiguation pass.
// ```
//
// Now if `bbA` is re-executed control flow must go via the "disambiguation
// block" `bbB` and our example paths would now be:
//
//  - `[bb.0, bb.3, bb.4, bb.0, bb.3, bb.4]`
//  - `[bb.0, bb.1, bb.2, bb.3]`
//
// And their initial mappings are:
//
//  - `[bbA, bbA, bbB, bbA, bbA, bbB]`
//  - `[bbA, bbA, bbA, bbA]`
//
// And consecutive repeated entries can be collapsed giving:
//
//  - `[bbA, bbB, bbA, bbB]`
//  - `[bbA]`
//
// The former unambiguously expresses that `bbA` was executed twice. The latter
// unambiguously expresses that `bbA` was executed only once.
//
// Return from direct recursion disambiguation
// -------------------------------------------
//
// A similar case that requires disambiguation is where a function recursively
// calls itself (i.e. the function is "directly recursive") immediately before
// returning.
//
// When a series of recursive calls bubble up from the recursion
// we will get repeated entries for the high-level block containing the
// return statement. But since the return block may include other instructions
// which may themselves lower to multiple machine basic blocks, we need to do
// something in order to differentiate recursive returning from non-recursive
// returning when we see repeated entries for a block containing a return
// statement.
//
// In other words, given the return block for a function `myself()` shown in
// Fig 3b. and the high-level trace `[bbRet, bbRet, bbRet, bbRet]`, how many
// times have we returned from recursive calls? We can't say because
// `<instructions>` may generate multiple machine basic blocks that all map
// back to `bbRet` [1].
//
// Our solution is to insert a (high-level) padding block between the return
// statement and the instructions preceding it.
//
//           ┌────────────────┐             ┌────────────────┐
//           │bbRet:          │             │bbRet1:         │
//           │  <instructions>│             │  <instructions>│
//           │  call @myself()│             │  call @myself()│
//           │  ret           │             │  br %bbRet2    │
//           └────────────────┘             └────────────────┘
//                                                  │
//            (Fig 3a, above)                       ▼
//            Return block before             ┌────────────┐
//            transformation.                 │bbRet2:     │
//                                            │  br %bbRet3│
//                                            └────────────┘
//                                                  │
//         (Fig 3b, right)                          ▼
//         Return block after transformation    ┌───────┐
//         with padding block.                  │bbRet3:│
//                                              │  ret  │
//                                              └───────┘
//
// After transformation, repeated entries for a block can only occur in the
// mapped trace if `<instructions>` becomes multiple machine blocks during
// code-gen (e.g. `[bbRet1, bbRet1, bbRet1, bbRet1]`), whereas returning from
// direct recursion will cause sequences of `bbRet2, bbRet3` to appear in the
// mapped trace.
//
// As with intra-function branch disambiguation, the mapper is then free to
// collapse repeated entries for the same block when constructing the mapped
// trace.
//
// Discussion
// ----------
//
// The pass runs after high-level IR optimisations (and requires some backend
// optimisations disabled) to ensure that LLVM doesn't undo our work, by
// folding the machine block for `bbB` back into its predecessor in `bbA`.
//
// Alternative approaches that we dismissed, and why:
//
//  - For intra-function branches, consider branches back to the entry machine
//    block of a high-level block as a re-execution of the high-level block.
//    Even assuming that we can identify the entry machine block for a
//    high-level block, this is flawed. As can be seen in the example above,
//    both internal and non-internal control flow can branch back to the entry
//    block. Additionally, there may not be a unique entry machine basic block.
//
//  - Mark (in the machine IR) which branches are exits to the high-level IR
//    block and encode this is the basic block map somehow. This is more
//    complicated, but may work. We may revisit this approach later:
//    https://github.com/ykjit/yk/issues/435
//
//  - Try to make it so that high-level IR blocks lower to exactly one machine
//    block. It will be difficult to find all of the (platform specific) cases
//    where a high-level block can lower to many machine blocks, and it's
//    likely that some LLVM IR constructs require internal control flow for
//    correct semantics.
//
// Footnotes
// ---------
//
// [0]: For some targets, a single high-level LLVM IR instruction can even
//      lower to a machine-IR-level loop, for example `cmpxchng` on some ARM
//      targets, and integer division on targets which have no dedicated
//      division instructions (e.g. AVR). A high-level instruction lowered to
//      a machine-level loop presents a worst-case scenario for ambiguity, as
//      a potentially unbounded number of machine blocks can be executed
//      within the confines of a single high-level basic block.
//
// [1]: Futher those machine blocks may branch to the same address that a
//      return from direct recursion would land at, adding another layer of
//      ambiguity.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Yk/BlockDisambiguate.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>

using namespace llvm;

#define DEBUG_TYPE "yk-block-disambiguate"

namespace llvm {
void initializeYkBlockDisambiguatePass(PassRegistry &);
}

namespace {
class YkBlockDisambiguate : public ModulePass {
public:
  static char ID;
  YkBlockDisambiguate() : ModulePass(ID) {
    initializeYkBlockDisambiguatePass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();
    for (Function &F : M)
      processFunction(Context, F);
    return true;
  }

private:
  // Create a block for intra-function branch disambiguation.
  BasicBlock *makeBranchDisambiguationBB(LLVMContext &Context, BasicBlock *BB,
                                         std::vector<BasicBlock *> &NewBBs) {
    BasicBlock *DBB = BasicBlock::Create(Context, "");
    NewBBs.push_back(DBB);
    IRBuilder<> Builder(DBB);
    Builder.CreateBr(BB);
    return DBB;
  }

  void processFunction(LLVMContext &Context, Function &F) {
    std::vector<BasicBlock *> NewBBs;
    for (BasicBlock &BB : F) {
      Instruction *TI = BB.getTerminator();

      // YKFIXME: not implemented.
      // https://github.com/ykjit/yk/issues/440
      assert(!isa<IndirectBrInst>(TI));

      if (isa<BranchInst>(TI)) {
        BranchInst *BI = cast<BranchInst>(TI);
        for (unsigned SuccIdx = 0; SuccIdx < BI->getNumSuccessors();
             SuccIdx++) {
          BasicBlock *SuccBB = BI->getSuccessor(SuccIdx);
          if (SuccBB == &BB) {
            BasicBlock *DBB = makeBranchDisambiguationBB(Context, &BB, NewBBs);
            BI->setSuccessor(SuccIdx, DBB);
            BB.replacePhiUsesWith(&BB, DBB);
          }
        }
      } else if (isa<SwitchInst>(TI)) {
        SwitchInst *SI = cast<SwitchInst>(TI);
        for (unsigned SuccIdx = 0; SuccIdx < SI->getNumSuccessors();
             SuccIdx++) {
          BasicBlock *SuccBB = SI->getSuccessor(SuccIdx);
          if (SuccBB == &BB) {
            BasicBlock *DBB = makeBranchDisambiguationBB(Context, &BB, NewBBs);
            SI->setSuccessor(SuccIdx, DBB);
            BB.replacePhiUsesWith(&BB, DBB);
          }
        }
      } else if (isa<ReturnInst>(TI)) {
        // Apply return from  direct recursion disambiguation.
        //
        // YKFIXME: We do this even if the function is not directly recursive.
        // If we can prove that it is not, then we can skip this step.

        // Make the New Return Block (NRBB) and the Padding Block (PBB).
        BasicBlock *NRBB = BasicBlock::Create(Context, "");
        BasicBlock *PBB = BasicBlock::Create(Context, "");

        // Make the original return block branch to the padding block.
        IRBuilder<> Builder(Context);
        Builder.SetInsertPoint(TI);
        Builder.CreateBr(PBB);

        // Make the padding block branch to the new return block.
        Builder.SetInsertPoint(PBB);
        Builder.CreateBr(NRBB);

        // Move the original return instruction into the new return block.
        Builder.SetInsertPoint(NRBB);
        TI->removeFromParent();
        Builder.Insert(TI);

        NewBBs.push_back(NRBB);
        NewBBs.push_back(PBB);
      }
    }

    // Insert new blocks at the end, so as to not iterate and mutate the
    // function's basic block list simultaneously.
    for (BasicBlock *BB : NewBBs)
      BB->insertInto(&F);
  }
};
} // namespace

char YkBlockDisambiguate::ID = 0;
INITIALIZE_PASS(YkBlockDisambiguate, DEBUG_TYPE, "yk block disambiguation",
                false, false)
ModulePass *llvm::createYkBlockDisambiguatePass() {
  return new YkBlockDisambiguate();
}
