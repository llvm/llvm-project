//===- MergeICmps.cpp - Optimize chains of integer comparisons ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass turns chains of integer comparisons into memcmp (the memcmp is
// later typically inlined as a chain of efficient hardware comparisons). This
// typically benefits c++ member or nonmember operator==().
//
// The basic idea is to replace a larger chain of integer comparisons loaded
// from contiguous memory locations into a smaller chain of such integer
// comparisons. Benefits are double:
//  - There are less jumps, and therefore less opportunities for mispredictions
//    and I-cache misses.
//  - Code size is smaller, both because jumps are removed and because the
//    encoding of a 2*n byte compare is smaller than that of two n-byte
//    compares.

//===----------------------------------------------------------------------===//

#include "llvm/ADT/APSInt.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

namespace {

#define DEBUG_TYPE "mergeicmps"

#define MERGEICMPS_DOT_ON

// A BCE atom.
struct BCEAtom {
  const Value *Base() const { return GEP ? GEP->getPointerOperand() : nullptr; }

  bool operator<(const BCEAtom &O) const {
    return Base() == O.Base() ? Offset.slt(O.Offset) : Base() < O.Base();
  }

  GetElementPtrInst *GEP = nullptr;
  LoadInst *LoadI = nullptr;
  APInt Offset;
};

// If this value is a load from a constant offset w.r.t. a base address, and
// there are no othe rusers of the load or address, returns the base address and
// the offset.
BCEAtom visitICmpLoadOperand(Value *const Val) {
  BCEAtom Result;
  if (auto *const LoadI = dyn_cast<LoadInst>(Val)) {
    DEBUG(dbgs() << "load\n");
    if (LoadI->isUsedOutsideOfBlock(LoadI->getParent())) {
      DEBUG(dbgs() << "used outside of block\n");
      return {};
    }
    if (LoadI->isVolatile()) {
      DEBUG(dbgs() << "volatile\n");
      return {};
    }
    Value *const Addr = LoadI->getOperand(0);
    if (auto *const GEP = dyn_cast<GetElementPtrInst>(Addr)) {
      DEBUG(dbgs() << "GEP\n");
      if (LoadI->isUsedOutsideOfBlock(LoadI->getParent())) {
        DEBUG(dbgs() << "used outside of block\n");
        return {};
      }
      const auto &DL = GEP->getModule()->getDataLayout();
      if (!isDereferenceablePointer(GEP, DL)) {
        DEBUG(dbgs() << "not dereferenceable\n");
        // We need to make sure that we can do comparison in any order, so we
        // require memory to be unconditionnally dereferencable.
        return {};
      }
      Result.Offset = APInt(DL.getPointerTypeSizeInBits(GEP->getType()), 0);
      if (GEP->accumulateConstantOffset(DL, Result.Offset)) {
        Result.GEP = GEP;
        Result.LoadI = LoadI;
      }
    }
  }
  return Result;
}

// A basic block with a comparison between two BCE atoms.
// Note: the terminology is misleading: the comparison is symmetric, so there
// is no real {l/r}hs. To break the symmetry, we use the smallest atom as Lhs.
class BCECmpBlock {
 public:
  BCECmpBlock() {}

  BCECmpBlock(BCEAtom L, BCEAtom R, int SizeBits)
      : Lhs_(L), Rhs_(R), SizeBits_(SizeBits) {
    if (Rhs_ < Lhs_)
      std::swap(Rhs_, Lhs_);
  }

  bool IsValid() const {
    return Lhs_.Base() != nullptr && Rhs_.Base() != nullptr;
  }

  // Assert the the block is consistent: If valid, it should also have
  // non-null members besides Lhs_ and Rhs_.
  void AssertConsistent() const {
    if (IsValid()) {
      assert(BB);
      assert(CmpI);
      assert(BranchI);
    }
  }

  const BCEAtom &Lhs() const { return Lhs_; }
  const BCEAtom &Rhs() const { return Rhs_; }
  int SizeBits() const { return SizeBits_; }

  // Returns true if the block does other works besides comparison.
  bool doesOtherWork() const;

  // The basic block where this comparison happens.
  BasicBlock *BB = nullptr;
  // The ICMP for this comparison.
  ICmpInst *CmpI = nullptr;
  // The terminating branch.
  BranchInst *BranchI = nullptr;

 private:
  BCEAtom Lhs_;
  BCEAtom Rhs_;
  int SizeBits_ = 0;
};

bool BCECmpBlock::doesOtherWork() const {
  AssertConsistent();
  // TODO(courbet): Can we allow some other things ? This is very conservative.
  // We might be able to get away with anything does does not have any side
  // effects outside of the basic block.
  // Note: The GEPs and/or loads are not necessarily in the same block.
  for (const Instruction &Inst : *BB) {
    if (const auto *const GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      if (!(Lhs_.GEP == GEP || Rhs_.GEP == GEP))
        return true;
    } else if (const auto *const L = dyn_cast<LoadInst>(&Inst)) {
      if (!(Lhs_.LoadI == L || Rhs_.LoadI == L))
        return true;
    } else if (const auto *const C = dyn_cast<ICmpInst>(&Inst)) {
      if (C != CmpI)
        return true;
    } else if (const auto *const Br = dyn_cast<BranchInst>(&Inst)) {
      if (Br != BranchI)
        return true;
    } else {
      return true;
    }
  }
  return false;
}

// Visit the given comparison. If this is a comparison between two valid
// BCE atoms, returns the comparison.
BCECmpBlock visitICmp(const ICmpInst *const CmpI,
                      const ICmpInst::Predicate ExpectedPredicate) {
  if (CmpI->getPredicate() == ExpectedPredicate) {
    DEBUG(dbgs() << "cmp "
                 << (ExpectedPredicate == ICmpInst::ICMP_EQ ? "eq" : "ne")
                 << "\n");
    auto Lhs = visitICmpLoadOperand(CmpI->getOperand(0));
    if (!Lhs.Base())
      return {};
    auto Rhs = visitICmpLoadOperand(CmpI->getOperand(1));
    if (!Rhs.Base())
      return {};
    return BCECmpBlock(std::move(Lhs), std::move(Rhs),
                       CmpI->getOperand(0)->getType()->getScalarSizeInBits());
  }
  return {};
}

// Visit the given comparison block. If this is a comparison between two valid
// BCE atoms, returns the comparison.
BCECmpBlock visitCmpBlock(Value *const Val, BasicBlock *const Block,
                          const BasicBlock *const PhiBlock) {
  if (Block->empty())
    return {};
  auto *const BranchI = dyn_cast<BranchInst>(Block->getTerminator());
  if (!BranchI)
    return {};
  DEBUG(dbgs() << "branch\n");
  if (BranchI->isUnconditional()) {
    // In this case, we expect an incoming value which is the result of the
    // comparison. This is the last link in the chain of comparisons (note
    // that this does not mean that this is the last incoming value, blocks
    // can be reordered).
    auto *const CmpI = dyn_cast<ICmpInst>(Val);
    if (!CmpI)
      return {};
    DEBUG(dbgs() << "icmp\n");
    auto Result = visitICmp(CmpI, ICmpInst::ICMP_EQ);
    Result.CmpI = CmpI;
    Result.BranchI = BranchI;
    return Result;
  } else {
    // In this case, we expect a constant incoming value (the comparison is
    // chained).
    const auto *const Const = dyn_cast<ConstantInt>(Val);
    DEBUG(dbgs() << "const\n");
    if (!Const->isZero())
      return {};
    DEBUG(dbgs() << "false\n");
    auto *const CmpI = dyn_cast<ICmpInst>(BranchI->getCondition());
    if (!CmpI)
      return {};
    DEBUG(dbgs() << "icmp\n");
    assert(BranchI->getNumSuccessors() == 2 && "expecting a cond branch");
    BasicBlock *const FalseBlock = BranchI->getSuccessor(1);
    auto Result = visitICmp(
        CmpI, FalseBlock == PhiBlock ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE);
    Result.CmpI = CmpI;
    Result.BranchI = BranchI;
    return Result;
  }
  return {};
}

// A chain of comparisons.
class BCECmpChain {
 public:
  BCECmpChain(const std::vector<BasicBlock *> &Blocks, PHINode &Phi);

  int size() const { return Comparisons_.size(); }

#ifdef MERGEICMPS_DOT_ON
  void dump() const;
#endif  // MERGEICMPS_DOT_ON

  bool simplify(const TargetLibraryInfo *const TLI);

 private:
  static bool IsContiguous(const BCECmpBlock &First,
                           const BCECmpBlock &Second) {
    return First.Lhs().Base() == Second.Lhs().Base() &&
           First.Rhs().Base() == Second.Rhs().Base() &&
           First.Lhs().Offset + First.SizeBits() / 8 == Second.Lhs().Offset &&
           First.Rhs().Offset + First.SizeBits() / 8 == Second.Rhs().Offset;
  }

  // Merges the given comparison blocks into one memcmp block and update
  // branches. Comparisons are assumed to be continguous. If NextBBInChain is
  // null, the merged block will link to the phi block.
  static void mergeComparisons(ArrayRef<BCECmpBlock> Comparisons,
                               BasicBlock *const NextBBInChain, PHINode &Phi,
                               const TargetLibraryInfo *const TLI);

  PHINode &Phi_;
  std::vector<BCECmpBlock> Comparisons_;
  // The original entry block (before sorting);
  BasicBlock *EntryBlock_;
};

BCECmpChain::BCECmpChain(const std::vector<BasicBlock *> &Blocks, PHINode &Phi)
    : Phi_(Phi) {
  // Now look inside blocks to check for BCE comparisons.
  std::vector<BCECmpBlock> Comparisons;
  for (BasicBlock *Block : Blocks) {
    BCECmpBlock Comparison = visitCmpBlock(Phi.getIncomingValueForBlock(Block),
                                           Block, Phi.getParent());
    Comparison.BB = Block;
    if (!Comparison.IsValid()) {
      DEBUG(dbgs() << "skip: not a valid BCECmpBlock\n");
      return;
    }
    if (Comparison.doesOtherWork()) {
      DEBUG(dbgs() << "block does extra work besides compare\n");
      if (Comparisons.empty()) {  // First block.
        // TODO(courbet): The first block can do other things, and we should
        // split them apart in a separate block before the comparison chain.
        // Right now we just discard it and make the chain shorter.
        DEBUG(dbgs()
              << "ignoring first block that does extra work besides compare\n");
        continue;
      }
      // TODO(courbet): Right now we abort the whole chain. We could be
      // merging only the blocks that don't do other work and resume the
      // chain from there. For example:
      //  if (a[0] == b[0]) {  // bb1
      //    if (a[1] == b[1]) {  // bb2
      //      some_value = 3; //bb3
      //      if (a[2] == b[2]) { //bb3
      //        do a ton of stuff  //bb4
      //      }
      //    }
      //  }
      //
      // This is:
      //
      // bb1 --eq--> bb2 --eq--> bb3* -eq--> bb4 --+
      //  \            \           \               \
      //   ne           ne          ne              \
      //    \            \           \               v
      //     +------------+-----------+----------> bb_phi
      //
      // We can only merge the first two comparisons, because bb3* does
      // "other work" (setting some_value to 3).
      // We could still merge bb1 and bb2 though.
      return;
    }
    DEBUG(dbgs() << "*Found cmp of " << Comparison.SizeBits()
                 << " bits between " << Comparison.Lhs().Base() << " + "
                 << Comparison.Lhs().Offset << " and "
                 << Comparison.Rhs().Base() << " + " << Comparison.Rhs().Offset
                 << "\n");
    DEBUG(dbgs() << "\n");
    Comparisons.push_back(Comparison);
  }
  EntryBlock_ = Comparisons[0].BB;
  Comparisons_ = std::move(Comparisons);
#ifdef MERGEICMPS_DOT_ON
  errs() << "BEFORE REORDERING:\n\n";
  dump();
#endif  // MERGEICMPS_DOT_ON
  // Reorder blocks by LHS. We can do that without changing the
  // semantics because we are only accessing dereferencable memory.
  std::sort(Comparisons_.begin(), Comparisons_.end(),
            [](const BCECmpBlock &a, const BCECmpBlock &b) {
              return a.Lhs() < b.Lhs();
            });
#ifdef MERGEICMPS_DOT_ON
  errs() << "AFTER REORDERING:\n\n";
  dump();
#endif  // MERGEICMPS_DOT_ON
}

#ifdef MERGEICMPS_DOT_ON
void BCECmpChain::dump() const {
  errs() << "digraph dag {\n";
  errs() << " graph [bgcolor=transparent];\n";
  errs() << " node [color=black,style=filled,fillcolor=lightyellow];\n";
  errs() << " edge [color=black];\n";
  for (size_t I = 0; I < Comparisons_.size(); ++I) {
    const auto &Comparison = Comparisons_[I];
    errs() << " \"" << I << "\" [label=\"%"
           << Comparison.Lhs().Base()->getName() << " + "
           << Comparison.Lhs().Offset << " == %"
           << Comparison.Rhs().Base()->getName() << " + "
           << Comparison.Rhs().Offset << " (" << (Comparison.SizeBits() / 8)
           << " bytes)\"];\n";
    const Value *const Val = Phi_.getIncomingValueForBlock(Comparison.BB);
    if (I > 0)
      errs() << " \"" << (I - 1) << "\" -> \"" << I << "\";\n";
    errs() << " \"" << I << "\" -> \"Phi\" [label=\"" << *Val << "\"];\n";
  }
  errs() << " \"Phi\" [label=\"Phi\"];\n";
  errs() << "}\n\n";
}
#endif  // MERGEICMPS_DOT_ON

bool BCECmpChain::simplify(const TargetLibraryInfo *const TLI) {
  // First pass to check if there is at least one merge. If not, we don't do
  // anything and we keep analysis passes intact.
  {
    bool AtLeastOneMerged = false;
    for (size_t I = 1; I < Comparisons_.size(); ++I) {
      if (IsContiguous(Comparisons_[I - 1], Comparisons_[I])) {
        AtLeastOneMerged = true;
        break;
      }
    }
    if (!AtLeastOneMerged)
      return false;
  }

  // Remove phi references to comparison blocks, they will be rebuilt as we
  // merge the blocks.
  for (const auto &Comparison : Comparisons_) {
    Phi_.removeIncomingValue(Comparison.BB, false);
  }

  // Point the predecessors of the chain to the first comparison block (which is
  // the new entry point).
  if (EntryBlock_ != Comparisons_[0].BB)
    EntryBlock_->replaceAllUsesWith(Comparisons_[0].BB);

  // Effectively merge blocks.
  int NumMerged = 1;
  for (size_t I = 1; I < Comparisons_.size(); ++I) {
    if (IsContiguous(Comparisons_[I - 1], Comparisons_[I])) {
      ++NumMerged;
    } else {
      // Merge all previous comparisons and start a new merge block.
      mergeComparisons(
          makeArrayRef(Comparisons_).slice(I - NumMerged, NumMerged),
          Comparisons_[I].BB, Phi_, TLI);
      NumMerged = 1;
    }
  }
  mergeComparisons(makeArrayRef(Comparisons_)
                       .slice(Comparisons_.size() - NumMerged, NumMerged),
                   nullptr, Phi_, TLI);

  return true;
}

void BCECmpChain::mergeComparisons(ArrayRef<BCECmpBlock> Comparisons,
                                   BasicBlock *const NextBBInChain,
                                   PHINode &Phi,
                                   const TargetLibraryInfo *const TLI) {
  assert(!Comparisons.empty());
  const auto &FirstComparison = *Comparisons.begin();
  BasicBlock *const BB = FirstComparison.BB;
  LLVMContext &Context = BB->getContext();

  if (Comparisons.size() >= 2) {
    DEBUG(dbgs() << "Merging " << Comparisons.size() << " comparisons\n");
    const auto TotalSize =
        std::accumulate(Comparisons.begin(), Comparisons.end(), 0,
                        [](int Size, const BCECmpBlock &C) {
                          return Size + C.SizeBits();
                        }) /
        8;

    // Incoming edges do not need to be updated, and both GEPs are already
    // computing the right address, we just need to:
    //   - replace the two loads and the icmp with the memcmp
    //   - update the branch
    //   - update the incoming values in the phi.
    FirstComparison.BranchI->eraseFromParent();
    FirstComparison.CmpI->eraseFromParent();
    FirstComparison.Lhs().LoadI->eraseFromParent();
    FirstComparison.Rhs().LoadI->eraseFromParent();

    IRBuilder<> Builder(BB);
    const auto &DL = Phi.getModule()->getDataLayout();
    Value *const MemCmpCall =
        emitMemCmp(FirstComparison.Lhs().GEP, FirstComparison.Rhs().GEP,
                   ConstantInt::get(DL.getIntPtrType(Context), TotalSize),
                   Builder, DL, TLI);
    Value *const MemCmpIsZero = Builder.CreateICmpEQ(
        MemCmpCall, ConstantInt::get(Type::getInt32Ty(Context), 0));

    // Add a branch to the next basic block in the chain.
    if (NextBBInChain) {
      Builder.CreateCondBr(MemCmpIsZero, NextBBInChain, Phi.getParent());
      Phi.addIncoming(ConstantInt::getFalse(Context), BB);
    } else {
      Builder.CreateBr(Phi.getParent());
      Phi.addIncoming(MemCmpIsZero, BB);
    }

    // Delete merged blocks.
    for (size_t I = 1; I < Comparisons.size(); ++I) {
      BasicBlock *CBB = Comparisons[I].BB;
      CBB->replaceAllUsesWith(BB);
      CBB->eraseFromParent();
    }
  } else {
    assert(Comparisons.size() == 1);
    // There are no blocks to merge, but we still need to update the branches.
    DEBUG(dbgs() << "Only one comparison, updating branches\n");
    if (NextBBInChain) {
      if (FirstComparison.BranchI->isConditional()) {
        DEBUG(dbgs() << "conditional -> conditional\n");
        // Just update the "true" target, the "false" target should already be
        // the phi block.
        assert(FirstComparison.BranchI->getSuccessor(1) == Phi.getParent());
        FirstComparison.BranchI->setSuccessor(0, NextBBInChain);
        Phi.addIncoming(ConstantInt::getFalse(Context), BB);
      } else {
        DEBUG(dbgs() << "unconditional -> conditional\n");
        // Replace the unconditional branch by a conditional one.
        FirstComparison.BranchI->eraseFromParent();
        IRBuilder<> Builder(BB);
        Builder.CreateCondBr(FirstComparison.CmpI, NextBBInChain,
                             Phi.getParent());
        Phi.addIncoming(FirstComparison.CmpI, BB);
      }
    } else {
      if (FirstComparison.BranchI->isConditional()) {
        DEBUG(dbgs() << "conditional -> unconditional\n");
        // Replace the conditional branch by an unconditional one.
        FirstComparison.BranchI->eraseFromParent();
        IRBuilder<> Builder(BB);
        Builder.CreateBr(Phi.getParent());
        Phi.addIncoming(FirstComparison.CmpI, BB);
      } else {
        DEBUG(dbgs() << "unconditional -> unconditional\n");
        Phi.addIncoming(FirstComparison.CmpI, BB);
      }
    }
  }
}

std::vector<BasicBlock *> getOrderedBlocks(PHINode &Phi,
                                           BasicBlock *const LastBlock,
                                           int NumBlocks) {
  // Walk up from the last block to find other blocks.
  std::vector<BasicBlock *> Blocks(NumBlocks);
  BasicBlock *CurBlock = LastBlock;
  for (int BlockIndex = NumBlocks - 1; BlockIndex > 0; --BlockIndex) {
    if (CurBlock->hasAddressTaken()) {
      // Somebody is jumping to the block through an address, all bets are
      // off.
      DEBUG(dbgs() << "skip: block " << BlockIndex
                   << " has its address taken\n");
      return {};
    }
    Blocks[BlockIndex] = CurBlock;
    auto *SinglePredecessor = CurBlock->getSinglePredecessor();
    if (!SinglePredecessor) {
      // The block has two or more predecessors.
      DEBUG(dbgs() << "skip: block " << BlockIndex
                   << " has two or more predecessors\n");
      return {};
    }
    if (Phi.getBasicBlockIndex(SinglePredecessor) < 0) {
      // The block does not link back to the phi.
      DEBUG(dbgs() << "skip: block " << BlockIndex
                   << " does not link back to the phi\n");
      return {};
    }
    CurBlock = SinglePredecessor;
  }
  Blocks[0] = CurBlock;
  return Blocks;
}

bool processPhi(PHINode &Phi, const TargetLibraryInfo *const TLI) {
  DEBUG(dbgs() << "processPhi()\n");
  if (Phi.getNumIncomingValues() <= 1) {
    DEBUG(dbgs() << "skip: only one incoming value in phi\n");
    return false;
  }
  // We are looking for something that has the following structure:
  //   bb1 --eq--> bb2 --eq--> bb3 --eq--> bb4 --+
  //     \            \           \               \
  //      ne           ne          ne              \
  //       \            \           \               v
  //        +------------+-----------+----------> bb_phi
  //
  //  - The last basic block (bb4 here) must branch unconditionally to bb_phi.
  //    It's the only block that contributes a non-constant value to the Phi.
  //  - All other blocks (b1, b2, b3) must have exactly two successors, one of
  //    them being the the phi block.
  //  - All intermediate blocks (bb2, bb3) must have only one predecessor.
  //  - Blocks cannot do other work besides the comparison, see doesOtherWork()

  // The blocks are not necessarily ordered in the phi, so we start from the
  // last block and reconstruct the order.
  BasicBlock *LastBlock = nullptr;
  for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
    if (isa<ConstantInt>(Phi.getIncomingValue(I)))
      continue;
    if (LastBlock) {
      // There are several non-constant values.
      DEBUG(dbgs() << "skip: several non-constant values\n");
      return false;
    }
    LastBlock = Phi.getIncomingBlock(I);
  }
  if (!LastBlock) {
    // There is no non-constant block.
    DEBUG(dbgs() << "skip: no non-constant block\n");
    return false;
  }
  if (LastBlock->getSingleSuccessor() != Phi.getParent()) {
    DEBUG(dbgs() << "skip: last block non-phi successor\n");
    return false;
  }

  const auto Blocks =
      getOrderedBlocks(Phi, LastBlock, Phi.getNumIncomingValues());
  if (Blocks.empty())
    return false;
  BCECmpChain CmpChain(Blocks, Phi);

  if (CmpChain.size() < 2) {
    DEBUG(dbgs() << "skip: only one compare block\n");
    return false;
  }

  return CmpChain.simplify(TLI);
}

class MergeICmps : public FunctionPass {
 public:
  static char ID;

  MergeICmps() : FunctionPass(ID) {
    initializeMergeICmpsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) return false;
    const auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto PA = runImpl(F, &TLI);
    return !PA.areAllPreserved();
  }

 private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  PreservedAnalyses runImpl(Function &F, const TargetLibraryInfo *TLI);
};

PreservedAnalyses MergeICmps::runImpl(Function &F,
                                      const TargetLibraryInfo *TLI) {
  DEBUG(dbgs() << "MergeICmpsPass: " << F.getName() << "\n");

  bool MadeChange = false;

  for (auto BBIt = ++F.begin(); BBIt != F.end(); ++BBIt) {
    // A Phi operation is always first in a basic block.
    if (auto *const Phi = dyn_cast<PHINode>(&*BBIt->begin()))
      MadeChange |= processPhi(*Phi, TLI);
  }

  if (MadeChange)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

}  // namespace

char MergeICmps::ID = 0;
INITIALIZE_PASS_BEGIN(MergeICmps, "mergeicmps",
                      "Merge contiguous icmps into a memcmp", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(MergeICmps, "mergeicmps",
                    "Merge contiguous icmps into a memcmp", false, false)

Pass *llvm::createMergeICmpsPass() { return new MergeICmps(); }

