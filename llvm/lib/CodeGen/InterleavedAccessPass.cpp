//===- InterleavedAccessPass.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Interleaved Access pass, which identifies
// interleaved memory accesses and transforms them into target specific
// intrinsics.
//
// An interleaved load reads data from memory into several vectors, with
// DE-interleaving the data on a factor. An interleaved store writes several
// vectors to memory with RE-interleaving the data on a factor.
//
// As interleaved accesses are difficult to identified in CodeGen (mainly
// because the VECTOR_SHUFFLE DAG node is quite different from the shufflevector
// IR), we identify and transform them to intrinsics in this pass so the
// intrinsics can be easily matched into target specific instructions later in
// CodeGen.
//
// E.g. An interleaved load (Factor = 2):
//        %wide.vec = load <8 x i32>, <8 x i32>* %ptr
//        %v0 = shuffle <8 x i32> %wide.vec, <8 x i32> poison, <0, 2, 4, 6>
//        %v1 = shuffle <8 x i32> %wide.vec, <8 x i32> poison, <1, 3, 5, 7>
//
// It could be transformed into a ld2 intrinsic in AArch64 backend or a vld2
// intrinsic in ARM backend.
//
// In X86, this can be further optimized into a set of target
// specific loads followed by an optimized sequence of shuffles.
//
// E.g. An interleaved store (Factor = 3):
//        %i.vec = shuffle <8 x i32> %v0, <8 x i32> %v1,
//                                    <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>
//        store <12 x i32> %i.vec, <12 x i32>* %ptr
//
// It could be transformed into a st3 intrinsic in AArch64 backend or a vst3
// intrinsic in ARM backend.
//
// Similarly, a set of interleaved stores can be transformed into an optimized
// sequence of shuffles followed by a set of target specific stores for X86.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "interleaved-access"

static cl::opt<bool> LowerInterleavedAccesses(
    "lower-interleaved-accesses",
    cl::desc("Enable lowering interleaved accesses to intrinsics"),
    cl::init(true), cl::Hidden);

namespace {

class InterleavedAccessImpl {
  friend class InterleavedAccess;

public:
  InterleavedAccessImpl() = default;
  InterleavedAccessImpl(DominatorTree *DT, const TargetLowering *TLI)
      : DT(DT), TLI(TLI), MaxFactor(TLI->getMaxSupportedInterleaveFactor()) {}
  bool runOnFunction(Function &F);

private:
  DominatorTree *DT = nullptr;
  const TargetLowering *TLI = nullptr;

  /// The maximum supported interleave factor.
  unsigned MaxFactor = 0u;

  /// Transform an interleaved load into target specific intrinsics.
  bool lowerInterleavedLoad(Instruction *Load,
                            SmallSetVector<Instruction *, 32> &DeadInsts);

  /// Transform an interleaved store into target specific intrinsics.
  bool lowerInterleavedStore(Instruction *Store,
                             SmallSetVector<Instruction *, 32> &DeadInsts);

  /// Transform a load and a deinterleave intrinsic into target specific
  /// instructions.
  bool lowerDeinterleaveIntrinsic(IntrinsicInst *II,
                                  SmallSetVector<Instruction *, 32> &DeadInsts);

  /// Transform an interleave intrinsic and a store into target specific
  /// instructions.
  bool lowerInterleaveIntrinsic(IntrinsicInst *II,
                                SmallSetVector<Instruction *, 32> &DeadInsts);

  /// Returns true if the uses of an interleaved load by the
  /// extractelement instructions in \p Extracts can be replaced by uses of the
  /// shufflevector instructions in \p Shuffles instead. If so, the necessary
  /// replacements are also performed.
  bool tryReplaceExtracts(ArrayRef<ExtractElementInst *> Extracts,
                          ArrayRef<ShuffleVectorInst *> Shuffles);

  /// Given a number of shuffles of the form shuffle(binop(x,y)), convert them
  /// to binop(shuffle(x), shuffle(y)) to allow the formation of an
  /// interleaving load. Any newly created shuffles that operate on \p LI will
  /// be added to \p Shuffles. Returns true, if any changes to the IR have been
  /// made.
  bool replaceBinOpShuffles(ArrayRef<ShuffleVectorInst *> BinOpShuffles,
                            SmallVectorImpl<ShuffleVectorInst *> &Shuffles,
                            Instruction *LI);
};

class InterleavedAccess : public FunctionPass {
  InterleavedAccessImpl Impl;

public:
  static char ID;

  InterleavedAccess() : FunctionPass(ID) {
    initializeInterleavedAccessPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Interleaved Access Pass"; }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace.

PreservedAnalyses InterleavedAccessPass::run(Function &F,
                                             FunctionAnalysisManager &FAM) {
  auto *DT = &FAM.getResult<DominatorTreeAnalysis>(F);
  auto *TLI = TM->getSubtargetImpl(F)->getTargetLowering();
  InterleavedAccessImpl Impl(DT, TLI);
  bool Changed = Impl.runOnFunction(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char InterleavedAccess::ID = 0;

bool InterleavedAccess::runOnFunction(Function &F) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC || !LowerInterleavedAccesses)
    return false;

  LLVM_DEBUG(dbgs() << "*** " << getPassName() << ": " << F.getName() << "\n");

  Impl.DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &TM = TPC->getTM<TargetMachine>();
  Impl.TLI = TM.getSubtargetImpl(F)->getTargetLowering();
  Impl.MaxFactor = Impl.TLI->getMaxSupportedInterleaveFactor();

  return Impl.runOnFunction(F);
}

INITIALIZE_PASS_BEGIN(InterleavedAccess, DEBUG_TYPE,
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(InterleavedAccess, DEBUG_TYPE,
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)

FunctionPass *llvm::createInterleavedAccessPass() {
  return new InterleavedAccess();
}

/// Check if the mask is a DE-interleave mask for an interleaved load.
///
/// E.g. DE-interleave masks (Factor = 2) could be:
///     <0, 2, 4, 6>    (mask of index 0 to extract even elements)
///     <1, 3, 5, 7>    (mask of index 1 to extract odd elements)
static bool isDeInterleaveMask(ArrayRef<int> Mask, unsigned &Factor,
                               unsigned &Index, unsigned MaxFactor,
                               unsigned NumLoadElements) {
  if (Mask.size() < 2)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++) {
    // Make sure we don't produce a load wider than the input load.
    if (Mask.size() * Factor > NumLoadElements)
      return false;
    if (ShuffleVectorInst::isDeInterleaveMaskOfFactor(Mask, Factor, Index))
      return true;
  }

  return false;
}

/// Check if the mask can be used in an interleaved store.
//
/// It checks for a more general pattern than the RE-interleave mask.
/// I.e. <x, y, ... z, x+1, y+1, ...z+1, x+2, y+2, ...z+2, ...>
/// E.g. For a Factor of 2 (LaneLen=4): <4, 32, 5, 33, 6, 34, 7, 35>
/// E.g. For a Factor of 3 (LaneLen=4): <4, 32, 16, 5, 33, 17, 6, 34, 18, 7, 35, 19>
/// E.g. For a Factor of 4 (LaneLen=2): <8, 2, 12, 4, 9, 3, 13, 5>
///
/// The particular case of an RE-interleave mask is:
/// I.e. <0, LaneLen, ... , LaneLen*(Factor - 1), 1, LaneLen + 1, ...>
/// E.g. For a Factor of 2 (LaneLen=4): <0, 4, 1, 5, 2, 6, 3, 7>
static bool isReInterleaveMask(ShuffleVectorInst *SVI, unsigned &Factor,
                               unsigned MaxFactor) {
  unsigned NumElts = SVI->getShuffleMask().size();
  if (NumElts < 4)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++) {
    if (SVI->isInterleave(Factor))
      return true;
  }

  return false;
}

// Return the corresponded deinterleaved mask, or nullptr if there is no valid
// mask.
static Value *getMask(Value *WideMask, unsigned Factor,
                      ElementCount LeafValueEC);

static Value *getMask(Value *WideMask, unsigned Factor,
                      VectorType *LeafValueTy) {
  return getMask(WideMask, Factor, LeafValueTy->getElementCount());
}

bool InterleavedAccessImpl::lowerInterleavedLoad(
    Instruction *Load, SmallSetVector<Instruction *, 32> &DeadInsts) {
  if (isa<ScalableVectorType>(Load->getType()))
    return false;

  if (auto *LI = dyn_cast<LoadInst>(Load)) {
    if (!LI->isSimple())
      return false;
  } else if (auto *VPLoad = dyn_cast<VPIntrinsic>(Load)) {
    assert(VPLoad->getIntrinsicID() == Intrinsic::vp_load);
    // Require a constant mask.
    if (!isa<ConstantVector>(VPLoad->getMaskParam()))
      return false;
  } else {
    llvm_unreachable("unsupported load operation");
  }

  // Check if all users of this load are shufflevectors. If we encounter any
  // users that are extractelement instructions or binary operators, we save
  // them to later check if they can be modified to extract from one of the
  // shufflevectors instead of the load.

  SmallVector<ShuffleVectorInst *, 4> Shuffles;
  SmallVector<ExtractElementInst *, 4> Extracts;
  // BinOpShuffles need to be handled a single time in case both operands of the
  // binop are the same load.
  SmallSetVector<ShuffleVectorInst *, 4> BinOpShuffles;

  for (auto *User : Load->users()) {
    auto *Extract = dyn_cast<ExtractElementInst>(User);
    if (Extract && isa<ConstantInt>(Extract->getIndexOperand())) {
      Extracts.push_back(Extract);
      continue;
    }
    if (auto *BI = dyn_cast<BinaryOperator>(User)) {
      if (!BI->user_empty() && all_of(BI->users(), [](auto *U) {
            auto *SVI = dyn_cast<ShuffleVectorInst>(U);
            return SVI && isa<UndefValue>(SVI->getOperand(1));
          })) {
        for (auto *SVI : BI->users())
          BinOpShuffles.insert(cast<ShuffleVectorInst>(SVI));
        continue;
      }
    }
    auto *SVI = dyn_cast<ShuffleVectorInst>(User);
    if (!SVI || !isa<UndefValue>(SVI->getOperand(1)))
      return false;

    Shuffles.push_back(SVI);
  }

  if (Shuffles.empty() && BinOpShuffles.empty())
    return false;

  unsigned Factor, Index;

  unsigned NumLoadElements =
      cast<FixedVectorType>(Load->getType())->getNumElements();
  auto *FirstSVI = Shuffles.size() > 0 ? Shuffles[0] : BinOpShuffles[0];
  // Check if the first shufflevector is DE-interleave shuffle.
  if (!isDeInterleaveMask(FirstSVI->getShuffleMask(), Factor, Index, MaxFactor,
                          NumLoadElements))
    return false;

  // Holds the corresponding index for each DE-interleave shuffle.
  SmallVector<unsigned, 4> Indices;

  Type *VecTy = FirstSVI->getType();

  // Check if other shufflevectors are also DE-interleaved of the same type
  // and factor as the first shufflevector.
  for (auto *Shuffle : Shuffles) {
    if (Shuffle->getType() != VecTy)
      return false;
    if (!ShuffleVectorInst::isDeInterleaveMaskOfFactor(
            Shuffle->getShuffleMask(), Factor, Index))
      return false;

    assert(Shuffle->getShuffleMask().size() <= NumLoadElements);
    Indices.push_back(Index);
  }
  for (auto *Shuffle : BinOpShuffles) {
    if (Shuffle->getType() != VecTy)
      return false;
    if (!ShuffleVectorInst::isDeInterleaveMaskOfFactor(
            Shuffle->getShuffleMask(), Factor, Index))
      return false;

    assert(Shuffle->getShuffleMask().size() <= NumLoadElements);

    if (cast<Instruction>(Shuffle->getOperand(0))->getOperand(0) == Load)
      Indices.push_back(Index);
    if (cast<Instruction>(Shuffle->getOperand(0))->getOperand(1) == Load)
      Indices.push_back(Index);
  }

  // Try and modify users of the load that are extractelement instructions to
  // use the shufflevector instructions instead of the load.
  if (!tryReplaceExtracts(Extracts, Shuffles))
    return false;

  bool BinOpShuffleChanged =
      replaceBinOpShuffles(BinOpShuffles.getArrayRef(), Shuffles, Load);

  if (auto *VPLoad = dyn_cast<VPIntrinsic>(Load)) {
    Value *LaneMask =
        getMask(VPLoad->getMaskParam(), Factor, cast<VectorType>(VecTy));
    if (!LaneMask)
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found an interleaved vp.load: " << *Load << "\n");

    // Sometimes the number of Shuffles might be less than Factor, we have to
    // fill the gaps with null. Also, lowerInterleavedVPLoad
    // expects them to be sorted.
    SmallVector<Value *, 4> ShuffleValues(Factor, nullptr);
    for (auto [Idx, ShuffleMaskIdx] : enumerate(Indices))
      ShuffleValues[ShuffleMaskIdx] = Shuffles[Idx];
    if (!TLI->lowerInterleavedVPLoad(VPLoad, LaneMask, ShuffleValues))
      // If Extracts is not empty, tryReplaceExtracts made changes earlier.
      return !Extracts.empty() || BinOpShuffleChanged;
  } else {
    LLVM_DEBUG(dbgs() << "IA: Found an interleaved load: " << *Load << "\n");

    // Try to create target specific intrinsics to replace the load and
    // shuffles.
    if (!TLI->lowerInterleavedLoad(cast<LoadInst>(Load), Shuffles, Indices,
                                   Factor))
      // If Extracts is not empty, tryReplaceExtracts made changes earlier.
      return !Extracts.empty() || BinOpShuffleChanged;
  }

  DeadInsts.insert_range(Shuffles);

  DeadInsts.insert(Load);
  return true;
}

bool InterleavedAccessImpl::replaceBinOpShuffles(
    ArrayRef<ShuffleVectorInst *> BinOpShuffles,
    SmallVectorImpl<ShuffleVectorInst *> &Shuffles, Instruction *Load) {
  for (auto *SVI : BinOpShuffles) {
    BinaryOperator *BI = cast<BinaryOperator>(SVI->getOperand(0));
    Type *BIOp0Ty = BI->getOperand(0)->getType();
    ArrayRef<int> Mask = SVI->getShuffleMask();
    assert(all_of(Mask, [&](int Idx) {
      return Idx < (int)cast<FixedVectorType>(BIOp0Ty)->getNumElements();
    }));

    BasicBlock::iterator insertPos = SVI->getIterator();
    auto *NewSVI1 =
        new ShuffleVectorInst(BI->getOperand(0), PoisonValue::get(BIOp0Ty),
                              Mask, SVI->getName(), insertPos);
    auto *NewSVI2 = new ShuffleVectorInst(
        BI->getOperand(1), PoisonValue::get(BI->getOperand(1)->getType()), Mask,
        SVI->getName(), insertPos);
    BinaryOperator *NewBI = BinaryOperator::CreateWithCopiedFlags(
        BI->getOpcode(), NewSVI1, NewSVI2, BI, BI->getName(), insertPos);
    SVI->replaceAllUsesWith(NewBI);
    LLVM_DEBUG(dbgs() << "  Replaced: " << *BI << "\n    And   : " << *SVI
                      << "\n  With    : " << *NewSVI1 << "\n    And   : "
                      << *NewSVI2 << "\n    And   : " << *NewBI << "\n");
    RecursivelyDeleteTriviallyDeadInstructions(SVI);
    if (NewSVI1->getOperand(0) == Load)
      Shuffles.push_back(NewSVI1);
    if (NewSVI2->getOperand(0) == Load)
      Shuffles.push_back(NewSVI2);
  }

  return !BinOpShuffles.empty();
}

bool InterleavedAccessImpl::tryReplaceExtracts(
    ArrayRef<ExtractElementInst *> Extracts,
    ArrayRef<ShuffleVectorInst *> Shuffles) {
  // If there aren't any extractelement instructions to modify, there's nothing
  // to do.
  if (Extracts.empty())
    return true;

  // Maps extractelement instructions to vector-index pairs. The extractlement
  // instructions will be modified to use the new vector and index operands.
  DenseMap<ExtractElementInst *, std::pair<Value *, int>> ReplacementMap;

  for (auto *Extract : Extracts) {
    // The vector index that is extracted.
    auto *IndexOperand = cast<ConstantInt>(Extract->getIndexOperand());
    auto Index = IndexOperand->getSExtValue();

    // Look for a suitable shufflevector instruction. The goal is to modify the
    // extractelement instruction (which uses an interleaved load) to use one
    // of the shufflevector instructions instead of the load.
    for (auto *Shuffle : Shuffles) {
      // If the shufflevector instruction doesn't dominate the extract, we
      // can't create a use of it.
      if (!DT->dominates(Shuffle, Extract))
        continue;

      // Inspect the indices of the shufflevector instruction. If the shuffle
      // selects the same index that is extracted, we can modify the
      // extractelement instruction.
      SmallVector<int, 4> Indices;
      Shuffle->getShuffleMask(Indices);
      for (unsigned I = 0; I < Indices.size(); ++I)
        if (Indices[I] == Index) {
          assert(Extract->getOperand(0) == Shuffle->getOperand(0) &&
                 "Vector operations do not match");
          ReplacementMap[Extract] = std::make_pair(Shuffle, I);
          break;
        }

      // If we found a suitable shufflevector instruction, stop looking.
      if (ReplacementMap.count(Extract))
        break;
    }

    // If we did not find a suitable shufflevector instruction, the
    // extractelement instruction cannot be modified, so we must give up.
    if (!ReplacementMap.count(Extract))
      return false;
  }

  // Finally, perform the replacements.
  IRBuilder<> Builder(Extracts[0]->getContext());
  for (auto &Replacement : ReplacementMap) {
    auto *Extract = Replacement.first;
    auto *Vector = Replacement.second.first;
    auto Index = Replacement.second.second;
    Builder.SetInsertPoint(Extract);
    Extract->replaceAllUsesWith(Builder.CreateExtractElement(Vector, Index));
    Extract->eraseFromParent();
  }

  return true;
}

bool InterleavedAccessImpl::lowerInterleavedStore(
    Instruction *Store, SmallSetVector<Instruction *, 32> &DeadInsts) {
  Value *StoredValue;
  if (auto *SI = dyn_cast<StoreInst>(Store)) {
    if (!SI->isSimple())
      return false;
    StoredValue = SI->getValueOperand();
  } else if (auto *VPStore = dyn_cast<VPIntrinsic>(Store)) {
    assert(VPStore->getIntrinsicID() == Intrinsic::vp_store);
    // Require a constant mask.
    if (!isa<ConstantVector>(VPStore->getMaskParam()))
      return false;
    StoredValue = VPStore->getArgOperand(0);
  } else {
    llvm_unreachable("unsupported store operation");
  }

  auto *SVI = dyn_cast<ShuffleVectorInst>(StoredValue);
  if (!SVI || !SVI->hasOneUse() || isa<ScalableVectorType>(SVI->getType()))
    return false;

  unsigned NumStoredElements =
      cast<FixedVectorType>(SVI->getType())->getNumElements();
  // Check if the shufflevector is RE-interleave shuffle.
  unsigned Factor;
  if (!isReInterleaveMask(SVI, Factor, MaxFactor))
    return false;
  assert(NumStoredElements % Factor == 0 &&
         "number of stored element should be a multiple of Factor");

  if (auto *VPStore = dyn_cast<VPIntrinsic>(Store)) {
    unsigned LaneMaskLen = NumStoredElements / Factor;
    Value *LaneMask = getMask(VPStore->getMaskParam(), Factor,
                              ElementCount::getFixed(LaneMaskLen));
    if (!LaneMask)
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found an interleaved vp.store: " << *Store
                      << "\n");

    IRBuilder<> Builder(VPStore);
    // We need to effectively de-interleave the shufflemask
    // because lowerInterleavedVPStore expects individual de-interleaved
    // values.
    SmallVector<Value *, 10> NewShuffles;
    SmallVector<int, 16> NewShuffleMask(LaneMaskLen);
    auto ShuffleMask = SVI->getShuffleMask();

    for (unsigned i = 0; i < Factor; i++) {
      for (unsigned j = 0; j < LaneMaskLen; j++)
        NewShuffleMask[j] = ShuffleMask[i + Factor * j];

      NewShuffles.push_back(Builder.CreateShuffleVector(
          SVI->getOperand(0), SVI->getOperand(1), NewShuffleMask));
    }

    // Try to create target specific intrinsics to replace the vp.store and
    // shuffle.
    if (!TLI->lowerInterleavedVPStore(VPStore, LaneMask, NewShuffles))
      // We already created new shuffles.
      return true;
  } else {
    LLVM_DEBUG(dbgs() << "IA: Found an interleaved store: " << *Store << "\n");

    // Try to create target specific intrinsics to replace the store and
    // shuffle.
    if (!TLI->lowerInterleavedStore(cast<StoreInst>(Store), SVI, Factor))
      return false;
  }

  // Already have a new target specific interleaved store. Erase the old store.
  DeadInsts.insert(Store);
  DeadInsts.insert(SVI);
  return true;
}

static unsigned getIntrinsicFactor(const IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::vector_deinterleave2:
  case Intrinsic::vector_interleave2:
    return 2;
  case Intrinsic::vector_deinterleave3:
  case Intrinsic::vector_interleave3:
    return 3;
  case Intrinsic::vector_deinterleave5:
  case Intrinsic::vector_interleave5:
    return 5;
  case Intrinsic::vector_deinterleave7:
  case Intrinsic::vector_interleave7:
    return 7;
  default:
    llvm_unreachable("Unexpected intrinsic");
  }
}

// For an (de)interleave tree like this:
//
//   A   C B   D
//   |___| |___|
//     |_____|
//        |
//     A B C D
//
//  We will get ABCD at the end while the leaf operands/results
//  are ACBD, which are also what we initially collected in
//  getVectorInterleaveFactor / getVectorDeinterleaveFactor. But TLI
//  hooks (e.g. lowerDeinterleaveIntrinsicToLoad) expect ABCD, so we need
//  to reorder them by interleaving these values.
static void interleaveLeafValues(MutableArrayRef<Value *> SubLeaves) {
  unsigned NumLeaves = SubLeaves.size();
  if (NumLeaves == 2 || !isPowerOf2_64(NumLeaves))
    return;

  assert(isPowerOf2_32(NumLeaves) && NumLeaves > 1);

  const unsigned HalfLeaves = NumLeaves / 2;
  // Visit the sub-trees.
  interleaveLeafValues(SubLeaves.take_front(HalfLeaves));
  interleaveLeafValues(SubLeaves.drop_front(HalfLeaves));

  SmallVector<Value *, 8> Buffer;
  //    a0 a1 a2 a3 b0 b1 b2 b3
  // -> a0 b0 a1 b1 a2 b2 a3 b3
  for (unsigned i = 0U; i < NumLeaves; ++i)
    Buffer.push_back(SubLeaves[i / 2 + (i % 2 ? HalfLeaves : 0)]);

  llvm::copy(Buffer, SubLeaves.begin());
}

static bool
getVectorInterleaveFactor(IntrinsicInst *II, SmallVectorImpl<Value *> &Operands,
                          SmallVectorImpl<Instruction *> &DeadInsts) {
  assert(II->getIntrinsicID() == Intrinsic::vector_interleave2 ||
         II->getIntrinsicID() == Intrinsic::vector_interleave3 ||
         II->getIntrinsicID() == Intrinsic::vector_interleave5 ||
         II->getIntrinsicID() == Intrinsic::vector_interleave7);

  // Visit with BFS
  SmallVector<IntrinsicInst *, 8> Queue;
  Queue.push_back(II);
  while (!Queue.empty()) {
    IntrinsicInst *Current = Queue.front();
    Queue.erase(Queue.begin());

    // All the intermediate intrinsics will be deleted.
    DeadInsts.push_back(Current);

    for (unsigned I = 0; I < getIntrinsicFactor(Current); ++I) {
      Value *Op = Current->getOperand(I);
      if (auto *OpII = dyn_cast<IntrinsicInst>(Op))
        if (OpII->getIntrinsicID() == Intrinsic::vector_interleave2) {
          Queue.push_back(OpII);
          continue;
        }

      // If this is not a perfectly balanced tree, the leaf
      // result types would be different.
      if (!Operands.empty() && Op->getType() != Operands.back()->getType())
        return false;

      Operands.push_back(Op);
    }
  }

  const unsigned Factor = Operands.size();
  // Currently we only recognize factors of 3, 5, 7, and powers of 2.
  // FIXME: should we assert here instead?
  if (Factor <= 1 ||
      (!isPowerOf2_32(Factor) && Factor != getIntrinsicFactor(II)))
    return false;

  interleaveLeafValues(Operands);
  return true;
}

static bool
getVectorDeinterleaveFactor(IntrinsicInst *II,
                            SmallVectorImpl<Value *> &Results,
                            SmallVectorImpl<Instruction *> &DeadInsts) {
  assert(II->getIntrinsicID() == Intrinsic::vector_deinterleave2 ||
         II->getIntrinsicID() == Intrinsic::vector_deinterleave3 ||
         II->getIntrinsicID() == Intrinsic::vector_deinterleave5 ||
         II->getIntrinsicID() == Intrinsic::vector_deinterleave7);
  using namespace PatternMatch;
  if (!II->hasNUses(getIntrinsicFactor(II)))
    return false;

  // Visit with BFS
  SmallVector<IntrinsicInst *, 8> Queue;
  Queue.push_back(II);
  while (!Queue.empty()) {
    IntrinsicInst *Current = Queue.front();
    Queue.erase(Queue.begin());
    assert(Current->hasNUses(getIntrinsicFactor(Current)));

    // All the intermediate intrinsics will be deleted from the bottom-up.
    DeadInsts.insert(DeadInsts.begin(), Current);

    SmallVector<ExtractValueInst *> EVs(getIntrinsicFactor(Current), nullptr);
    for (User *Usr : Current->users()) {
      if (!isa<ExtractValueInst>(Usr))
        return 0;

      auto *EV = cast<ExtractValueInst>(Usr);
      // Intermediate ExtractValue instructions will also be deleted.
      DeadInsts.insert(DeadInsts.begin(), EV);
      ArrayRef<unsigned> Indices = EV->getIndices();
      if (Indices.size() != 1)
        return false;

      if (!EVs[Indices[0]])
        EVs[Indices[0]] = EV;
      else
        return false;
    }

    // We have legal indices. At this point we're either going
    // to continue the traversal or push the leaf values into Results.
    for (ExtractValueInst *EV : EVs) {
      // Continue the traversal. We're playing safe here and matching only the
      // expression consisting of a perfectly balanced binary tree in which all
      // intermediate values are only used once.
      if (EV->hasOneUse() &&
          match(EV->user_back(),
                m_Intrinsic<Intrinsic::vector_deinterleave2>()) &&
          EV->user_back()->hasNUses(2)) {
        auto *EVUsr = cast<IntrinsicInst>(EV->user_back());
        Queue.push_back(EVUsr);
        continue;
      }

      // If this is not a perfectly balanced tree, the leaf
      // result types would be different.
      if (!Results.empty() && EV->getType() != Results.back()->getType())
        return false;

      // Save the leaf value.
      Results.push_back(EV);
    }
  }

  const unsigned Factor = Results.size();
  // Currently we only recognize factors of 3, 5, 7, and powers of 2.
  // FIXME: should we assert here instead?
  if (Factor <= 1 ||
      (!isPowerOf2_32(Factor) && Factor != getIntrinsicFactor(II)))
    return 0;

  interleaveLeafValues(Results);
  return true;
}

static Value *getMask(Value *WideMask, unsigned Factor,
                      ElementCount LeafValueEC) {
  if (auto *IMI = dyn_cast<IntrinsicInst>(WideMask)) {
    SmallVector<Value *, 8> Operands;
    SmallVector<Instruction *, 8> DeadInsts;
    if (getVectorInterleaveFactor(IMI, Operands, DeadInsts)) {
      assert(!Operands.empty());
      if (Operands.size() == Factor && llvm::all_equal(Operands))
        return Operands[0];
    }
  }

  if (auto *ConstMask = dyn_cast<Constant>(WideMask)) {
    if (auto *Splat = ConstMask->getSplatValue())
      // All-ones or all-zeros mask.
      return ConstantVector::getSplat(LeafValueEC, Splat);

    if (LeafValueEC.isFixed()) {
      unsigned LeafMaskLen = LeafValueEC.getFixedValue();
      SmallVector<Constant *, 8> LeafMask(LeafMaskLen, nullptr);
      // If this is a fixed-length constant mask, each lane / leaf has to
      // use the same mask. This is done by checking if every group with Factor
      // number of elements in the interleaved mask has homogeneous values.
      for (unsigned Idx = 0U; Idx < LeafMaskLen * Factor; ++Idx) {
        Constant *C = ConstMask->getAggregateElement(Idx);
        if (LeafMask[Idx / Factor] && LeafMask[Idx / Factor] != C)
          return nullptr;
        LeafMask[Idx / Factor] = C;
      }

      return ConstantVector::get(LeafMask);
    }
  }

  return nullptr;
}

bool InterleavedAccessImpl::lowerDeinterleaveIntrinsic(
    IntrinsicInst *DI, SmallSetVector<Instruction *, 32> &DeadInsts) {
  Value *LoadedVal = DI->getOperand(0);
  if (!LoadedVal->hasOneUse() || !isa<LoadInst, VPIntrinsic>(LoadedVal))
    return false;

  SmallVector<Value *, 8> DeinterleaveValues;
  SmallVector<Instruction *, 8> DeinterleaveDeadInsts;
  if (!getVectorDeinterleaveFactor(DI, DeinterleaveValues,
                                   DeinterleaveDeadInsts))
    return false;

  const unsigned Factor = DeinterleaveValues.size();

  if (auto *VPLoad = dyn_cast<VPIntrinsic>(LoadedVal)) {
    if (VPLoad->getIntrinsicID() != Intrinsic::vp_load)
      return false;
    // Check mask operand. Handle both all-true/false and interleaved mask.
    Value *WideMask = VPLoad->getOperand(1);
    Value *Mask = getMask(WideMask, Factor,
                          cast<VectorType>(DeinterleaveValues[0]->getType()));
    if (!Mask)
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found a vp.load with deinterleave intrinsic "
                      << *DI << " and factor = " << Factor << "\n");

    // Since lowerInterleaveLoad expects Shuffles and LoadInst, use special
    // TLI function to emit target-specific interleaved instruction.
    if (!TLI->lowerInterleavedVPLoad(VPLoad, Mask, DeinterleaveValues))
      return false;

  } else {
    auto *LI = cast<LoadInst>(LoadedVal);
    if (!LI->isSimple())
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found a load with deinterleave intrinsic " << *DI
                      << " and factor = " << Factor << "\n");

    // Try and match this with target specific intrinsics.
    if (!TLI->lowerDeinterleaveIntrinsicToLoad(LI, DeinterleaveValues))
      return false;
  }

  DeadInsts.insert_range(DeinterleaveDeadInsts);
  // We now have a target-specific load, so delete the old one.
  DeadInsts.insert(cast<Instruction>(LoadedVal));
  return true;
}

bool InterleavedAccessImpl::lowerInterleaveIntrinsic(
    IntrinsicInst *II, SmallSetVector<Instruction *, 32> &DeadInsts) {
  if (!II->hasOneUse())
    return false;
  Value *StoredBy = II->user_back();
  if (!isa<StoreInst, VPIntrinsic>(StoredBy))
    return false;

  SmallVector<Value *, 8> InterleaveValues;
  SmallVector<Instruction *, 8> InterleaveDeadInsts;
  if (!getVectorInterleaveFactor(II, InterleaveValues, InterleaveDeadInsts))
    return false;

  const unsigned Factor = InterleaveValues.size();

  if (auto *VPStore = dyn_cast<VPIntrinsic>(StoredBy)) {
    if (VPStore->getIntrinsicID() != Intrinsic::vp_store)
      return false;

    Value *WideMask = VPStore->getOperand(2);
    Value *Mask = getMask(WideMask, Factor,
                          cast<VectorType>(InterleaveValues[0]->getType()));
    if (!Mask)
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found a vp.store with interleave intrinsic "
                      << *II << " and factor = " << Factor << "\n");

    // Since lowerInterleavedStore expects Shuffle and StoreInst, use special
    // TLI function to emit target-specific interleaved instruction.
    if (!TLI->lowerInterleavedVPStore(VPStore, Mask, InterleaveValues))
      return false;
  } else {
    auto *SI = cast<StoreInst>(StoredBy);
    if (!SI->isSimple())
      return false;

    LLVM_DEBUG(dbgs() << "IA: Found a store with interleave intrinsic " << *II
                      << " and factor = " << Factor << "\n");

    // Try and match this with target specific intrinsics.
    if (!TLI->lowerInterleaveIntrinsicToStore(SI, InterleaveValues))
      return false;
  }

  // We now have a target-specific store, so delete the old one.
  DeadInsts.insert(cast<Instruction>(StoredBy));
  DeadInsts.insert_range(InterleaveDeadInsts);
  return true;
}

bool InterleavedAccessImpl::runOnFunction(Function &F) {
  // Holds dead instructions that will be erased later.
  SmallSetVector<Instruction *, 32> DeadInsts;
  bool Changed = false;

  using namespace PatternMatch;
  for (auto &I : instructions(F)) {
    if (match(&I, m_CombineOr(m_Load(m_Value()),
                              m_Intrinsic<Intrinsic::vp_load>())))
      Changed |= lowerInterleavedLoad(&I, DeadInsts);

    if (match(&I, m_CombineOr(m_Store(m_Value(), m_Value()),
                              m_Intrinsic<Intrinsic::vp_store>())))
      Changed |= lowerInterleavedStore(&I, DeadInsts);

    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      // At present, we only have intrinsics to represent (de)interleaving
      // with a factor of 2,3,5 and 7.
      switch (II->getIntrinsicID()) {
      case Intrinsic::vector_deinterleave2:
      case Intrinsic::vector_deinterleave3:
      case Intrinsic::vector_deinterleave5:
      case Intrinsic::vector_deinterleave7:
        Changed |= lowerDeinterleaveIntrinsic(II, DeadInsts);
        break;
      case Intrinsic::vector_interleave2:
      case Intrinsic::vector_interleave3:
      case Intrinsic::vector_interleave5:
      case Intrinsic::vector_interleave7:
        Changed |= lowerInterleaveIntrinsic(II, DeadInsts);
        break;
      default:
        break;
      }
    }
  }

  for (auto *I : DeadInsts)
    I->eraseFromParent();

  return Changed;
}
