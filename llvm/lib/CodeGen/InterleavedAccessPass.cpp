//===--------------------- InterleavedAccessPass.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
//        %v0 = shuffle <8 x i32> %wide.vec, <8 x i32> undef, <0, 2, 4, 6>
//        %v1 = shuffle <8 x i32> %wide.vec, <8 x i32> undef, <1, 3, 5, 7>
//
// It could be transformed into a ld2 intrinsic in AArch64 backend or a vld2
// intrinsic in ARM backend.
//
// E.g. An interleaved store (Factor = 3):
//        %i.vec = shuffle <8 x i32> %v0, <8 x i32> %v1,
//                                    <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>
//        store <12 x i32> %i.vec, <12 x i32>* %ptr
//
// It could be transformed into a st3 intrinsic in AArch64 backend or a vst3
// intrinsic in ARM backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "interleaved-access"

static cl::opt<bool> LowerInterleavedAccesses(
    "lower-interleaved-accesses",
    cl::desc("Enable lowering interleaved accesses to intrinsics"),
    cl::init(true), cl::Hidden);

static unsigned MaxFactor; // The maximum supported interleave factor.

namespace {

class InterleavedAccess : public FunctionPass {

public:
  static char ID;
  InterleavedAccess(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), DT(nullptr), TM(TM), TLI(nullptr) {
    initializeInterleavedAccessPass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override { return "Interleaved Access Pass"; }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

private:
  DominatorTree *DT;
  const TargetMachine *TM;
  const TargetLowering *TLI;

  /// \brief Transform an interleaved load into target specific intrinsics.
  bool lowerInterleavedLoad(LoadInst *LI,
                            SmallVector<Instruction *, 32> &DeadInsts);

  /// \brief Transform an interleaved store into target specific intrinsics.
  bool lowerInterleavedStore(StoreInst *SI,
                             SmallVector<Instruction *, 32> &DeadInsts);

  /// \brief Returns true if the uses of an interleaved load by the
  /// extractelement instructions in \p Extracts can be replaced by uses of the
  /// shufflevector instructions in \p Shuffles instead. If so, the necessary
  /// replacements are also performed.
  bool tryReplaceExtracts(ArrayRef<ExtractElementInst *> Extracts,
                          ArrayRef<ShuffleVectorInst *> Shuffles);
};
} // end anonymous namespace.

char InterleavedAccess::ID = 0;
INITIALIZE_TM_PASS_BEGIN(
    InterleavedAccess, "interleaved-access",
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_TM_PASS_END(
    InterleavedAccess, "interleaved-access",
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)

FunctionPass *llvm::createInterleavedAccessPass(const TargetMachine *TM) {
  return new InterleavedAccess(TM);
}

/// \brief Check if the mask is a DE-interleave mask of the given factor
/// \p Factor like:
///     <Index, Index+Factor, ..., Index+(NumElts-1)*Factor>
static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor,
                                       unsigned &Index) {
  // Check all potential start indices from 0 to (Factor - 1).
  for (Index = 0; Index < Factor; Index++) {
    unsigned i = 0;

    // Check that elements are in ascending order by Factor. Ignore undef
    // elements.
    for (; i < Mask.size(); i++)
      if (Mask[i] >= 0 && static_cast<unsigned>(Mask[i]) != Index + i * Factor)
        break;

    if (i == Mask.size())
      return true;
  }

  return false;
}

/// \brief Check if the mask is a DE-interleave mask for an interleaved load.
///
/// E.g. DE-interleave masks (Factor = 2) could be:
///     <0, 2, 4, 6>    (mask of index 0 to extract even elements)
///     <1, 3, 5, 7>    (mask of index 1 to extract odd elements)
static bool isDeInterleaveMask(ArrayRef<int> Mask, unsigned &Factor,
                               unsigned &Index) {
  if (Mask.size() < 2)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++)
    if (isDeInterleaveMaskOfFactor(Mask, Factor, Index))
      return true;

  return false;
}

/// \brief Check if the mask is RE-interleave mask for an interleaved store.
///
/// I.e. <0, NumSubElts, ... , NumSubElts*(Factor - 1), 1, NumSubElts + 1, ...>
///
/// E.g. The RE-interleave mask (Factor = 2) could be:
///     <0, 4, 1, 5, 2, 6, 3, 7>
static bool isReInterleaveMask(ArrayRef<int> Mask, unsigned &Factor) {
  unsigned NumElts = Mask.size();
  if (NumElts < 4)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++) {
    if (NumElts % Factor)
      continue;

    unsigned NumSubElts = NumElts / Factor;
    if (!isPowerOf2_32(NumSubElts))
      continue;

    // Check whether each element matchs the RE-interleaved rule. Ignore undef
    // elements.
    unsigned i = 0;
    for (; i < NumElts; i++)
      if (Mask[i] >= 0 &&
          static_cast<unsigned>(Mask[i]) !=
              (i % Factor) * NumSubElts + i / Factor)
        break;

    // Find a RE-interleaved mask of current factor.
    if (i == NumElts)
      return true;
  }

  return false;
}

bool InterleavedAccess::lowerInterleavedLoad(
    LoadInst *LI, SmallVector<Instruction *, 32> &DeadInsts) {
  if (!LI->isSimple())
    return false;

  SmallVector<ShuffleVectorInst *, 4> Shuffles;
  SmallVector<ExtractElementInst *, 4> Extracts;

  // Check if all users of this load are shufflevectors. If we encounter any
  // users that are extractelement instructions, we save them to later check if
  // they can be modifed to extract from one of the shufflevectors instead of
  // the load.
  for (auto UI = LI->user_begin(), E = LI->user_end(); UI != E; UI++) {
    auto *Extract = dyn_cast<ExtractElementInst>(*UI);
    if (Extract && isa<ConstantInt>(Extract->getIndexOperand())) {
      Extracts.push_back(Extract);
      continue;
    }
    ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(*UI);
    if (!SVI || !isa<UndefValue>(SVI->getOperand(1)))
      return false;

    Shuffles.push_back(SVI);
  }

  if (Shuffles.empty())
    return false;

  unsigned Factor, Index;

  // Check if the first shufflevector is DE-interleave shuffle.
  if (!isDeInterleaveMask(Shuffles[0]->getShuffleMask(), Factor, Index))
    return false;

  // Holds the corresponding index for each DE-interleave shuffle.
  SmallVector<unsigned, 4> Indices;
  Indices.push_back(Index);

  Type *VecTy = Shuffles[0]->getType();

  // Check if other shufflevectors are also DE-interleaved of the same type
  // and factor as the first shufflevector.
  for (unsigned i = 1; i < Shuffles.size(); i++) {
    if (Shuffles[i]->getType() != VecTy)
      return false;

    if (!isDeInterleaveMaskOfFactor(Shuffles[i]->getShuffleMask(), Factor,
                                    Index))
      return false;

    Indices.push_back(Index);
  }

  // Try and modify users of the load that are extractelement instructions to
  // use the shufflevector instructions instead of the load.
  if (!tryReplaceExtracts(Extracts, Shuffles))
    return false;

  DEBUG(dbgs() << "IA: Found an interleaved load: " << *LI << "\n");

  // Try to create target specific intrinsics to replace the load and shuffles.
  if (!TLI->lowerInterleavedLoad(LI, Shuffles, Indices, Factor))
    return false;

  for (auto SVI : Shuffles)
    DeadInsts.push_back(SVI);

  DeadInsts.push_back(LI);
  return true;
}

bool InterleavedAccess::tryReplaceExtracts(
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

bool InterleavedAccess::lowerInterleavedStore(
    StoreInst *SI, SmallVector<Instruction *, 32> &DeadInsts) {
  if (!SI->isSimple())
    return false;

  ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(SI->getValueOperand());
  if (!SVI || !SVI->hasOneUse())
    return false;

  // Check if the shufflevector is RE-interleave shuffle.
  unsigned Factor;
  if (!isReInterleaveMask(SVI->getShuffleMask(), Factor))
    return false;

  DEBUG(dbgs() << "IA: Found an interleaved store: " << *SI << "\n");

  // Try to create target specific intrinsics to replace the store and shuffle.
  if (!TLI->lowerInterleavedStore(SI, SVI, Factor))
    return false;

  // Already have a new target specific interleaved store. Erase the old store.
  DeadInsts.push_back(SI);
  DeadInsts.push_back(SVI);
  return true;
}

bool InterleavedAccess::runOnFunction(Function &F) {
  if (!TM || !LowerInterleavedAccesses)
    return false;

  DEBUG(dbgs() << "*** " << getPassName() << ": " << F.getName() << "\n");

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  TLI = TM->getSubtargetImpl(F)->getTargetLowering();
  MaxFactor = TLI->getMaxSupportedInterleaveFactor();

  // Holds dead instructions that will be erased later.
  SmallVector<Instruction *, 32> DeadInsts;
  bool Changed = false;

  for (auto &I : instructions(F)) {
    if (LoadInst *LI = dyn_cast<LoadInst>(&I))
      Changed |= lowerInterleavedLoad(LI, DeadInsts);

    if (StoreInst *SI = dyn_cast<StoreInst>(&I))
      Changed |= lowerInterleavedStore(SI, DeadInsts);
  }

  for (auto I : DeadInsts)
    I->eraseFromParent();

  return Changed;
}
