
//===- AArch64LoopIdiomTransform.cpp - Loop idiom recognition -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64LoopIdiomTransform.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-lit"

static cl::opt<bool>
    DisableAll("disable-aarch64-lit-all", cl::Hidden, cl::init(false),
               cl::desc("Disable AArch64 Loop Idiom Transform Pass."));

static cl::opt<bool> DisableByteCmp(
    "disable-aarch64-lit-bytecmp", cl::Hidden, cl::init(false),
    cl::desc("Proceed with AArch64 Loop Idiom Transform Pass, but do "
             "not convert byte-compare loop(s)."));

namespace llvm {

void initializeAArch64LoopIdiomTransformLegacyPassPass(PassRegistry &);
Pass *createAArch64LoopIdiomTransformPass();

} // end namespace llvm

namespace {

class AArch64LoopIdiomTransform {
  Loop *CurLoop = nullptr;
  DominatorTree *DT;
  LoopInfo *LI;
  const TargetTransformInfo *TTI;
  const DataLayout *DL;

public:
  explicit AArch64LoopIdiomTransform(DominatorTree *DT, LoopInfo *LI,
                                     const TargetTransformInfo *TTI,
                                     const DataLayout *DL)
      : DT(DT), LI(LI), TTI(TTI), DL(DL) {}

  bool run(Loop *L);

private:
  /// \name Countable Loop Idiom Handling
  /// @{

  bool runOnCountableLoop();
  bool runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                      SmallVectorImpl<BasicBlock *> &ExitBlocks);

  bool recognizeByteCompare();
  Value *expandFindMismatch(IRBuilder<> &Builder, GetElementPtrInst *GEPA,
                            GetElementPtrInst *GEPB, Value *Start,
                            Value *MaxLen);
  void transformByteCompare(GetElementPtrInst *GEPA, GetElementPtrInst *GEPB,
                            Value *MaxLen, Value *Index, Value *Start,
                            bool IncIdx, BasicBlock *FoundBB,
                            BasicBlock *EndBB);
  /// @}
};

class AArch64LoopIdiomTransformLegacyPass : public LoopPass {
public:
  static char ID;

  explicit AArch64LoopIdiomTransformLegacyPass() : LoopPass(ID) {
    initializeAArch64LoopIdiomTransformLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Recognize AArch64-specific loop idioms";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;
};

bool AArch64LoopIdiomTransformLegacyPass::runOnLoop(Loop *L,
                                                    LPPassManager &LPM) {

  if (skipLoop(L))
    return false;

  auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
      *L->getHeader()->getParent());
  return AArch64LoopIdiomTransform(
             DT, LI, &TTI, &L->getHeader()->getModule()->getDataLayout())
      .run(L);
}

} // end anonymous namespace

char AArch64LoopIdiomTransformLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    AArch64LoopIdiomTransformLegacyPass, "aarch64-lit",
    "Transform specific loop idioms into optimised vector forms", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(
    AArch64LoopIdiomTransformLegacyPass, "aarch64-lit",
    "Transform specific loop idioms into optimised vector forms", false, false)

Pass *llvm::createAArch64LoopIdiomTransformPass() {
  return new AArch64LoopIdiomTransformLegacyPass();
}

PreservedAnalyses
AArch64LoopIdiomTransformPass::run(Loop &L, LoopAnalysisManager &AM,
                                   LoopStandardAnalysisResults &AR,
                                   LPMUpdater &) {
  if (DisableAll)
    return PreservedAnalyses::all();

  const auto *DL = &L.getHeader()->getModule()->getDataLayout();

  AArch64LoopIdiomTransform LIT(&AR.DT, &AR.LI, &AR.TTI, DL);
  if (!LIT.run(&L))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

//===----------------------------------------------------------------------===//
//
//          Implementation of AArch64LoopIdiomTransform
//
//===----------------------------------------------------------------------===//

bool AArch64LoopIdiomTransform::run(Loop *L) {
  CurLoop = L;

  if (DisableAll)
    return false;

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!L->getLoopPreheader())
    return false;

  LLVM_DEBUG(dbgs() << DEBUG_TYPE " Scanning: F["
                    << CurLoop->getHeader()->getParent()->getName()
                    << "] Loop %" << CurLoop->getHeader()->getName() << "\n");

  return recognizeByteCompare();
}

/// Match loop-invariant value.
template <typename SubPattern_t> struct match_LoopInvariant {
  SubPattern_t SubPattern;
  const Loop *L;

  match_LoopInvariant(const SubPattern_t &SP, const Loop *L)
      : SubPattern(SP), L(L) {}

  template <typename ITy> bool match(ITy *V) {
    return L->isLoopInvariant(V) && SubPattern.match(V);
  }
};

/// Matches if the value is loop-invariant.
template <typename Ty>
inline match_LoopInvariant<Ty> m_LoopInvariant(const Ty &M, const Loop *L) {
  return match_LoopInvariant<Ty>(M, L);
}

bool AArch64LoopIdiomTransform::recognizeByteCompare() {
  if (!TTI->supportsScalableVectors() || !TTI->getMinPageSize().has_value() ||
      DisableByteCmp)
    return false;

  BasicBlock *Header = CurLoop->getHeader();
  BasicBlock *PH = CurLoop->getLoopPreheader();

  // In AArch64LoopIdiomTransform::run we have already checked that the loop
  // has a preheader so we can assume it's in a canonical form.
  auto *EntryBI = cast<BranchInst>(PH->getTerminator());

  if (CurLoop->getNumBackEdges() != 1 || CurLoop->getNumBlocks() != 2)
    return false;

  PHINode *PN = dyn_cast<PHINode>(&Header->front());
  if (!PN || PN->getNumIncomingValues() != 2)
    return false;

  auto LoopBlocks = CurLoop->getBlocks();
  // The first block in the loop should contain only 4 instructions, e.g.
  //
  //  while.cond:
  //   %res.phi = phi i32 [ %start, %ph ], [ %inc, %while.body ]
  //   %inc = add i32 %res.phi, 1
  //   %cmp.not = icmp eq i32 %inc, %n
  //   br i1 %cmp.not, label %while.end, label %while.body
  //
  auto CondBBInsts = LoopBlocks[0]->instructionsWithoutDebug();
  if (std::distance(CondBBInsts.begin(), CondBBInsts.end()) > 4)
    return false;

  // The second block should contain 7 instructions, e.g.
  //
  // while.body:
  //   %idx = zext i32 %inc to i64
  //   %idx.a = getelementptr inbounds i8, ptr %a, i64 %idx
  //   %load.a = load i8, ptr %idx.a
  //   %idx.b = getelementptr inbounds i8, ptr %b, i64 %idx
  //   %load.b = load i8, ptr %idx.b
  //   %cmp.not.ld = icmp eq i8 %load.a, %load.b
  //   br i1 %cmp.not.ld, label %while.cond, label %while.end
  //
  auto LoopBBInsts = LoopBlocks[1]->instructionsWithoutDebug();
  if (std::distance(LoopBBInsts.begin(), LoopBBInsts.end()) > 7)
    return false;

  using namespace PatternMatch;

  // The incoming value to the PHI node from the loop should be an add of 1.
  Instruction *Index = nullptr;
  Value *StartIdx = nullptr;
  for (BasicBlock *BB : PN->blocks()) {
    if (!CurLoop->contains(BB)) {
      StartIdx = PN->getIncomingValueForBlock(BB);
      continue;
    }
    Index = dyn_cast<Instruction>(PN->getIncomingValueForBlock(BB));
    // Limit to 32-bit types for now
    if (!Index || !Index->getType()->isIntegerTy(32) ||
        !match(Index, m_c_Add(m_Specific(PN), m_One())))
      return false;
  }

  // If we match the pattern, PN and Index will be replaced with the result of
  // the cttz.elts intrinsic. If any other instructions are used outside of
  // the loop, we cannot replace it.
  for (BasicBlock *BB : LoopBlocks)
    for (Instruction &I : *BB)
      if (&I != PN && &I != Index)
        for (User *U : I.users()) {
          auto UI = cast<Instruction>(U);
          if (!CurLoop->contains(UI))
            return false;
        }

  // Don't replace the loop if the add has a wrap flag.
  if (Index->hasNoSignedWrap() || Index->hasNoUnsignedWrap())
    return false;

  // Match the branch instruction for the header
  ICmpInst::Predicate Pred;
  Value *MaxLen;
  BasicBlock *EndBB, *WhileBB;
  if (!match(Header->getTerminator(),
             m_Br(m_ICmp(Pred, m_Specific(Index), m_Value(MaxLen)),
                  m_BasicBlock(EndBB), m_BasicBlock(WhileBB))) ||
      Pred != ICmpInst::Predicate::ICMP_EQ || !CurLoop->contains(WhileBB))
    return false;

  // WhileBB should contain the pattern of load & compare instructions. Match
  // the pattern and find the GEP instructions used by the loads.
  ICmpInst::Predicate WhilePred;
  BasicBlock *FoundBB;
  BasicBlock *TrueBB;
  Value *LoadA, *LoadB;
  if (!match(WhileBB->getTerminator(),
             m_Br(m_ICmp(WhilePred, m_Value(LoadA), m_Value(LoadB)),
                  m_BasicBlock(TrueBB), m_BasicBlock(FoundBB))) ||
      WhilePred != ICmpInst::Predicate::ICMP_EQ || !CurLoop->contains(TrueBB))
    return false;

  Value *A, *B;
  if (!match(LoadA, m_Load(m_Value(A))) || !match(LoadB, m_Load(m_Value(B))))
    return false;

  GetElementPtrInst *GEPA = dyn_cast<GetElementPtrInst>(A);
  GetElementPtrInst *GEPB = dyn_cast<GetElementPtrInst>(B);

  if (!GEPA || !GEPB)
    return false;

  Value *PtrA = GEPA->getPointerOperand();
  Value *PtrB = GEPB->getPointerOperand();

  // Check we are loading i8 values from two loop invariant pointers
  if (!CurLoop->isLoopInvariant(PtrA) || !CurLoop->isLoopInvariant(PtrB) ||
      !GEPA->getResultElementType()->isIntegerTy(8) ||
      !GEPB->getResultElementType()->isIntegerTy(8) ||
      !cast<LoadInst>(LoadA)->getType()->isIntegerTy(8) ||
      !cast<LoadInst>(LoadB)->getType()->isIntegerTy(8) || PtrA == PtrB)
    return false;

  // Check that the index to the GEPs is the index we found earlier
  if (GEPA->getNumIndices() > 1 || GEPB->getNumIndices() > 1)
    return false;

  Value *IdxA = GEPA->getOperand(GEPA->getNumIndices());
  Value *IdxB = GEPB->getOperand(GEPB->getNumIndices());
  if (IdxA != IdxB || !match(IdxA, m_ZExt(m_Specific(Index))))
    return false;

  LLVM_DEBUG(dbgs() << "FOUND IDIOM IN LOOP: \n"
                    << *(EndBB->getParent()) << "\n\n");

  // The index is incremented before the GEP/Load pair so we need to
  // add 1 to the start value.
  transformByteCompare(GEPA, GEPB, MaxLen, Index, StartIdx, /*IncIdx=*/true, FoundBB,
                       EndBB);
  return true;
}

Value *AArch64LoopIdiomTransform::expandFindMismatch(IRBuilder<> &Builder,
                                                     GetElementPtrInst *GEPA,
                                                     GetElementPtrInst *GEPB,
                                                     Value *Start,
                                                     Value *MaxLen) {
  Value *PtrA = GEPA->getPointerOperand();
  Value *PtrB = GEPB->getPointerOperand();

  // Get the arguments and types for the intrinsic.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  BranchInst *PHBranch = cast<BranchInst>(Preheader->getTerminator());
  LLVMContext &Ctx = PHBranch->getContext();
  Type *LoadType = Type::getInt8Ty(Ctx);
  Type *ResType = Builder.getInt32Ty();

  // Split block in the original loop preheader.
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  BasicBlock *EndBlock =
      SplitBlock(Preheader, PHBranch, DT, LI, nullptr, "mismatch_end");

  // Create the blocks that we're going to need:
  //  1. A block for checking the zero-extended length exceeds 0
  //  2. A block to check that the start and end addresses of a given array
  //     lie on the same page.
  //  3. The SVE loop preheader.
  //  4. The first SVE loop block.
  //  5. The SVE loop increment block.
  //  6. A block we can jump to from the SVE loop when a mismatch is found.
  //  7. The first block of the scalar loop itself, containing PHIs , loads
  //  and cmp.
  //  8. A scalar loop increment block to increment the PHIs and go back
  //  around the loop.

  BasicBlock *MinItCheckBlock = BasicBlock::Create(
      Ctx, "mismatch_min_it_check", EndBlock->getParent(), EndBlock);

  // Update the terminator added by SplitBlock to branch to the first block
  Preheader->getTerminator()->setSuccessor(0, MinItCheckBlock);

  BasicBlock *MemCheckBlock = BasicBlock::Create(
      Ctx, "mismatch_mem_check", EndBlock->getParent(), EndBlock);

  BasicBlock *SVELoopPreheaderBlock = BasicBlock::Create(
      Ctx, "mismatch_sve_loop_preheader", EndBlock->getParent(), EndBlock);

  BasicBlock *SVELoopStartBlock = BasicBlock::Create(
      Ctx, "mismatch_sve_loop", EndBlock->getParent(), EndBlock);

  BasicBlock *SVELoopIncBlock = BasicBlock::Create(
      Ctx, "mismatch_sve_loop_inc", EndBlock->getParent(), EndBlock);

  BasicBlock *SVELoopMismatchBlock = BasicBlock::Create(
      Ctx, "mismatch_sve_loop_found", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopPreHeaderBlock = BasicBlock::Create(
      Ctx, "mismatch_loop_pre", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopStartBlock =
      BasicBlock::Create(Ctx, "mismatch_loop", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopIncBlock = BasicBlock::Create(
      Ctx, "mismatch_loop_inc", EndBlock->getParent(), EndBlock);

  DTU.applyUpdates({{DominatorTree::Insert, Preheader, MinItCheckBlock},
                    {DominatorTree::Delete, Preheader, EndBlock}});

  // Update LoopInfo with the new SVE & scalar loops.
  auto SVELoop = LI->AllocateLoop();
  auto ScalarLoop = LI->AllocateLoop();
  if (CurLoop->getParentLoop()) {
    CurLoop->getParentLoop()->addChildLoop(SVELoop);
    CurLoop->getParentLoop()->addChildLoop(ScalarLoop);
  } else {
    LI->addTopLevelLoop(SVELoop);
    LI->addTopLevelLoop(ScalarLoop);
  }

  // Add the new basic blocks to their associated loops.
  SVELoop->addBasicBlockToLoop(MinItCheckBlock, *LI);
  SVELoop->addBasicBlockToLoop(MemCheckBlock, *LI);
  SVELoop->addBasicBlockToLoop(SVELoopPreheaderBlock, *LI);
  SVELoop->addBasicBlockToLoop(SVELoopStartBlock, *LI);
  SVELoop->addBasicBlockToLoop(SVELoopIncBlock, *LI);
  SVELoop->addBasicBlockToLoop(SVELoopMismatchBlock, *LI);

  ScalarLoop->addBasicBlockToLoop(LoopPreHeaderBlock, *LI);
  ScalarLoop->addBasicBlockToLoop(LoopStartBlock, *LI);
  ScalarLoop->addBasicBlockToLoop(LoopIncBlock, *LI);

  // Set up some types and constants that we intend to reuse.
  Type *I64Type = Builder.getInt64Ty();

  // Check the zero-extended iteration count > 0
  Builder.SetInsertPoint(MinItCheckBlock);
  Value *ExtStart = Builder.CreateZExt(Start, I64Type);
  Value *ExtEnd = Builder.CreateZExt(MaxLen, I64Type);
  // This check doesn't really cost us very much.

  Value *LimitCheck = Builder.CreateICmpULE(Start, MaxLen);
  BranchInst *MinItCheckBr =
      BranchInst::Create(MemCheckBlock, LoopPreHeaderBlock, LimitCheck);
  MinItCheckBr->setMetadata(
      LLVMContext::MD_prof,
      MDBuilder(MinItCheckBr->getContext()).createBranchWeights(99, 1));
  Builder.Insert(MinItCheckBr);

  DTU.applyUpdates(
      {{DominatorTree::Insert, MinItCheckBlock, MemCheckBlock},
       {DominatorTree::Insert, MinItCheckBlock, LoopPreHeaderBlock}});

  // For each of the arrays, check the start/end addresses are on the same
  // page.
  Builder.SetInsertPoint(MemCheckBlock);

  // For each start address calculate the offset into the min architecturally
  // allowed page size. Then determine how many bytes there are left on the
  // page and see if this is >= MaxLen.
  Value *LhsStartGEP = Builder.CreateGEP(LoadType, PtrA, ExtStart);
  Value *RhsStartGEP = Builder.CreateGEP(LoadType, PtrB, ExtStart);
  Value *RhsStart = Builder.CreatePtrToInt(RhsStartGEP, I64Type);
  Value *LhsStart = Builder.CreatePtrToInt(LhsStartGEP, I64Type);
  Value *LhsEndGEP = Builder.CreateGEP(LoadType, PtrA, ExtEnd);
  Value *RhsEndGEP = Builder.CreateGEP(LoadType, PtrB, ExtEnd);
  Value *LhsEnd = Builder.CreatePtrToInt(LhsEndGEP, I64Type);
  Value *RhsEnd = Builder.CreatePtrToInt(RhsEndGEP, I64Type);

  const uint64_t MinPageSize = TTI->getMinPageSize().value();
  const uint64_t AddrShiftAmt = llvm::Log2_64(MinPageSize);
  Value *LhsStartPage = Builder.CreateLShr(LhsStart, AddrShiftAmt);
  Value *LhsEndPage = Builder.CreateLShr(LhsEnd, AddrShiftAmt);
  Value *RhsStartPage = Builder.CreateLShr(RhsStart, AddrShiftAmt);
  Value *RhsEndPage = Builder.CreateLShr(RhsEnd, AddrShiftAmt);
  Value *LhsPageCmp = Builder.CreateICmpNE(LhsStartPage, LhsEndPage);
  Value *RhsPageCmp = Builder.CreateICmpNE(RhsStartPage, RhsEndPage);

  Value *CombinedPageCmp = Builder.CreateOr(LhsPageCmp, RhsPageCmp);
  BranchInst *CombinedPageCmpCmpBr = BranchInst::Create(
      LoopPreHeaderBlock, SVELoopPreheaderBlock, CombinedPageCmp);
  CombinedPageCmpCmpBr->setMetadata(
      LLVMContext::MD_prof, MDBuilder(CombinedPageCmpCmpBr->getContext())
                                .createBranchWeights(10, 90));
  Builder.Insert(CombinedPageCmpCmpBr);

  DTU.applyUpdates(
      {{DominatorTree::Insert, MemCheckBlock, LoopPreHeaderBlock},
       {DominatorTree::Insert, MemCheckBlock, SVELoopPreheaderBlock}});

  // Set up the SVE loop preheader, i.e. calculate initial loop predicate,
  // zero-extend MaxLen to 64-bits, determine the number of vector elements
  // processed in each iteration, etc.
  Builder.SetInsertPoint(SVELoopPreheaderBlock);

  // At this point we know two things must be true:
  //  1. Start <= End
  //  2. ExtMaxLen <= 4096 due to the page checks.
  // Therefore, we know that we can use a 64-bit induction variable that
  // starts from 0 -> ExtMaxLen and it will not overflow.
  ScalableVectorType *PredVTy =
      ScalableVectorType::get(Builder.getInt1Ty(), 16);

  Value *InitialPred = Builder.CreateIntrinsic(
      Intrinsic::get_active_lane_mask, {PredVTy, I64Type}, {ExtStart, ExtEnd});

  Value *VecLen = Builder.CreateIntrinsic(Intrinsic::vscale, {I64Type}, {});
  VecLen = Builder.CreateMul(VecLen, ConstantInt::get(I64Type, 16), "",
                             /*HasNUW=*/true, /*HasNSW=*/true);

  Value *PFalse = Builder.CreateVectorSplat(PredVTy->getElementCount(),
                                            Builder.getInt1(false));

  BranchInst *JumpToSVELoop = BranchInst::Create(SVELoopStartBlock);
  Builder.Insert(JumpToSVELoop);

  DTU.applyUpdates(
      {{DominatorTree::Insert, SVELoopPreheaderBlock, SVELoopStartBlock}});

  // Set up the first SVE loop block by creating the PHIs, doing the vector
  // loads and comparing the vectors.
  Builder.SetInsertPoint(SVELoopStartBlock);
  PHINode *LoopPred = Builder.CreatePHI(PredVTy, 2, "mismatch_sve_loop_pred");
  LoopPred->addIncoming(InitialPred, SVELoopPreheaderBlock);
  PHINode *SVEIndexPhi = Builder.CreatePHI(I64Type, 2, "mismatch_sve_index");
  SVEIndexPhi->addIncoming(ExtStart, SVELoopPreheaderBlock);
  Type *SVELoadType = ScalableVectorType::get(Builder.getInt8Ty(), 16);
  Value *GepOffset = SVEIndexPhi;
  Value *Passthru = ConstantInt::getNullValue(SVELoadType);

  Value *SVELhsGep = Builder.CreateGEP(LoadType, PtrA, GepOffset);
  if (GEPA->isInBounds())
    cast<GetElementPtrInst>(SVELhsGep)->setIsInBounds(true);
  Value *SVELhsLoad = Builder.CreateMaskedLoad(SVELoadType, SVELhsGep, Align(1),
                                               LoopPred, Passthru);

  Value *SVERhsGep = Builder.CreateGEP(LoadType, PtrB, GepOffset);
  if (GEPB->isInBounds())
    cast<GetElementPtrInst>(SVERhsGep)->setIsInBounds(true);
  Value *SVERhsLoad = Builder.CreateMaskedLoad(SVELoadType, SVERhsGep, Align(1),
                                               LoopPred, Passthru);

  Value *SVEMatchCmp = Builder.CreateICmpNE(SVELhsLoad, SVERhsLoad);
  SVEMatchCmp = Builder.CreateSelect(LoopPred, SVEMatchCmp, PFalse);
  Value *SVEMatchHasActiveLanes = Builder.CreateOrReduce(SVEMatchCmp);
  BranchInst *SVEEarlyExit = BranchInst::Create(
      SVELoopMismatchBlock, SVELoopIncBlock, SVEMatchHasActiveLanes);
  Builder.Insert(SVEEarlyExit);

  DTU.applyUpdates(
      {{DominatorTree::Insert, SVELoopStartBlock, SVELoopMismatchBlock},
       {DominatorTree::Insert, SVELoopStartBlock, SVELoopIncBlock}});

  // Increment the index counter and calculate the predicate for the next
  // iteration of the loop. We branch back to the start of the loop if there
  // is at least one active lane.
  Builder.SetInsertPoint(SVELoopIncBlock);
  Value *NewSVEIndexPhi = Builder.CreateAdd(SVEIndexPhi, VecLen, "",
                                            /*HasNUW=*/true, /*HasNSW=*/true);
  SVEIndexPhi->addIncoming(NewSVEIndexPhi, SVELoopIncBlock);
  Value *NewPred =
      Builder.CreateIntrinsic(Intrinsic::get_active_lane_mask,
                              {PredVTy, I64Type}, {NewSVEIndexPhi, ExtEnd});
  LoopPred->addIncoming(NewPred, SVELoopIncBlock);

  Value *PredHasActiveLanes =
      Builder.CreateExtractElement(NewPred, uint64_t(0));
  BranchInst *SVELoopBranchBack =
      BranchInst::Create(SVELoopStartBlock, EndBlock, PredHasActiveLanes);
  Builder.Insert(SVELoopBranchBack);

  DTU.applyUpdates({{DominatorTree::Insert, SVELoopIncBlock, SVELoopStartBlock},
                    {DominatorTree::Insert, SVELoopIncBlock, EndBlock}});

  // If we found a mismatch then we need to calculate which lane in the vector
  // had a mismatch and add that on to the current loop index.
  Builder.SetInsertPoint(SVELoopMismatchBlock);
  Value *PredMatchCmp = Builder.CreateAnd(LoopPred, SVEMatchCmp);
  Value *Ctz = Builder.CreateIntrinsic(
      Intrinsic::experimental_cttz_elts, {ResType, SVEMatchCmp->getType()},
      {PredMatchCmp, /*ZeroIsPoison=*/Builder.getInt1(true)});
  Ctz = Builder.CreateZExt(Ctz, I64Type);
  Value *SVELoopRes64 = Builder.CreateAdd(SVEIndexPhi, Ctz, "",
                                          /*HasNUW=*/true, /*HasNSW=*/true);
  Value *SVELoopRes = Builder.CreateTrunc(SVELoopRes64, ResType);

  Builder.Insert(BranchInst::Create(EndBlock));

  DTU.applyUpdates({{DominatorTree::Insert, SVELoopMismatchBlock, EndBlock}});

  // Generate code for scalar loop.
  Builder.SetInsertPoint(LoopPreHeaderBlock);
  Builder.Insert(BranchInst::Create(LoopStartBlock));

  DTU.applyUpdates(
      {{DominatorTree::Insert, LoopPreHeaderBlock, LoopStartBlock}});

  Builder.SetInsertPoint(LoopStartBlock);
  PHINode *IndexPhi = Builder.CreatePHI(ResType, 2, "mismatch_index");
  IndexPhi->addIncoming(Start, LoopPreHeaderBlock);

  // Otherwise compare the values
  // Load bytes from each array and compare them.
  GepOffset = Builder.CreateZExt(IndexPhi, I64Type);

  Value *LhsGep = Builder.CreateGEP(LoadType, PtrA, GepOffset);
  if (GEPA->isInBounds())
    cast<GetElementPtrInst>(LhsGep)->setIsInBounds(true);
  Value *LhsLoad = Builder.CreateLoad(LoadType, LhsGep);

  Value *RhsGep = Builder.CreateGEP(LoadType, PtrB, GepOffset);
  if (GEPB->isInBounds())
    cast<GetElementPtrInst>(RhsGep)->setIsInBounds(true);
  Value *RhsLoad = Builder.CreateLoad(LoadType, RhsGep);

  Value *MatchCmp = Builder.CreateICmpEQ(LhsLoad, RhsLoad);
  // If we have a mismatch then exit the loop ...
  BranchInst *MatchCmpBr = BranchInst::Create(LoopIncBlock, EndBlock, MatchCmp);
  Builder.Insert(MatchCmpBr);

  DTU.applyUpdates({{DominatorTree::Insert, LoopStartBlock, LoopIncBlock},
                    {DominatorTree::Insert, LoopStartBlock, EndBlock}});

  // Have we reached the maximum permitted length for the loop?
  Builder.SetInsertPoint(LoopIncBlock);
  Value *PhiInc = Builder.CreateAdd(IndexPhi, ConstantInt::get(ResType, 1));
  IndexPhi->addIncoming(PhiInc, LoopIncBlock);
  Value *IVCmp = Builder.CreateICmpEQ(IndexPhi, MaxLen);
  BranchInst *IVCmpBr = BranchInst::Create(EndBlock, LoopStartBlock, IVCmp);
  Builder.Insert(IVCmpBr);

  DTU.applyUpdates({{DominatorTree::Insert, LoopIncBlock, EndBlock},
                    {DominatorTree::Insert, LoopIncBlock, LoopStartBlock}});

  // In the end block we need to insert a PHI node to deal with three cases:
  //  1. We didn't find a mismatch in the scalar loop, so we return MaxLen.
  //  2. We exitted the scalar loop early due to a mismatch and need to return
  //  the index that we found.
  //  3. We didn't find a mismatch in the SVE loop, so we return MaxLen.
  //  4. We exitted the SVE loop early due to a mismatch and need to return
  //  the index that we found.
  Builder.SetInsertPoint(EndBlock, EndBlock->getFirstInsertionPt());
  PHINode *ResPhi = Builder.CreatePHI(ResType, 4, "mismatch_result");
  ResPhi->addIncoming(MaxLen, LoopIncBlock);
  ResPhi->addIncoming(IndexPhi, LoopStartBlock);
  ResPhi->addIncoming(MaxLen, SVELoopIncBlock);
  ResPhi->addIncoming(SVELoopRes, SVELoopMismatchBlock);

  return Builder.CreateTrunc(ResPhi, ResType);
}

void AArch64LoopIdiomTransform::transformByteCompare(
    GetElementPtrInst *GEPA, GetElementPtrInst *GEPB, Value *MaxLen,
    Value *Index, Value *Start, bool IncIdx, BasicBlock *FoundBB,
    BasicBlock *EndBB) {

  // Insert the byte compare intrinsic at the end of the preheader block
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  BasicBlock *Header = CurLoop->getHeader();
  BranchInst *PHBranch = cast<BranchInst>(Preheader->getTerminator());
  IRBuilder<> Builder(PHBranch);
  Builder.SetCurrentDebugLocation(PHBranch->getDebugLoc());

  // Increment the pointer if this was done before the loads in the loop.
  if (IncIdx)
    Start = Builder.CreateAdd(Start, ConstantInt::get(Start->getType(), 1));

  Value *ByteCmpRes = expandFindMismatch(Builder, GEPA, GEPB, Start, MaxLen);

  // Replaces uses of index & induction Phi with intrinsic (we already
  // checked that the the first instruction of Header is the Phi above).
  auto IndPhi = &Header->front();
  IndPhi->replaceAllUsesWith(ByteCmpRes);
  Index->replaceAllUsesWith(ByteCmpRes);

  assert(PHBranch->isUnconditional() &&
         "Expected preheader to terminate with an unconditional branch.");

  // If no mismatch was found, we can jump to the end block. Create a
  // new basic block for the compare instruction.
  auto *CmpBB = BasicBlock::Create(Preheader->getContext(), "byte.compare",
                                   Preheader->getParent());
  CmpBB->moveBefore(EndBB);

  // Replace the branch in the preheader with an always-true conditional branch.
  // This ensures there is still a reference to the original loop.
  Builder.CreateCondBr(Builder.getTrue(), CmpBB, Header);
  PHBranch->eraseFromParent();

  // Create the branch to either the end or found block depending on the value
  // returned by the intrinsic.
  Builder.SetInsertPoint(CmpBB);
  Value *FoundCmp = Builder.CreateICmpEQ(ByteCmpRes, MaxLen);
  Builder.CreateCondBr(FoundCmp, EndBB, FoundBB);

  auto fixSuccessorPhis = [&](BasicBlock *SuccBB) {
    for (PHINode &PN : SuccBB->phis()) {
      // At this point we've already replaced all uses of the result from the
      // loop with ByteCmp. Look through the incoming values to find ByteCmp,
      // meaning this is a Phi collecting the results of the byte compare.
      bool ResPhi = false;
      for (Value *Op : PN.incoming_values())
        if (Op == CmpBB)
          ResPhi = true;

      // If any of the incoming values were ByteCmp, we need to also add
      // it as an incoming value from CmpBB.
      if (ResPhi)
        PN.addIncoming(ByteCmpRes, CmpBB);
      else {
        // Otherwise, this is a Phi for different values. We should create
        // a new incoming value from CmpBB matching the same value as from
        // the old loop.
        for (BasicBlock *BB : PN.blocks())
          if (CurLoop->contains(BB)) {
            PN.addIncoming(PN.getIncomingValueForBlock(BB), CmpBB);
            break;
          }
      }
    }
  };

  // Ensure all Phis in the successors of CmpBB have an incoming value from it.
  fixSuccessorPhis(EndBB);
  fixSuccessorPhis(FoundBB);

  // The new CmpBB block isn't part of the loop, but will need to be added to
  // the outer loop if there is one.
  if (!CurLoop->isOutermost())
    CurLoop->getParentLoop()->addBasicBlockToLoop(CmpBB, *LI);

  // Update the dominator tree with the new block.
  DT->addNewBlock(CmpBB, Preheader);
}
