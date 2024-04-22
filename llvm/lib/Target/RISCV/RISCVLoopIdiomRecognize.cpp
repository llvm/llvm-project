//===-------- RISCVLoopIdiomRecognize.cpp - Loop idiom recognition --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVLoopIdiomRecognize.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-loop-idiom"

static cl::opt<bool>
    DisableAll("riscv-disable-all-loop-idiom", cl::Hidden, cl::init(true),
               cl::desc("Disable RISCV Loop Idiom Recognize Pass."));

static cl::opt<bool> DisableByteCmp(
    "disable-riscv-loop-idiom-bytecmp", cl::Hidden, cl::init(false),
    cl::desc("Proceed with RISCV Loop Idiom Recognize Pass, but do "
             "not convert byte-compare loop(s)."));

// CustomLoopIdiomLMUL can be used to customize LMUL for vectorizing loops.
// It uses the exponent value to represent LMUL i.e. 0 -> LMUL 1, 1 -> LMUL 2, 2
// -> LMUL 4, 3 -> LMUL 8, etc.
static cl::opt<unsigned>
    CustomLoopIdiomLMUL("riscv-loop-idiom-lmul", cl::Hidden, cl::init(1),
                        cl::Optional,
                        cl::desc("Customize LMUL for vector loop."));

namespace {

class RISCVLoopIdiomRecognize {
  Loop *CurLoop = nullptr;
  DominatorTree &DT;
  LoopInfo &LI;
  TargetLibraryInfo &TLI;
  const TargetTransformInfo &TTI;
  const DataLayout &DL;

public:
  explicit RISCVLoopIdiomRecognize(DominatorTree &DT, LoopInfo &LI,
                                   TargetLibraryInfo &TLI,
                                   const TargetTransformInfo &TTI,
                                   const DataLayout &DL)
      : DT(DT), LI(LI), TLI(TLI), TTI(TTI), DL(DL) {}

  bool run(Loop *L);

private:
  /// \name Countable Loop Idiom Handling
  /// @{

  bool runOnCountableLoop();
  bool runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                      SmallVectorImpl<BasicBlock *> &ExitBlocks);

  bool recognizeAndTransformByteCompare();
  Value *expandFindMismatch(IRBuilder<> &Builder, GetElementPtrInst *GEPA,
                            GetElementPtrInst *GEPB, Instruction *Index,
                            Value *Start, Value *MaxLen);
  void transformByteCompare(GetElementPtrInst *GEPA, GetElementPtrInst *GEPB,
                            PHINode *IndPhi, Value *MaxLen, Instruction *Index,
                            Value *Start, bool IncIdx, BasicBlock *FoundBB,
                            BasicBlock *EndBB);

  /// @}
};
} // end anonymous namespace

static VectorType *getBestVectorTypeForLoopIdiom(LLVMContext &Ctx) {
  unsigned LMULExp = std::min(3U, CustomLoopIdiomLMUL.getValue());
  unsigned VF = (RISCV::RVVBitsPerBlock / 8) << LMULExp;
  ElementCount EC = ElementCount::getScalable(VF);
  return VectorType::get(Type::getInt8Ty(Ctx), EC);
}

PreservedAnalyses
RISCVLoopIdiomRecognizePass::run(Loop &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &) {
  if (DisableAll)
    return PreservedAnalyses::all();

  Function &F = *L.getHeader()->getParent();
  if (F.hasFnAttribute(Attribute::NoImplicitFloat)) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << " is disabled on " << F.getName()
                      << " due to its NoImplicitFloat attribute");
    return PreservedAnalyses::all();
  }

  // Only enabled on RV64 for now.
  if (L.getHeader()->getModule()->getDataLayout().getPointerSizeInBits() != 64)
    return PreservedAnalyses::all();

  // Only enabled when vector extension is present.
  if (!AR.TTI.supportsScalableVectors())
    return PreservedAnalyses::all();

  const auto DL = L.getHeader()->getModule()->getDataLayout();

  RISCVLoopIdiomRecognize LIR(AR.DT, AR.LI, AR.TLI, AR.TTI, DL);
  if (!LIR.run(&L))
    return PreservedAnalyses::all();

  auto PA = PreservedAnalyses::none();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

//===----------------------------------------------------------------------===//
//
//          Implementation of RISCVLoopIdiomRecognize
//
//===----------------------------------------------------------------------===//

bool RISCVLoopIdiomRecognize::run(Loop *L) {
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

  return recognizeAndTransformByteCompare();
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

bool RISCVLoopIdiomRecognize::recognizeAndTransformByteCompare() {
  if (DisableByteCmp)
    return false;

  BasicBlock *PH = CurLoop->getLoopPreheader();

  // The preheader should only contain an unconditional branch.
  if (!PH || &PH->front() != PH->getTerminator())
    return false;

  using namespace PatternMatch;

  BasicBlock *Header;
  if (!match(PH->getTerminator(), m_UnconditionalBr(Header)))
    return false;

  if (Header != CurLoop->getHeader())
    return false;

  if (CurLoop->getNumBackEdges() != 1 || CurLoop->getNumBlocks() != 2)
    return false;

  auto *PN = dyn_cast<PHINode>(&Header->front());
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
  if (std::distance(CondBBInsts.begin(), CondBBInsts.end()) != 4)
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
  if (std::distance(LoopBBInsts.begin(), LoopBBInsts.end()) != 7)
    return false;

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

  for (BasicBlock *BB : LoopBlocks)
    for (Instruction &I : *BB)
      if (&I != PN && &I != Index)
        for (User *U : I.users()) {
          auto UI = dyn_cast<Instruction>(U);
          if (!CurLoop->contains(UI))
            return false;
        }

  // Match the branch instruction for the header
  ICmpInst::Predicate Pred;
  Value *MaxLen;
  BasicBlock *EndBB, *WhileBB;
  if (!match(Header->getTerminator(),
             m_Br(m_ICmp(Pred, m_Specific(Index), m_Value(MaxLen)),
                  m_BasicBlock(EndBB), m_BasicBlock(WhileBB))))
    return false;

  // Make sure Pred is comparing for equal
  if (Pred != ICmpInst::ICMP_EQ)
    return false;

  // Make sure EndBB is outside the loop and WhileBB is inside the loop.
  if (CurLoop->contains(EndBB) || !CurLoop->contains(WhileBB))
    return false;

  // WhileBB should contain the pattern of load & compare instructions. Match
  // the pattern and find the GEP instructions used by the loads.
  ICmpInst::Predicate WhilePred;
  BasicBlock *FoundBB;
  BasicBlock *TrueBB;
  Value *A, *B;
  if (!match(WhileBB->getTerminator(),
             m_Br(m_ICmp(WhilePred, m_Load(m_Value(A)), m_Load(m_Value(B))),
                  m_BasicBlock(TrueBB), m_BasicBlock(FoundBB))))
    return false;

  // Make sure WhilePred is comparing for equal
  if (WhilePred != ICmpInst::ICMP_EQ)
    return false;

  // Make sure TrueBB is the loop header and FoundBB is outside the loop.
  if (CurLoop->getHeader() != TrueBB || CurLoop->contains(FoundBB))
    return false;

  auto *GEPA = dyn_cast<GetElementPtrInst>(A);
  auto *GEPB = dyn_cast<GetElementPtrInst>(B);
  if (!GEPA || !GEPB)
    return false;

  Value *PtrA = GEPA->getPointerOperand();
  Value *PtrB = GEPB->getPointerOperand();

  // Check PtrA and PtrB stride at i8.
  if (!CurLoop->isLoopInvariant(PtrA) || !CurLoop->isLoopInvariant(PtrB) ||
      !GEPA->getResultElementType()->isIntegerTy(8) ||
      !GEPB->getResultElementType()->isIntegerTy(8) || PtrA == PtrB)
    return false;

  // Check loads from GEPA and GEPB are i8.
  auto *LoadA = dyn_cast<LoadInst>(GEPA->getNextNode());
  if (!LoadA || !LoadA->getType()->isIntegerTy(8))
    return false;
  auto *LoadB = dyn_cast<LoadInst>(GEPB->getNextNode());
  if (!LoadB || !LoadB->getType()->isIntegerTy(8))
    return false;

  // Check that the index to the GEPs is the index we found earlier
  if (GEPA->getNumIndices() > 1 || GEPB->getNumIndices() > 1)
    return false;

  Value *IdxA = GEPA->getOperand(GEPA->getNumIndices());
  Value *IdxB = GEPB->getOperand(GEPB->getNumIndices());

  if (IdxA != IdxB || !match(IdxA, m_ZExt(m_Specific(Index))))
    return false;

  // We only ever expect the pre-incremented index value to be used inside the
  // loop.
  if (!PN->hasOneUse())
    return false;

  // Ensure that when the Found and End blocks are identical the PHIs have the
  // supported format. We don't currently allow cases like this:
  // while.cond:
  //   ...
  //   br i1 %cmp.not, label %while.end, label %while.body
  //
  // while.body:
  //   ...
  //   br i1 %cmp.not2, label %while.cond, label %while.end
  //
  // while.end:
  //   %final_ptr = phi ptr [ %c, %while.body ], [ %d, %while.cond ]
  //
  // Where the incoming values for %final_ptr are unique and from each of the
  // loop blocks, but not actually defined in the loop. This requires extra
  // work setting up the byte.compare block, i.e. by introducing a select to
  // choose the correct value.
  // TODO: We could add support for this in future.
  if (FoundBB == EndBB) {
    for (PHINode &EndPN : EndBB->phis()) {
      Value *WhileCondVal = EndPN.getIncomingValueForBlock(Header);
      Value *WhileBodyVal = EndPN.getIncomingValueForBlock(WhileBB);

      // The value of the index when leaving the while.cond block is always the
      // same as the end value (MaxLen) so we permit either. Otherwise for any
      // other value defined outside the loop we only allow values that are the
      // same as the exit value for while.body.
      if (WhileCondVal != WhileBodyVal &&
          ((WhileCondVal != Index && WhileCondVal != MaxLen) ||
           (WhileBodyVal != Index && WhileBodyVal != MaxLen)))
        return false;
    }
  }

  LLVM_DEBUG(dbgs() << "FOUND IDIOM IN LOOP: \n"
                    << *(EndBB->getParent()) << "\n\n");
  transformByteCompare(GEPA, GEPB, PN, MaxLen, Index, StartIdx, true, FoundBB,
                       EndBB);
  LLVM_DEBUG(dbgs() << "AFTER IDIOM TRANSFORMATION: \n"
                    << *(EndBB->getParent()) << "\n\n");
  return true;
}

Value *RISCVLoopIdiomRecognize::expandFindMismatch(
    IRBuilder<> &Builder, GetElementPtrInst *GEPA, GetElementPtrInst *GEPB,
    Instruction *Index, Value *Start, Value *MaxLen) {
  Value *PtrA = GEPA->getPointerOperand();
  Value *PtrB = GEPB->getPointerOperand();

  // Get the arguments and types for the intrinsic.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  auto *PHBranch = cast<BranchInst>(Preheader->getTerminator());
  LLVMContext &Ctx = PHBranch->getContext();
  Type *LoadType = Type::getInt8Ty(Ctx);
  Type *ResType = Builder.getInt32Ty();

  // Split block at the original callsite, where the EndBlock continues from
  // where the original call ended.
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  BasicBlock *EndBlock =
      SplitBlock(Preheader, PHBranch, &DT, &LI, nullptr, "mismatch_end");

  // Safeguard to check if we build the correct DomTree with DTU.
  auto CheckDTU = llvm::make_scope_exit([&]() {
    assert(DTU.getDomTree().verify() && "Ill-formed DomTree built by DTU");
  });

  // Create the blocks that we're going to need:
  //  1. A block for checking the zero-extended length exceeds 0
  //  2. A block to check that the start and end addresses of a given array
  //     lie on the same page.
  //  3. The RVV loop preheader i.e. vector_loop_preheader
  //  4. The first RVV loop block i.e. vector_loop
  //  5. The RVV loop increment block i.e. vector_loop_inc
  //  6. A block we can jump to from the RVV loop when a mismatch is found i.e.
  //  vector_loop_exit
  //  7. The first block of the scalar loop itself, containing PHIs , loads
  //  and cmp.
  //  8. A scalar loop increment block to increment the PHIs and go back
  //  around the loop.

  BasicBlock *MinItCheckBlock = BasicBlock::Create(
      Ctx, "mismatch_min_it_check", EndBlock->getParent(), EndBlock);

  // This DTU update is actually the only one we need to cover all control flow
  // changes made in this function. Because the current DTU algorithm
  // recaculates the whole sub-tree between a deleted edge. And the edge between
  // Preheader and EndBlock happens to enclose all the blocks we inserted
  // in this function.
  DTU.applyUpdates({{DominatorTree::Insert, Preheader, MinItCheckBlock},
                    {DominatorTree::Delete, Preheader, EndBlock}});

  // Update the terminator added by SplitBlock to branch to the first block
  Preheader->getTerminator()->setSuccessor(0, MinItCheckBlock);

  BasicBlock *MemCheckBlock = BasicBlock::Create(
      Ctx, "mismatch_mem_check", EndBlock->getParent(), EndBlock);

  BasicBlock *RVVLoopPreheaderBlock = BasicBlock::Create(
      Ctx, "mismatch_vector_loop_preheader", EndBlock->getParent(), EndBlock);

  BasicBlock *RVVLoopStartBlock = BasicBlock::Create(
      Ctx, "mismatch_vector_loop", EndBlock->getParent(), EndBlock);

  BasicBlock *RVVLoopIncBlock = BasicBlock::Create(
      Ctx, "mismatch_vector_loop_inc", EndBlock->getParent(), EndBlock);

  BasicBlock *RVVLoopMismatchBlock = BasicBlock::Create(
      Ctx, "mismatch_vector_loop_found", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopPreHeaderBlock = BasicBlock::Create(
      Ctx, "mismatch_loop_pre", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopStartBlock =
      BasicBlock::Create(Ctx, "mismatch_loop", EndBlock->getParent(), EndBlock);

  BasicBlock *LoopIncBlock = BasicBlock::Create(
      Ctx, "mismatch_loop_inc", EndBlock->getParent(), EndBlock);

  // Update LoopInfo with the new RVV & scalar loops.
  auto RVVLoop = LI.AllocateLoop();
  auto ScalarLoop = LI.AllocateLoop();
  if (CurLoop->getParentLoop()) {
    CurLoop->getParentLoop()->addChildLoop(RVVLoop);
    CurLoop->getParentLoop()->addChildLoop(ScalarLoop);

    CurLoop->getParentLoop()->addBasicBlockToLoop(MinItCheckBlock, LI);
    CurLoop->getParentLoop()->addBasicBlockToLoop(MemCheckBlock, LI);
    CurLoop->getParentLoop()->addBasicBlockToLoop(RVVLoopPreheaderBlock, LI);
    CurLoop->getParentLoop()->addBasicBlockToLoop(RVVLoopMismatchBlock, LI);
    CurLoop->getParentLoop()->addBasicBlockToLoop(LoopPreHeaderBlock, LI);
  } else {
    LI.addTopLevelLoop(RVVLoop);
    LI.addTopLevelLoop(ScalarLoop);
  }

  // Add the new basic blocks to their associated loops.
  RVVLoop->addBasicBlockToLoop(RVVLoopStartBlock, LI);
  RVVLoop->addBasicBlockToLoop(RVVLoopIncBlock, LI);

  ScalarLoop->addBasicBlockToLoop(LoopStartBlock, LI);
  ScalarLoop->addBasicBlockToLoop(LoopIncBlock, LI);

  // Set up some types and constants that we intend to reuse.
  Type *I64Type = Builder.getInt64Ty();
  Type *I32Type = Builder.getInt32Ty();

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

  // For each of the arrays, check the start/end addresses are on the same
  // page.
  Builder.SetInsertPoint(MemCheckBlock);

  // For each start address calculate the offset into the min architecturally
  // allowed page size (4096). Then determine how many bytes there are left on
  // the page and see if this is >= MaxLen.
  Value *LhsStartPage = Builder.CreateLShr(
      Builder.CreatePtrToInt(Builder.CreateGEP(LoadType, PtrA, ExtStart),
                             I64Type),
      uint64_t(12));
  Value *LhsEndPage = Builder.CreateLShr(
      Builder.CreatePtrToInt(Builder.CreateGEP(LoadType, PtrA, ExtEnd),
                             I64Type),
      uint64_t(12));
  Value *RhsStartPage = Builder.CreateLShr(
      Builder.CreatePtrToInt(Builder.CreateGEP(LoadType, PtrB, ExtStart),
                             I64Type),
      uint64_t(12));
  Value *RhsEndPage = Builder.CreateLShr(
      Builder.CreatePtrToInt(Builder.CreateGEP(LoadType, PtrB, ExtEnd),
                             I64Type),
      uint64_t(12));
  Value *LhsPageCmp = Builder.CreateICmpNE(LhsStartPage, LhsEndPage);
  Value *RhsPageCmp = Builder.CreateICmpNE(RhsStartPage, RhsEndPage);

  BranchInst *CombinedPageCmpCmpBr =
      BranchInst::Create(LoopPreHeaderBlock, RVVLoopPreheaderBlock,
                         Builder.CreateOr(LhsPageCmp, RhsPageCmp));
  CombinedPageCmpCmpBr->setMetadata(
      LLVMContext::MD_prof, MDBuilder(CombinedPageCmpCmpBr->getContext())
                                .createBranchWeights(10, 90));
  Builder.Insert(CombinedPageCmpCmpBr);

  // Set up the RVV loop preheader, i.e. calculate initial loop predicate,
  // zero-extend MaxLen to 64-bits, determine the number of vector elements
  // processed in each iteration, etc.
  Builder.SetInsertPoint(RVVLoopPreheaderBlock);

  // At this point we know two things must be true:
  //  1. Start <= End
  //  2. ExtMaxLen <= 4096 due to the page checks.
  // Therefore, we know that we can use a 64-bit induction variable that
  // starts from 0 -> ExtMaxLen and it will not overflow.
  auto *JumpToRVVLoop = BranchInst::Create(RVVLoopStartBlock);
  Builder.Insert(JumpToRVVLoop);

  // Set up the first RVV loop block by creating the PHIs, doing the vector
  // loads and comparing the vectors.
  Builder.SetInsertPoint(RVVLoopStartBlock);
  auto *RVVIndexPhi = Builder.CreatePHI(I64Type, 2, "mismatch_vector_index");
  RVVIndexPhi->addIncoming(ExtStart, RVVLoopPreheaderBlock);

  // Calculate AVL by subtracting the vector loop index from the trip count
  Value *AVL = Builder.CreateSub(ExtEnd, RVVIndexPhi, "avl", /*HasNUW=*/true,
                                 /*HasNSW=*/true);

  VectorType *RVVLoadType = getBestVectorTypeForLoopIdiom(Builder.getContext());
  auto *VF = ConstantInt::get(
      I32Type, RVVLoadType->getElementCount().getKnownMinValue());
  auto *IsScalable = ConstantInt::getBool(
      Builder.getContext(), RVVLoadType->getElementCount().isScalable());

  Value *RVL =
      Builder.CreateIntrinsic(Intrinsic::experimental_get_vector_length,
                              {I64Type}, {AVL, VF, IsScalable});
  Value *GepOffset = RVVIndexPhi;

  Value *RVVLhsGep = Builder.CreateGEP(LoadType, PtrA, GepOffset);
  if (GEPA->isInBounds())
    cast<GetElementPtrInst>(RVVLhsGep)->setIsInBounds(true);
  VectorType *TrueMaskTy =
      VectorType::get(Builder.getInt1Ty(), RVVLoadType->getElementCount());
  Value *AllTrueMask = Constant::getAllOnesValue(TrueMaskTy);
  Value *RVVLhsLoad = Builder.CreateIntrinsic(
      Intrinsic::vp_load, {RVVLoadType, RVVLhsGep->getType()},
      {RVVLhsGep, AllTrueMask, RVL}, nullptr, "lhs.load");

  Value *RVVRhsGep = Builder.CreateGEP(LoadType, PtrB, GepOffset);
  if (GEPB->isInBounds())
    cast<GetElementPtrInst>(RVVRhsGep)->setIsInBounds(true);
  Value *RVVRhsLoad = Builder.CreateIntrinsic(
      Intrinsic::vp_load, {RVVLoadType, RVVLhsGep->getType()},
      {RVVRhsGep, AllTrueMask, RVL}, nullptr, "rhs.load");

  StringRef PredicateStr = CmpInst::getPredicateName(CmpInst::ICMP_NE);
  auto *PredicateMDS = MDString::get(RVVLhsLoad->getContext(), PredicateStr);
  Value *Pred = MetadataAsValue::get(RVVLhsLoad->getContext(), PredicateMDS);
  Value *RVVMatchCmp =
      Builder.CreateIntrinsic(Intrinsic::vp_icmp, {RVVLhsLoad->getType()},
                              {RVVLhsLoad, RVVRhsLoad, Pred, AllTrueMask, RVL},
                              nullptr, "mismatch.cmp");
  Value *CTZ = Builder.CreateIntrinsic(
      Intrinsic::vp_cttz_elts, {ResType, RVVMatchCmp->getType()},
      {RVVMatchCmp, /*ZeroIsPoison=*/Builder.getInt1(true), AllTrueMask, RVL});
  // RISC-V refines/lowers the poison returned by cttz.elts to -1.
  Value *MismatchFound =
      Builder.CreateICmpSGE(CTZ, ConstantInt::get(ResType, 0));
  auto *RVVEarlyExit =
      BranchInst::Create(RVVLoopMismatchBlock, RVVLoopIncBlock, MismatchFound);
  Builder.Insert(RVVEarlyExit);

  // Increment the index counter and calculate the predicate for the next
  // iteration of the loop. We branch back to the start of the loop if there
  // is at least one active lane.
  Builder.SetInsertPoint(RVVLoopIncBlock);
  Value *RVL64 = Builder.CreateZExt(RVL, I64Type);
  Value *NewRVVIndexPhi = Builder.CreateAdd(RVVIndexPhi, RVL64, "",
                                            /*HasNUW=*/true, /*HasNSW=*/true);
  RVVIndexPhi->addIncoming(NewRVVIndexPhi, RVVLoopIncBlock);
  Value *ExitCond = Builder.CreateICmpNE(NewRVVIndexPhi, ExtEnd);
  auto *RVVLoopBranchBack =
      BranchInst::Create(RVVLoopStartBlock, EndBlock, ExitCond);
  Builder.Insert(RVVLoopBranchBack);

  // If we found a mismatch then we need to calculate which lane in the vector
  // had a mismatch and add that on to the current loop index.
  Builder.SetInsertPoint(RVVLoopMismatchBlock);

  // Add LCSSA phis for CTZ and RVVIndexPhi.
  auto *CTZLCSSAPhi = Builder.CreatePHI(CTZ->getType(), 1, "ctz");
  CTZLCSSAPhi->addIncoming(CTZ, RVVLoopStartBlock);
  auto *RVVIndexLCSSAPhi =
      Builder.CreatePHI(RVVIndexPhi->getType(), 1, "mismatch_vector_index");
  RVVIndexLCSSAPhi->addIncoming(RVVIndexPhi, RVVLoopStartBlock);

  Value *CTZI64 = Builder.CreateZExt(CTZLCSSAPhi, I64Type);
  Value *RVVLoopRes64 = Builder.CreateAdd(RVVIndexLCSSAPhi, CTZI64, "",
                                          /*HasNUW=*/true, /*HasNSW=*/true);
  Value *RVVLoopRes = Builder.CreateTrunc(RVVLoopRes64, ResType);

  Builder.Insert(BranchInst::Create(EndBlock));

  // Generate code for scalar loop.
  Builder.SetInsertPoint(LoopPreHeaderBlock);
  auto *StartIndexPhi = Builder.CreatePHI(ResType, 2, "mismatch_start_index");
  StartIndexPhi->addIncoming(Start, MemCheckBlock);
  StartIndexPhi->addIncoming(Start, MinItCheckBlock);
  Builder.Insert(BranchInst::Create(LoopStartBlock));

  Builder.SetInsertPoint(LoopStartBlock);
  auto *IndexPhi = Builder.CreatePHI(ResType, 2, "mismatch_index");
  IndexPhi->addIncoming(StartIndexPhi, LoopPreHeaderBlock);

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
  auto *MatchCmpBr = BranchInst::Create(LoopIncBlock, EndBlock, MatchCmp);
  Builder.Insert(MatchCmpBr);
  // Have we reached the maximum permitted length for the loop?
  Builder.SetInsertPoint(LoopIncBlock);
  Value *PhiInc = Builder.CreateAdd(IndexPhi, ConstantInt::get(ResType, 1), "",
                                    /*HasNUW=*/Index->hasNoUnsignedWrap(),
                                    /*HasNSW=*/Index->hasNoSignedWrap());
  IndexPhi->addIncoming(PhiInc, LoopIncBlock);
  Value *IVCmp = Builder.CreateICmpEQ(IndexPhi, MaxLen);
  auto *IVCmpBr = BranchInst::Create(EndBlock, LoopStartBlock, IVCmp);
  Builder.Insert(IVCmpBr);

  // In the end block we need to insert a PHI node to deal with three cases:
  //  1. The length of the loop was zero, hence we jumped straight from
  //     MinItCheckBlock.
  //  2. We didn't find a mismatch in the scalar loop, so we should return
  //  MaxLen.
  //  3. We exitted the scalar loop early due to a mismatch and need to return
  //  the index that we found.
  //  4. We didn't find a mismatch in the RVV loop, so we should return
  //  MaxLen.
  //  5. We exitted the RVV loop early due to a mismatch and need to return
  //  the index that we found.
  Builder.SetInsertPoint(EndBlock, EndBlock->getFirstInsertionPt());
  auto *ResPhi = Builder.CreatePHI(ResType, 4, "mismatch_result");
  ResPhi->addIncoming(MaxLen, LoopIncBlock);
  ResPhi->addIncoming(IndexPhi, LoopStartBlock);
  ResPhi->addIncoming(MaxLen, RVVLoopIncBlock);
  ResPhi->addIncoming(RVVLoopRes, RVVLoopMismatchBlock);

  return Builder.CreateTrunc(ResPhi, ResType);
}

void RISCVLoopIdiomRecognize::transformByteCompare(
    GetElementPtrInst *GEPA, GetElementPtrInst *GEPB, PHINode *IndPhi,
    Value *MaxLen, Instruction *Index, Value *Start, bool IncIdx,
    BasicBlock *FoundBB, BasicBlock *EndBB) {

  // Insert the byte compare intrinsic at the end of the preheader block
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  BasicBlock *Header = CurLoop->getHeader();
  auto *PHBranch = cast<BranchInst>(Preheader->getTerminator());
  IRBuilder<> Builder(PHBranch);
  Builder.SetCurrentDebugLocation(PHBranch->getDebugLoc());

  // Increment the pointer if this was done before the loads in the loop.
  if (IncIdx)
    Start = Builder.CreateAdd(Start, ConstantInt::get(Start->getType(), 1));

  Value *ByteCmpRes =
      expandFindMismatch(Builder, GEPA, GEPB, Index, Start, MaxLen);

  // Replaces uses of index with intrinsic.
  assert(IndPhi->hasOneUse() && "Index phi node has more than one use!");
  Index->replaceAllUsesWith(ByteCmpRes);

  // If no mismatch was found, we can jump to the end block. Create a
  // new basic block for the compare instruction.
  auto *CmpBB = BasicBlock::Create(Preheader->getContext(), "byte.compare",
                                   Preheader->getParent());
  CmpBB->moveBefore(EndBB);

  // Replace the branch in the preheader with an always-true conditional branch.
  // This ensures there is still a reference to the original loop.
  Value *BrCnd = Builder.CreateICmpEQ(ConstantInt::get(Start->getType(), 1),
                                      ConstantInt::get(Start->getType(), 1));
  Builder.CreateCondBr(BrCnd, CmpBB, Header);
  PHBranch->eraseFromParent();

  // Create the branch to either the end or found block depending on the value
  // returned by the intrinsic.
  Builder.SetInsertPoint(CmpBB);
  Value *FoundCmp = Builder.CreateICmpEQ(ByteCmpRes, MaxLen);
  Builder.CreateCondBr(FoundCmp, EndBB, FoundBB);

  auto FixSuccessorPhis = [&](BasicBlock *SuccBB) {
    for (PHINode &PN : SuccBB->phis()) {
      // At this point we've already replaced all uses of the result from the
      // loop with ByteCmp. Look through the incoming values to find ByteCmp,
      // meaning this is a Phi collecting the results of the byte compare.
      bool ResPhi =
          any_of(PN.incoming_values(), [=](Value *Op) { return Op == CmpBB; });

      // If any of the incoming values were ByteCmp, we need to also add
      // it as an incoming value from CmpBB.
      if (ResPhi) {
        PN.addIncoming(ByteCmpRes, CmpBB);
      } else {
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
  FixSuccessorPhis(EndBB);
  FixSuccessorPhis(FoundBB);

  // The new CmpBB block isn't part of the loop, but will need to be added to
  // the outer loop if there is one.
  if (!CurLoop->isOutermost())
    CurLoop->getParentLoop()->addBasicBlockToLoop(CmpBB, LI);

  // Update the dominator tree with the new block.
  DT.addNewBlock(CmpBB, Preheader);
}
