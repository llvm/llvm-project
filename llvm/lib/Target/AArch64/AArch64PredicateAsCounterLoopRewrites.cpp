//===- AArch64PredicateAsCounterLoopRewrites.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites IR for loop-carried wide masks that can be represented as
// predicate-as-counter values. This applies when the mask is used by load/store
// operations that can be mapped to multi-vector instructions (with +sve2p1).
//
// For example, a loop like:
//
//   entry:
//     %step = vscale x 64
//     %mask.entry = @get.active.lane.mask(0, %n)
//
//   loop:
//     %iv   = phi i64 [0, entry], [%iv.next, loop]
//     %mask = phi <vscale x 64 x i1> [%mask.entry, entry],
//                                    [%mask.next, loop]
//
//     %load = load <vscale x 64 x i8> %src[%iv], %mask
//     store <vscale x 64 x i8> %load, %dst[%iv], %mask
//
//     %iv.next   = %iv + %step
//     %mask.next = @get.active.lane.mask(%iv.next, %n)
//     br first.active(%mask.next), loop, exit
//
// Could be rewritten to:
//
//   entry:
//     %step = vscale x 64
//     %mask.entry = @whilelo.c8(0, %n, VLx4)
//
//   loop:
//     %iv   = phi i64 [0, entry], [%iv.next, loop]
//     %mask = phi target("aarch64.svcount") [%mask.entry, entry],
//                                           [%mask.next, loop]
//
//     %load = @ld1.pn.x4 <4 x <vscale x 16 x i8>> %src[%iv], %mask
//     @st1.pn.x4 <4 x <vscale x 16 x i8>> %load, %dest[%iv], %mask
//
//     %iv.next   = %iv + %step
//     %mask.next = @whilelo.c8(%iv.next, %n, VLx4)
//     br first.active(@pext(%mask.next, 0)), loop, exit
//
// This replaces the `get.active.lane.mask` intrinsics with AArch64
// predicate-as-counter `whilelo` intrinsics and updates the mask phi to use the
// `aarch64.svcount` target type. Within the loop, load/store users are mapped
// to multi-vector load/store intrinsics where possible. Users that cannot be
// mapped to multi-vector instructions materialize vector masks using the `pext`
// intrinsic (which extracts vector predicates from a predicate-as-counter).
//
// This pass may be a temporary solution that is removed if we gain support
// for target-specific VPlan transforms in the loop vectorizer.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "aarch64-predicate-as-counter-loop-rewrites"
namespace {

STATISTIC(LoopsRewritten, "Number of loops rewritten");

struct MaskRewriteCandidate {
  /// The preheader block for the loop.
  BasicBlock *Preheader = nullptr;
  /// The latch block for the loop.
  BasicBlock *Latch = nullptr;
  /// The mask phi node (used by masked operations within the loop).
  PHINode *MaskPhi = nullptr;
  /// The initial value for the mask (incoming value from the preheader).
  IntrinsicInst *StartMask = nullptr;
  /// The updated value for the mask (incoming value from the loop latch).
  IntrinsicInst *NextMask = nullptr;
  /// The multi-vector scale for the predicate-as-counter (2 or 4).
  unsigned VectorScale = 0;
  /// The element size (in bits) for the predicate-as-counter.
  unsigned ElementSizeInBits = 0;
};

static void logLoopBailout(const Loop &L, const Twine &Reason) {
  LLVM_DEBUG({
    dbgs() << "PAC loop rewrite: skipping loop with header ";
    L.getHeader()->printAsOperand(dbgs(), /*PrintType=*/false);
    dbgs() << ": " << Reason << "\n";
  });
}

static void logMatchFailure(const PHINode &Phi, const Twine &Reason) {
  LLVM_DEBUG({
    dbgs() << "PAC loop rewrite: failed to match mask phi ";
    Phi.printAsOperand(dbgs(), /*PrintType=*/false);
    dbgs() << ": " << Reason << "\n";
  });
}

/// Returns the scalar size in bits for a type according to the data layout.
static unsigned getScalarSizeInBits(const DataLayout &DL, Type *Ty) {
  return DL.getTypeSizeInBits(Ty->getScalarType()).getFixedValue();
}

/// Returns the SVE element count for \p ElementSizeInBits.
static ElementCount getSVEElementCount(unsigned ElementSizeInBits) {
  return ElementCount::getScalable(AArch64::SVEBitsPerBlock /
                                   ElementSizeInBits);
}

/// Returns the most common scalar access size of masked loads/stores in loop
/// \p L where \p MaskPhi is used as the mask. Ties are broken in favor of
/// larger access sizes.
static std::optional<unsigned>
getMostCommonMaskedMemAccessSizeInBits(const Loop &L, PHINode &MaskPhi) {
  const DataLayout &DL = MaskPhi.getModule()->getDataLayout();
  DenseMap<unsigned, unsigned> AccessSizeCounts;
  std::optional<unsigned> BestAccessSizeInBits;
  unsigned BestAccessSizeCount = 0;

  for (User *U : MaskPhi.users()) {
    auto *II = dyn_cast<IntrinsicInst>(U);
    if (!II || !L.contains(II))
      continue;

    Intrinsic::ID IID = II->getIntrinsicID();
    if (IID != Intrinsic::masked_load && IID != Intrinsic::masked_store)
      continue;

    unsigned MaskOpIdx = IID == Intrinsic::masked_load ? 1 : 2;
    if (II->getArgOperand(MaskOpIdx) != &MaskPhi)
      continue;

    unsigned AccessSizeInBits = getScalarSizeInBits(DL, II->getAccessType());
    unsigned AccessSizeCount = ++AccessSizeCounts[AccessSizeInBits];

    if (!BestAccessSizeInBits || AccessSizeCount > BestAccessSizeCount ||
        (AccessSizeCount == BestAccessSizeCount &&
         AccessSizeInBits > *BestAccessSizeInBits)) {
      BestAccessSizeInBits = AccessSizeInBits;
      BestAccessSizeCount = AccessSizeCount;
    }
  }

  return BestAccessSizeInBits;
}

/// Returns the predicate-as-counter whilelo intrinsic ID for
/// \p ElementSizeInBits.
static Intrinsic::ID getWhileLOIntrinsic(unsigned ElementSizeInBits) {
  switch (ElementSizeInBits) {
  case 8:
    return Intrinsic::aarch64_sve_whilelo_c8;
  case 16:
    return Intrinsic::aarch64_sve_whilelo_c16;
  case 32:
    return Intrinsic::aarch64_sve_whilelo_c32;
  case 64:
    return Intrinsic::aarch64_sve_whilelo_c64;
  default:
    llvm_unreachable("unsupported predicate-as-counter element size");
  }
}

/// Expands the predicate-as-counter mask \p Count into a wide vector mask.
/// Masks are extracted from the counter using the paired pext intrinsics,
/// then concatenated to form the wide mask value.
static Value *buildWideMask(IRBuilder<> &Builder, const MaskRewriteCandidate &C,
                            Value *Count) {
  ElementCount LegalEC = getSVEElementCount(C.ElementSizeInBits);
  Module *M = Builder.GetInsertBlock()->getModule();
  Type *LegalMaskTy = VectorType::get(Builder.getInt1Ty(), LegalEC);
  FunctionCallee PExtX2 = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sve_pext_x2, {LegalMaskTy});

  Value *WideMask = PoisonValue::get(C.MaskPhi->getType());
  for (unsigned PairOffset = 0; PairOffset != C.VectorScale / 2; ++PairOffset) {
    auto *Pair = Builder.CreateCall(
        PExtX2, {Count, Builder.getInt32(PairOffset)}, "pac.pext.pair");
    for (unsigned SliceInPair = 0; SliceInPair != 2; ++SliceInPair) {
      Value *Part = Builder.CreateExtractValue(Pair, SliceInPair, "pac.pext");
      unsigned Slice = PairOffset * 2 + SliceInPair;
      WideMask = Builder.CreateInsertVector(
          C.MaskPhi->getType(), WideMask, Part,
          Slice * LegalEC.getKnownMinValue(), "pac.mask");
    }
  }

  return WideMask;
}

/// Creates a predicate-as-counter whilelo for \p ElementSizeInBits between
/// \p Start and \p End with multi-vector \p VectorScale (2 or 4).
static Value *createWhileLO(IRBuilder<> &Builder, unsigned ElementSizeInBits,
                            Value *Start, Value *End, unsigned VectorScale) {
  if (Start->getType()->getIntegerBitWidth() < 64) {
    Start = Builder.CreateZExt(Start, Builder.getInt64Ty());
    End = Builder.CreateZExt(End, Builder.getInt64Ty());
  }
  Module *M = Builder.GetInsertBlock()->getModule();
  auto ID = getWhileLOIntrinsic(ElementSizeInBits);
  FunctionCallee WhileLO = Intrinsic::getOrInsertDeclaration(M, ID);
  return Builder.CreateCall(
      WhileLO, {Start, End, Builder.getInt32(VectorScale)}, "pac.mask");
}

/// Returns the multi-vector load/store intrinsic ID for \p VectorScale.
static Intrinsic::ID getPNLoadStoreIntrinsic(unsigned VectorScale,
                                             bool IsLoad) {
  if (VectorScale == 2)
    return IsLoad ? Intrinsic::aarch64_sve_ld1_pn_x2
                  : Intrinsic::aarch64_sve_st1_pn_x2;
  if (VectorScale == 4)
    return IsLoad ? Intrinsic::aarch64_sve_ld1_pn_x4
                  : Intrinsic::aarch64_sve_st1_pn_x4;
  llvm_unreachable("unsupported predicate-as-counter scale");
}

/// If \p UserI is a masked-load user of the original loop mask, rewrite it to
/// a masked multi-vector load (using the predicate-as-counter) if the load
/// access size matches the predicate-as-counter element size.
static bool tryRewriteMaskedLoadUser(Instruction &UserI,
                                     const MaskRewriteCandidate &C,
                                     Value *Count) {
  auto *II = dyn_cast<IntrinsicInst>(&UserI);
  if (!II || II->getIntrinsicID() != Intrinsic::masked_load)
    return false;

  unsigned ElementSizeInBits =
      getScalarSizeInBits(II->getModule()->getDataLayout(), II->getType());
  if (ElementSizeInBits != C.ElementSizeInBits)
    return false;

  if (!isa<PoisonValue, UndefValue>(II->getArgOperand(2)))
    return false;

  IRBuilder<> Builder(II);
  Builder.SetCurrentDebugLocation(II->getDebugLoc());

  ElementCount LegalEC = getSVEElementCount(C.ElementSizeInBits);
  Type *ScalarType = II->getType()->getScalarType();
  auto *WideDataTy = VectorType::get(ScalarType, LegalEC * C.VectorScale);
  auto *LegalDataTy = VectorType::get(ScalarType, LegalEC);

  Module *M = II->getModule();
  FunctionCallee LD1 = Intrinsic::getOrInsertDeclaration(
      M, getPNLoadStoreIntrinsic(C.VectorScale, /*IsLoad=*/true),
      {LegalDataTy, II->getArgOperand(0)->getType()});
  auto *PNLoad =
      Builder.CreateCall(LD1, {Count, II->getArgOperand(0)}, "pac.ld1");

  // Copy pointer parameter attributes.
  for (Attribute ParamAttr : II->getParamAttributes(0))
    PNLoad->addParamAttr(1, ParamAttr);

  // Concatenate all results into the original wide value.
  Value *WideData = PoisonValue::get(WideDataTy);
  for (unsigned Slice = 0; Slice != C.VectorScale; ++Slice) {
    Value *Part = Builder.CreateExtractValue(PNLoad, Slice, "pac.data");
    WideData = Builder.CreateInsertVector(WideDataTy, WideData, Part,
                                          Slice * LegalEC.getKnownMinValue(),
                                          "pac.vec");
  }

  II->replaceAllUsesWith(WideData);
  II->eraseFromParent();
  return true;
}

/// If \p UserI is a masked-store user of the original loop mask, rewrite it to
/// a masked multi-vector store (using the predicate-as-counter) if the store
/// access size matches the predicate-as-counter element size.
static bool tryRewriteMaskedStoreUser(Instruction &UserI,
                                      const MaskRewriteCandidate &C,
                                      Value *Count) {
  auto *II = dyn_cast<IntrinsicInst>(&UserI);
  if (!II || II->getIntrinsicID() != Intrinsic::masked_store)
    return false;

  unsigned ElementSizeInBits = getScalarSizeInBits(
      II->getModule()->getDataLayout(), II->getArgOperand(0)->getType());
  if (ElementSizeInBits != C.ElementSizeInBits)
    return false;

  IRBuilder<> Builder(II);
  Builder.SetCurrentDebugLocation(II->getDebugLoc());

  Type *ScalarType = II->getArgOperand(0)->getType()->getScalarType();
  ElementCount LegalEC = getSVEElementCount(C.ElementSizeInBits);
  auto *LegalDataTy = VectorType::get(ScalarType, LegalEC);

  SmallVector<Value *, 6> StoreArgs;
  for (unsigned Slice = 0; Slice != C.VectorScale; ++Slice)
    StoreArgs.push_back(Builder.CreateExtractVector(
        LegalDataTy, II->getArgOperand(0), Slice * LegalEC.getKnownMinValue(),
        "pac.data"));
  StoreArgs.push_back(Count);
  StoreArgs.push_back(II->getArgOperand(1));

  Module *M = II->getModule();
  FunctionCallee ST1 = Intrinsic::getOrInsertDeclaration(
      M, getPNLoadStoreIntrinsic(C.VectorScale, /*IsLoad=*/false),
      {LegalDataTy, II->getArgOperand(1)->getType()});
  auto *PNStore = Builder.CreateCall(ST1, StoreArgs);

  // Copy pointer parameter attributes.
  for (Attribute ParamAttr : II->getParamAttributes(1))
    PNStore->addParamAttr(C.VectorScale + 1, ParamAttr);

  II->eraseFromParent();
  return true;
}

/// If \p UserI is an extractelement use of the original loop mask, attempt to
/// rewrite it to `extractelement(pext(count, 0), idx)` if the extract index is
/// known to be within the first mask section (which means pext index = 0). If
/// the user of the extract is a branch and the source of the mask is a whilelo,
/// this form can be optimized into checking the status flags.
static bool tryRewriteExtractElement(Instruction &UserI,
                                     const MaskRewriteCandidate &C,
                                     Value *Count) {
  auto *EEI = dyn_cast<ExtractElementInst>(&UserI);
  if (!EEI)
    return false;

  ElementCount LegalEC = getSVEElementCount(C.ElementSizeInBits);
  auto *Idx = dyn_cast<ConstantInt>(EEI->getIndexOperand());
  if (!Idx || Idx->getValue().uge(LegalEC.getKnownMinValue()))
    return false;

  IRBuilder<> Builder(EEI);
  Builder.SetCurrentDebugLocation(EEI->getDebugLoc());

  Module *M = EEI->getModule();
  FunctionCallee PExt = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sve_pext,
      {VectorType::get(Builder.getInt1Ty(), LegalEC)});
  auto *ExtractMask =
      Builder.CreateCall(PExt, {Count, Builder.getInt32(0)}, "pac.pext");

  Value *Extracted = Builder.CreateExtractElement(
      ExtractMask, EEI->getIndexOperand(), EEI->getName() + ".pac");

  EEI->replaceAllUsesWith(Extracted);
  EEI->eraseFromParent();
  return true;
}

class AArch64PredicateAsCounterLoopRewrites : public LoopPass {
public:
  static char ID;

  AArch64PredicateAsCounterLoopRewrites() : LoopPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    // Require loop simplify to ensure loops have a preheader.
    AU.addRequiredID(LoopSimplifyID);
    AU.addPreservedID(LoopSimplifyID);
    AU.setPreservesCFG();
  }

  bool runOnLoop(Loop *L, LPPassManager &) override;

private:
  std::optional<MaskRewriteCandidate> matchMaskPhi(Loop &L, PHINode &Phi) const;
  bool rewriteCandidate(const MaskRewriteCandidate &C, Loop &L) const;
};

} // end anonymous namespace

char AArch64PredicateAsCounterLoopRewrites::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64PredicateAsCounterLoopRewrites, DEBUG_TYPE,
                      "AArch64 Predicate As Counter Loop Rewrites", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(AArch64PredicateAsCounterLoopRewrites, DEBUG_TYPE,
                    "AArch64 Predicate As Counter Loop Rewrites", false, false)

Pass *llvm::createAArch64PredicateAsCounterLoopRewritesPass() {
  return new AArch64PredicateAsCounterLoopRewrites();
}

bool AArch64PredicateAsCounterLoopRewrites::runOnLoop(Loop *L,
                                                      LPPassManager &) {
  if (skipLoop(L)) {
    logLoopBailout(*L, "skipLoop requested the loop to be skipped");
    return false;
  }

  Function &F = *L->getHeader()->getParent();
  auto &TPC = getAnalysis<TargetPassConfig>();
  const AArch64Subtarget *ST =
      TPC.getTM<AArch64TargetMachine>().getSubtargetImpl(F);
  if (!ST->isSVEorStreamingSVEAvailable()) {
    logLoopBailout(*L, "SVE or streaming SVE is unavailable");
    return false;
  }
  if (!ST->hasSVE2p1() && !(ST->hasSME2() && ST->isStreaming())) {
    logLoopBailout(*L, "neither SVE2.1 nor SME2 is available");
    return false;
  }

  bool Changed = false;
  BasicBlock *Header = L->getHeader();
  for (PHINode &Phi : make_early_inc_range(Header->phis())) {
    auto Candidate = matchMaskPhi(*L, Phi);
    if (Candidate)
      Changed |= rewriteCandidate(*Candidate, *L);
  }

  if (Changed)
    ++LoopsRewritten;

  return Changed;
}

static IntrinsicInst *getGetActiveLaneMask(Value *V) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  return II && II->getIntrinsicID() == Intrinsic::get_active_lane_mask
             ? II
             : nullptr;
}

std::optional<MaskRewriteCandidate>
AArch64PredicateAsCounterLoopRewrites::matchMaskPhi(Loop &L,
                                                    PHINode &Phi) const {
  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *Latch = L.getLoopLatch();
  if (!Preheader) {
    logMatchFailure(Phi, "loop has no preheader");
    return std::nullopt;
  }
  if (!Latch) {
    logMatchFailure(Phi, "loop has no latch");
    return std::nullopt;
  }

  auto *PhiTy = dyn_cast<ScalableVectorType>(Phi.getType());
  if (!PhiTy || !PhiTy->getElementType()->isIntegerTy(1)) {
    logMatchFailure(Phi, "phi type is not a scalable i1 vector mask");
    return std::nullopt;
  }
  if (Phi.getNumIncomingValues() != 2) {
    logMatchFailure(Phi, Twine("phi has ")
                             .concat(Twine(Phi.getNumIncomingValues()))
                             .concat(" incoming values; expected 2"));
    return std::nullopt;
  }

  auto *StartMask =
      getGetActiveLaneMask(Phi.getIncomingValueForBlock(Preheader));
  auto *NextMask = getGetActiveLaneMask(Phi.getIncomingValueForBlock(Latch));
  if (!StartMask) {
    logMatchFailure(Phi,
                    "preheader incoming value is not get_active_lane_mask");
    return std::nullopt;
  }
  if (!NextMask) {
    logMatchFailure(Phi, "latch incoming value is not get_active_lane_mask");
    return std::nullopt;
  }

  unsigned WideMaskElements = PhiTy->getMinNumElements();
  if (!isPowerOf2_32(WideMaskElements)) {
    logMatchFailure(Phi, Twine("wide mask element count is not a power of 2: ")
                             .concat(Twine(WideMaskElements)));
    return std::nullopt;
  }

  if (StartMask->getArgOperand(0)->getType()->getIntegerBitWidth() > 64) {
    logMatchFailure(Phi, "start mask induction operand is wider than i64");
    return std::nullopt;
  }
  if (NextMask->getArgOperand(0)->getType()->getIntegerBitWidth() > 64) {
    logMatchFailure(Phi, "next mask induction operand is wider than i64");
    return std::nullopt;
  }

  std::optional<unsigned> PreferredMaskElementSizeInBits =
      getMostCommonMaskedMemAccessSizeInBits(L, Phi);
  if (!PreferredMaskElementSizeInBits) {
    logMatchFailure(Phi, "mask phi has no masked load/store users in the loop");
    return std::nullopt;
  }

  if (!is_contained({8u, 16u, 32u, 64u}, *PreferredMaskElementSizeInBits)) {
    logMatchFailure(Phi, Twine("unsupported element size in bits: ")
                             .concat(Twine(*PreferredMaskElementSizeInBits)));
    return std::nullopt;
  }

  unsigned SVEMaskElements =
      getSVEElementCount(*PreferredMaskElementSizeInBits).getKnownMinValue();
  if (WideMaskElements <= SVEMaskElements) {
    logMatchFailure(Phi, Twine("wide mask element count ")
                             .concat(Twine(WideMaskElements))
                             .concat(" is not wider than the legal mask width ")
                             .concat(Twine(SVEMaskElements)));
    return std::nullopt;
  }

  unsigned VectorScale = WideMaskElements / SVEMaskElements;
  if (VectorScale != 2 && VectorScale != 4) {
    logMatchFailure(Phi, Twine("unsupported predicate-as-counter scale: ")
                             .concat(Twine(VectorScale)));
    return std::nullopt;
  }

  return MaskRewriteCandidate{Preheader,
                              Latch,
                              &Phi,
                              StartMask,
                              NextMask,
                              VectorScale,
                              *PreferredMaskElementSizeInBits};
}

bool AArch64PredicateAsCounterLoopRewrites::rewriteCandidate(
    const MaskRewriteCandidate &C, Loop &L) const {
  IRBuilder<> Builder(C.StartMask);
  Value *NewStart =
      createWhileLO(Builder, C.ElementSizeInBits, C.StartMask->getArgOperand(0),
                    C.StartMask->getArgOperand(1), C.VectorScale);
  Builder.SetInsertPoint(C.NextMask);
  Value *NewNext =
      createWhileLO(Builder, C.ElementSizeInBits, C.NextMask->getArgOperand(0),
                    C.NextMask->getArgOperand(1), C.VectorScale);

  Builder.SetInsertPoint(C.MaskPhi);
  auto *NewPhi =
      Builder.CreatePHI(NewStart->getType(), 2, C.MaskPhi->getName() + ".pn");
  NewPhi->addIncoming(NewStart, C.Preheader);
  NewPhi->addIncoming(NewNext, C.Latch);

  auto RewriteUses = [&](Instruction *OldMask, Value *Count,
                         function_ref<bool(Use & U)> Predicate = nullptr) {
    SmallVector<Use *, 8> UsesToRewrite;
    for (Use &U : OldMask->uses()) {
      if (!Predicate || Predicate(U))
        UsesToRewrite.push_back(&U);
    }

    Value *WideMask = nullptr;
    for (Use *U : UsesToRewrite) {
      auto *UserI = cast<Instruction>(U->getUser());
      if (tryRewriteMaskedLoadUser(*UserI, C, Count) ||
          tryRewriteMaskedStoreUser(*UserI, C, Count) ||
          tryRewriteExtractElement(*UserI, C, Count))
        continue;

      if (!WideMask) {
        BasicBlock::iterator InsertPt = OldMask->getIterator();
        if (isa<PHINode>(OldMask))
          InsertPt = OldMask->getParent()->getFirstNonPHIIt();

        Builder.SetInsertPoint(InsertPt);
        Builder.SetCurrentDebugLocation(OldMask->getDebugLoc());
        WideMask = buildWideMask(Builder, C, Count);
      }

      U->set(WideMask);
    }
  };

  // For the start/next mask, we only replace non-phi in-loop users. Due to CSE,
  // these masks could be used by other loops, and replacing them with
  // predicate-as-counter masks could prevent matching those loops.
  auto IsNonPhiUseInLoop = [&](Use &U) {
    auto *UseInst = cast<Instruction>(U.getUser());
    return !isa<PHINode>(UseInst) && L.contains(UseInst);
  };

  RewriteUses(C.MaskPhi, NewPhi);
  RewriteUses(C.StartMask, NewStart, IsNonPhiUseInLoop);
  RewriteUses(C.NextMask, NewNext, IsNonPhiUseInLoop);

  RecursivelyDeleteTriviallyDeadInstructions(C.MaskPhi);
  return true;
}
