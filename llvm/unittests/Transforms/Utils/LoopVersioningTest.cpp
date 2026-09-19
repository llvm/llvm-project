//===- LoopVersioningTest.cpp - Unit tests for LoopVersioning -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests exercise LoopVersioning directly for non-innermost loops, which
// the innermost-only loop-versioning pass (see runImpl in LoopVersioning.cpp)
// does not reach. They cover empty pointer checks and predicate unions,
// caller-supplied conditions, versioned and fallback identities, live-out
// repair, loop nesting, canonical form, and empty versus nonempty
// SCEVUnionPredicate instances. LoopFlatten uses the same empty-check setup
// when it versions its outer loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("LoopVersioningTest", errs());
  return Mod;
}

// AnalysisHolder builds the analyses required by LoopVersioning. Its
// LoopAccessInfoManager receives &TLI rather than LoopFlatten's null TLI.
struct AnalysisHolder {
  DominatorTree DT;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  AssumptionCache AC;
  LoopInfo LI;
  ScalarEvolution SE;
  TargetTransformInfo TTI;
  AAResults AA;
  BasicAAResult BAA;
  LoopAccessInfoManager LAIM;

  AnalysisHolder(Module &M, Function &F)
      : DT(F), TLII(M.getTargetTriple()), TLI(TLII), AC(F), LI(DT),
        SE(F, TLI, AC, DT, LI), TTI(M.getDataLayout()), AA(TLI),
        BAA(M.getDataLayout(), F, TLI, AC, &DT),
        LAIM(SE, AA, DT, LI, &TTI, &TLI, &AC) {
    AA.addAAResult(BAA);
  }
};

static Loop *getLoopByHeaderName(LoopInfo &LI, StringRef HeaderName) {
  for (Loop *L : LI.getLoopsInPreorder())
    if (L->getHeader()->getName() == HeaderName)
      return L;
  return nullptr;
}

// Return true if \p V originates in loop \p L, following the single-input
// LCSSA PHIs that formDedicatedExitBlocks inserts in the dedicated exits.
// This distinguishes versioned and fallback values without hard-coding the
// exit-block layout.
static bool tracesIntoLoop(Value *V, const Loop *L) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  if (L->contains(I))
    return true;
  if (auto *PN = dyn_cast<PHINode>(I))
    if (PN->getNumIncomingValues() == 1)
      return tracesIntoLoop(PN->getIncomingValue(0), L);
  return false;
}

// Return true if \p I has a user outside both \p L and \p Clone. Such a user
// should consume the shared-join PHI rather than a version-specific value.
static bool hasUserOutsideLoops(Instruction *I, const Loop *L,
                                const Loop *Clone) {
  for (User *U : I->users()) {
    auto *UI = cast<Instruction>(U);
    if (!L->contains(UI->getParent()) && !Clone->contains(UI->getParent()))
      return true;
  }
  return false;
}

// TopLevelIR contains a top-level, non-innermost selected outer loop
// (%outer.header) around a simple counting leaf (%inner.header). Its only
// memory access is a load whose address is affine in the outer IV and uniform
// in the inner IV, so LoopAccessInfo for the outer loop yields an empty
// predicate union. Three reductions escape to a shared exit, so
// findDefsUsedOutsideOfLoop returns exactly three live-outs. The preheader is
// the predecessor-free function entry, and the loop is in loop-simplify and
// LCSSA form.
static const char *TopLevelIR = R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @version_toplevel(ptr noalias %A, ptr noalias %R, i64 %n) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %entry ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %entry ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %entry ], [ %v.next, %outer.latch ]
  %ai = getelementptr inbounds i64, ptr %A, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %x = load i64, ptr %ai, align 8
  %p.next.j = add i64 %p.j, %x
  %u.next.j = add i64 %u.j, %x
  %v.next.j = add i64 %v.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %outer.latch, label %inner.header
outer.latch:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds i64, ptr %R, i64 1
  %r2 = getelementptr inbounds i64, ptr %R, i64 2
  store i64 %p.res, ptr %R, align 8
  store i64 %u.res, ptr %r1, align 8
  store i64 %v.res, ptr %r2, align 8
  ret void
}
)IR";

// A nested selected-outer loop: %i.header is the versioning target, itself the
// child of %k.header and the parent of the %j.header leaf. Its preheader is the
// enclosing %k.header, which has an entry and a backedge predecessor. Its
// access is affine in %i and uniform in %j, so the LoopAccessInfo predicate
// union is empty. A single partial sum escapes to %i.exit and feeds the
// enclosing accumulation.
static const char *NestedIR = R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @version_nested(ptr noalias %A, ptr noalias %R) {
entry:
  br label %k.ph
k.ph:
  br label %k.header
k.header:
  %k = phi i64 [ 0, %k.ph ], [ %k.next, %k.latch ]
  %acc.k = phi i64 [ 0, %k.ph ], [ %acc.next, %k.latch ]
  br label %i.header
i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  %s.i = phi i64 [ 0, %k.header ], [ %s.next, %i.latch ]
  %ai = getelementptr inbounds i64, ptr %A, i64 %i
  br label %j.header
j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.header ]
  %s.j = phi i64 [ %s.i, %i.header ], [ %s.next.j, %j.header ]
  %x = load i64, ptr %ai, align 8
  %s.next.j = add i64 %s.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %i.latch, label %j.header
i.latch:
  %s.next = phi i64 [ %s.next.j, %j.header ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %i.exit, label %i.header
i.exit:
  %s.lcssa = phi i64 [ %s.next, %i.latch ]
  br label %k.latch
k.latch:
  %acc.next = add i64 %acc.k, %s.lcssa
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %k.header
exit:
  %acc.res = phi i64 [ %acc.next, %k.latch ]
  store i64 %acc.res, ptr %R, align 8
  ret void
}
)IR";

// NonemptyPredicateIR contains an innermost loop for which LoopAccessAnalysis
// records a nonempty PSE predicate union from nusw assumptions on a truncating
// GEP index with a 16-bit pointer index type. The test case shows that the
// empty union accepted above is a distinguishable LAA state. LAA rejects
// non-innermost loops before adding such predicates, so this test checks the
// API state rather than a transform candidate.
static const char *NonemptyPredicateIR = R"IR(
target datalayout = "p:16:16-p3:32:32"

define void @nonempty_pred(ptr %v, i32 %N) {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.i16 = trunc i64 %iv to i16
  %gep.iv.i16 = getelementptr { i16, i16 }, ptr %v, i16 %iv.i16
  store i16 0, ptr %gep.iv.i16, align 1
  store i16 0, ptr %v, align 1
  %iv.next = add i64 %iv, 1
  %iv.i32 = trunc i64 %iv to i32
  %.not = icmp ult i32 %N, %iv.i32
  br i1 %.not, label %exit, label %loop
exit:
  ret void
}
)IR";

TEST(LoopVersioningTest, TopLevelSelectedOuterEmptyChecks) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, TopLevelIR);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("version_toplevel");
  ASSERT_NE(F, nullptr);

  AnalysisHolder H(*M, *F);
  Loop *L = getLoopByHeaderName(H.LI, "outer.header");
  ASSERT_NE(L, nullptr);
  ASSERT_FALSE(L->isInnermost());
  ASSERT_EQ(L->getParentLoop(), nullptr);
  ASSERT_TRUE(L->isLoopSimplifyForm());
  ASSERT_TRUE(L->isRecursivelyLCSSAForm(H.DT, H.LI));

  // Capture the anchors that survive versioning by pointer identity.
  BasicBlock *CheckBB = L->getLoopPreheader();
  ASSERT_NE(CheckBB, nullptr);
  ASSERT_EQ(CheckBB, &F->getEntryBlock());
  ASSERT_EQ(pred_size(CheckBB), 0u);
  BasicBlock *JoinBB = L->getExitBlock();
  ASSERT_NE(JoinBB, nullptr);

  // Model a custom failure predicate expanded before versioning. It must remain
  // in the original preheader when LoopVersioning splits that block.
  IRBuilder<> Builder(CheckBB->getTerminator());
  Value *Bad =
      Builder.CreateICmpUGT(F->getArg(2), Builder.getInt64(4), "bound.failure");
  auto *BadI = cast<Instruction>(Bad);

  const LoopAccessInfo &LAI = H.LAIM.getInfo(*L);
  // A custom-check client accepts only an empty predicate union, not merely an
  // always-true one.
  const SCEVPredicate &Pred = LAI.getPSE().getPredicate();
  EXPECT_TRUE(Pred.isAlwaysTrue());
  EXPECT_TRUE(cast<SCEVUnionPredicate>(Pred).getPredicates().empty());

  // Exactly the three escaping reductions are live-out.
  SmallVector<Instruction *, 8> DefsUsedOutside = findDefsUsedOutsideOfLoop(L);
  EXPECT_EQ(DefsUsedOutside.size(), 3u);

  // Prime SCEV for the original one-input LCSSA join PHIs.  SCEV looks through
  // a trivial LCSSA PHI to its incoming value, so a stale cache would still
  // report that (in-loop) value after versioning adds the clone predecessor.
  SmallVector<PHINode *, 4> JoinPHIs;
  SmallVector<const SCEV *, 4> PrimedSCEV;
  for (PHINode &PN : JoinBB->phis()) {
    ASSERT_EQ(PN.getNumIncomingValues(), 1u);
    JoinPHIs.push_back(&PN);
    PrimedSCEV.push_back(H.SE.getSCEV(&PN));
  }
  ASSERT_EQ(JoinPHIs.size(), 3u);

  LoopVersioning LVer(LAI, /*Checks=*/{}, L, &H.LI, &H.DT, &H.SE);
  LVer.versionLoop(DefsUsedOutside);

  // The original is the versioned loop, and the clone is the fallback.
  EXPECT_EQ(LVer.getVersionedLoop(), L);
  Loop *Clone = LVer.getNonVersionedLoop();
  ASSERT_NE(Clone, nullptr);
  EXPECT_NE(Clone, L);
  EXPECT_EQ(Clone->getParentLoop(), nullptr);

  // An empty runtime check selects the versioned loop by default. versionLoop
  // places the fallback on successor 0 and the versioned loop on successor 1.
  auto *BI = dyn_cast<CondBrInst>(CheckBB->getTerminator());
  ASSERT_NE(BI, nullptr);
  EXPECT_TRUE(match(BI->getCondition(), m_Zero()));
  EXPECT_EQ(BI->getSuccessor(0), Clone->getLoopPreheader());
  EXPECT_EQ(BI->getSuccessor(1), L->getLoopPreheader());
  ASSERT_EQ(BadI->getParent(), CheckBB);
  EXPECT_TRUE(BadI->comesBefore(BI));
  EXPECT_TRUE(H.DT.dominates(CheckBB, Clone->getLoopPreheader()));
  EXPECT_TRUE(H.DT.dominates(CheckBB, L->getLoopPreheader()));
  BI->setCondition(Bad);
  EXPECT_EQ(BI->getCondition(), Bad);

  // Dedicated exits and preserved simplify / recursive-LCSSA form for both.
  EXPECT_TRUE(L->hasDedicatedExits());
  EXPECT_TRUE(Clone->hasDedicatedExits());
  EXPECT_TRUE(L->isLoopSimplifyForm());
  EXPECT_TRUE(Clone->isLoopSimplifyForm());
  EXPECT_TRUE(L->isRecursivelyLCSSAForm(H.DT, H.LI));
  EXPECT_TRUE(Clone->isRecursivelyLCSSAForm(H.DT, H.LI));

  // The two shared-join predecessors are exactly the versioned and fallback
  // dedicated exits (each the unique exit of its version).
  BasicBlock *VersionedExit = L->getExitBlock();
  BasicBlock *FallbackExit = Clone->getExitBlock();
  ASSERT_NE(VersionedExit, nullptr);
  ASSERT_NE(FallbackExit, nullptr);
  EXPECT_NE(VersionedExit, FallbackExit);
  EXPECT_EQ(VersionedExit->getSingleSuccessor(), JoinBB);
  EXPECT_EQ(FallbackExit->getSingleSuccessor(), JoinBB);

  // The shared join holds three two-input live-out PHIs. For each one, check
  // the exact versioned and fallback incoming blocks, check that the
  // corresponding values trace into the original loop and the clone, and check
  // that it has a user outside both loops.
  unsigned NumLiveOutPHIs = 0;
  for (PHINode &PN : JoinBB->phis()) {
    ASSERT_EQ(PN.getNumIncomingValues(), 2u);
    ASSERT_NE(PN.getBasicBlockIndex(VersionedExit), -1);
    ASSERT_NE(PN.getBasicBlockIndex(FallbackExit), -1);
    EXPECT_TRUE(tracesIntoLoop(PN.getIncomingValueForBlock(VersionedExit), L));
    EXPECT_TRUE(
        tracesIntoLoop(PN.getIncomingValueForBlock(FallbackExit), Clone));
    EXPECT_TRUE(hasUserOutsideLoops(&PN, L, Clone));
    ++NumLiveOutPHIs;
  }
  EXPECT_EQ(NumLiveOutPHIs, 3u);

  // The three exit stores store the shared-join PHIs (in JoinBB), not raw
  // per-version values.
  unsigned StoresToJoinPHI = 0;
  for (Instruction &I : *JoinBB)
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      auto *ValPN = dyn_cast<PHINode>(SI->getValueOperand());
      EXPECT_TRUE(ValPN && ValPN->getParent() == JoinBB);
      ++StoresToJoinPHI;
    }
  EXPECT_EQ(StoresToJoinPHI, 3u);

  // SCEV cache repair: each primed one-input LCSSA SCEV must be recomputed to
  // the opaque two-input merge (SCEVUnknown of the PHI itself), never left
  // stale.
  for (unsigned K = 0, E = JoinPHIs.size(); K != E; ++K) {
    const SCEV *After = H.SE.getSCEV(JoinPHIs[K]);
    EXPECT_NE(After, PrimedSCEV[K]);
    ASSERT_TRUE(isa<SCEVUnknown>(After));
    EXPECT_EQ(cast<SCEVUnknown>(After)->getValue(), JoinPHIs[K]);
  }

  EXPECT_FALSE(verifyFunction(*F, &errs()));
  EXPECT_TRUE(H.DT.verify());
  H.LI.verify();
}

TEST(LoopVersioningTest, NestedSelectedOuterNoArgOverload) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, NestedIR);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("version_nested");
  ASSERT_NE(F, nullptr);

  AnalysisHolder H(*M, *F);
  Loop *L = getLoopByHeaderName(H.LI, "i.header");
  ASSERT_NE(L, nullptr);
  Loop *Parent = getLoopByHeaderName(H.LI, "k.header");
  ASSERT_NE(Parent, nullptr);
  ASSERT_FALSE(L->isInnermost());
  ASSERT_EQ(L->getParentLoop(), Parent);
  ASSERT_TRUE(L->isLoopSimplifyForm());
  ASSERT_TRUE(L->isRecursivelyLCSSAForm(H.DT, H.LI));

  BasicBlock *CheckBB = L->getLoopPreheader();
  ASSERT_NE(CheckBB, nullptr);
  ASSERT_EQ(CheckBB, Parent->getHeader());
  ASSERT_EQ(pred_size(CheckBB), 2u);
  BasicBlock *JoinBB = L->getExitBlock();
  ASSERT_NE(JoinBB, nullptr);

  const LoopAccessInfo &LAI = H.LAIM.getInfo(*L);
  const SCEVPredicate &Pred = LAI.getPSE().getPredicate();
  EXPECT_TRUE(Pred.isAlwaysTrue());
  EXPECT_TRUE(cast<SCEVUnionPredicate>(Pred).getPredicates().empty());

  // One partial sum escapes. This test exercises the no-argument overload that
  // computes the live-out set.
  EXPECT_EQ(findDefsUsedOutsideOfLoop(L).size(), 1u);

  // Prime SCEV for the sole one-input LCSSA join PHI before versioning.
  PHINode *JoinPHI = nullptr;
  unsigned PreCount = 0;
  for (PHINode &PN : JoinBB->phis()) {
    JoinPHI = &PN;
    ++PreCount;
  }
  ASSERT_EQ(PreCount, 1u);
  ASSERT_EQ(JoinPHI->getNumIncomingValues(), 1u);
  const SCEV *PrimedSCEV = H.SE.getSCEV(JoinPHI);

  LoopVersioning LVer(LAI, /*Checks=*/{}, L, &H.LI, &H.DT, &H.SE);
  LVer.versionLoop();

  EXPECT_EQ(LVer.getVersionedLoop(), L);
  Loop *Clone = LVer.getNonVersionedLoop();
  ASSERT_NE(Clone, nullptr);
  EXPECT_NE(Clone, L);
  // The clone is installed as a sibling under the same parent loop. Nested
  // versioning must not raise it to a top-level loop.
  EXPECT_EQ(Clone->getParentLoop(), Parent);

  auto *BI = dyn_cast<CondBrInst>(CheckBB->getTerminator());
  ASSERT_NE(BI, nullptr);
  EXPECT_TRUE(match(BI->getCondition(), m_Zero()));
  EXPECT_EQ(BI->getSuccessor(0), Clone->getLoopPreheader());
  EXPECT_EQ(BI->getSuccessor(1), L->getLoopPreheader());

  EXPECT_TRUE(L->hasDedicatedExits());
  EXPECT_TRUE(Clone->hasDedicatedExits());
  EXPECT_TRUE(L->isLoopSimplifyForm());
  EXPECT_TRUE(Clone->isLoopSimplifyForm());
  EXPECT_TRUE(L->isRecursivelyLCSSAForm(H.DT, H.LI));
  EXPECT_TRUE(Clone->isRecursivelyLCSSAForm(H.DT, H.LI));

  BasicBlock *VersionedExit = L->getExitBlock();
  BasicBlock *FallbackExit = Clone->getExitBlock();
  ASSERT_NE(VersionedExit, nullptr);
  ASSERT_NE(FallbackExit, nullptr);
  EXPECT_NE(VersionedExit, FallbackExit);
  EXPECT_EQ(VersionedExit->getSingleSuccessor(), JoinBB);
  EXPECT_EQ(FallbackExit->getSingleSuccessor(), JoinBB);

  // The single live-out PHI merges the exact versioned and fallback dedicated
  // exits, with values tracing into the original loop and the clone, and it
  // still feeds the enclosing accumulation (an outside user).
  unsigned NumLiveOutPHIs = 0;
  for (PHINode &PN : JoinBB->phis()) {
    ASSERT_EQ(PN.getNumIncomingValues(), 2u);
    ASSERT_NE(PN.getBasicBlockIndex(VersionedExit), -1);
    ASSERT_NE(PN.getBasicBlockIndex(FallbackExit), -1);
    EXPECT_TRUE(tracesIntoLoop(PN.getIncomingValueForBlock(VersionedExit), L));
    EXPECT_TRUE(
        tracesIntoLoop(PN.getIncomingValueForBlock(FallbackExit), Clone));
    EXPECT_TRUE(hasUserOutsideLoops(&PN, L, Clone));
    ++NumLiveOutPHIs;
  }
  EXPECT_EQ(NumLiveOutPHIs, 1u);

  // SCEV cache repair for the reused LCSSA join PHI (same pointer, now
  // two-input).
  const SCEV *After = H.SE.getSCEV(JoinPHI);
  EXPECT_NE(After, PrimedSCEV);
  ASSERT_TRUE(isa<SCEVUnknown>(After));
  EXPECT_EQ(cast<SCEVUnknown>(After)->getValue(), JoinPHI);

  EXPECT_FALSE(verifyFunction(*F, &errs()));
  EXPECT_TRUE(H.DT.verify());
  H.LI.verify();
}

// A custom-check client requires an empty PSE predicate union, which is
// strictly stronger than isAlwaysTrue(). This test shows that a nonempty union
// is a real LAA state. The next test shows why emptiness must be checked
// separately.
TEST(LoopVersioningTest, NonemptyPredicateUnionIsDistinguishable) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, NonemptyPredicateIR);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("nonempty_pred");
  ASSERT_NE(F, nullptr);

  AnalysisHolder H(*M, *F);
  Loop *L = getLoopByHeaderName(H.LI, "loop");
  ASSERT_NE(L, nullptr);

  const LoopAccessInfo &LAI = H.LAIM.getInfo(*L);
  const SCEVPredicate &Pred = LAI.getPSE().getPredicate();
  // This LAA union is both nonempty and not always true, so the accepted
  // empty state above is genuinely different. The constructed-union test
  // below separately isolates emptiness from isAlwaysTrue().
  EXPECT_FALSE(cast<SCEVUnionPredicate>(Pred).getPredicates().empty());
  EXPECT_FALSE(Pred.isAlwaysTrue());
}

// An always-true wrap predicate can remain in a nonempty union. Clients
// requiring an empty union must check getPredicates().empty(), not just
// isAlwaysTrue().
TEST(LoopVersioningTest, AlwaysTrueWrapPredicateKeepsUnionNonempty) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, TopLevelIR);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("version_toplevel");
  ASSERT_NE(F, nullptr);

  AnalysisHolder H(*M, *F);
  Loop *L = getLoopByHeaderName(H.LI, "outer.header");
  ASSERT_NE(L, nullptr);

  // A {0,+,1} recurrence over L carrying a static no-signed-wrap flag.
  Type *I64 = Type::getInt64Ty(C);
  const auto *AR = cast<SCEVAddRecExpr>(H.SE.getAddRecExpr(
      H.SE.getZero(I64), H.SE.getOne(I64), L, SCEV::FlagNSW));
  ASSERT_TRUE(AR->hasNoSignedWrap());

  // The NSSW wrap predicate over an already-NSW AddRec is always-true.
  const SCEVPredicate *Wrap =
      H.SE.getWrapPredicate(AR, SCEVWrapPredicate::IncrementNSSW);
  EXPECT_TRUE(Wrap->isAlwaysTrue());

  // SCEVUnionPredicate::add() prunes only by implication between predicates,
  // never because a predicate is always true on its own. This union therefore
  // remains nonempty while isAlwaysTrue() returns true.
  const SCEVPredicate *Preds[] = {Wrap};
  SCEVUnionPredicate Union(Preds, H.SE);
  EXPECT_TRUE(Union.isAlwaysTrue());
  EXPECT_FALSE(Union.getPredicates().empty());
  EXPECT_EQ(Union.getPredicates().size(), 1u);
}

} // namespace
