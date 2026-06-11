//===- AIRCrossBufferStoreSeparate.cpp - Re-separate device stores ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX-1 root-fix. See AIRCrossBufferStoreSeparate.h and AGX_BUGS.md.
//
// The Apple AGX JIT coalesces device stores by *address*, ignoring source
// order. When two stores to different buffer args land at the same per-thread
// element offset and are program-adjacent (one straight-line block), the JIT
// fuses them and drops warp 0's value on one of the pair. At -O0 the bug never
// fires because each masked store sits in its OWN predicated basic block:
//
//     br i1 %mask, label %store_bb, label %cont
//   store_bb:  store v, ptr ; br label %cont
//   cont:      ...
//
// so the same-offset cross-buffer stores are separated by real control flow the
// JIT does not speculate across. -O1+ flattens this into one straight-line run
// and the bug fires.
//
// This pass restores the -O0 structure: it groups the conflicting device stores
// BY their per-thread offset SSA, and hoists each offset group into its own
// freshly-created predicated block guarded by THAT group's reaching in-bounds
// mask (the `icmp slt offset, N` that already dominates the store at -O0). The
// predicate is always taken for in-bounds threads (identical semantics) but is
// a real conditional branch, so the JIT can no longer place the cross-buffer
// pair in one straight-line run -> it cannot address-coalesce them. This is
// precisely the transform that distinguishes the passing -O0 IR from the
// failing -O1 IR.
//
// Grouping per offset (rather than one shared guard over the whole conflicting
// run) means a store under a DIFFERENT predicate/offset is never pulled under
// the wrong guard: only the stores that share one offset (hence one in-bounds
// condition) move together, and non-conflicting stores stay unguarded in the
// continuation exactly as before.
//
//===----------------------------------------------------------------------===//

#include "AIRCrossBufferStoreSeparate.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "air-cross-buffer-store-separate"

namespace {

// Device (AS1) address space, per the Metal address-space map.
static constexpr unsigned kDeviceAS = metal::AS::Device;

// Walk a store-pointer expression back through pointer-preserving ops to the
// reaching base value (ideally an addrspace(1) Argument) and collect the SSA
// offset that selects the per-thread element. Returns the base; OffsetOut is
// the dominating index value (or null if a constant-only / unrecognised shape).
static Value *traceBaseAndOffset(Value *Ptr, Value *&OffsetOut) {
  OffsetOut = nullptr;
  Value *Cur = Ptr;
  while (Cur) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      // Capture the last (innermost) non-constant index as the per-thread
      // offset. Element-typed GEPs in this IR are single-index (i64/float/i8).
      for (Value *Idx : GEP->indices()) {
        if (!isa<ConstantInt>(Idx))
          OffsetOut = Idx;
      }
      Cur = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(Cur)) {
      Cur = BC->getOperand(0);
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
      Cur = ASC->getOperand(0);
      continue;
    }
    if (auto *GEPOp = dyn_cast<GEPOperator>(Cur)) {
      // Constant-expression GEP (rare post-prepare); peel it too.
      for (Value *Idx : GEPOp->indices())
        if (!isa<ConstantInt>(Idx))
          OffsetOut = Idx;
      Cur = GEPOp->getPointerOperand();
      continue;
    }
    break;
  }
  return Cur;
}

// Is this a plain (non-volatile, non-atomic) store to device memory whose
// pointer reaches a kernel Argument? Fills Base / Offset on success.
static bool isDeviceArgStore(StoreInst *SI, Value *&Base, Value *&Offset) {
  if (SI->isVolatile() || !SI->isSimple())
    return false;
  if (SI->getPointerAddressSpace() != kDeviceAS)
    return false;
  Base = traceBaseAndOffset(SI->getPointerOperand(), Offset);
  return isa<Argument>(Base);
}

// Find the in-bounds mask (an i1) that dominates SI and selects this offset.
// At -O0 each store is guarded by `icmp slt <offset>, N`; that icmp is still
// present in the flattened IR (the assume operands). We look for an i1 user of
// the offset that is an icmp comparing the offset against a bound. Returns null
// if none is found — the caller then skips the group (a constant-true guard
// would fold away and the AGX bug would return).
static Value *findMaskForOffset(Value *Offset, Instruction *InsertPt) {
  // Peel sext/zext: the GEP index is often `sext %i32off`, while the in-bounds
  // icmp compares the i32 offset directly. Collect candidate offset values.
  SmallVector<Value *, 3> Cands;
  if (Offset) {
    Cands.push_back(Offset);
    if (auto *Ext = dyn_cast<CastInst>(Offset))
      if (Ext->getOpcode() == Instruction::SExt ||
          Ext->getOpcode() == Instruction::ZExt)
        Cands.push_back(Ext->getOperand(0));
  }
  for (Value *Off : Cands) {
    // The offset SSA is typically an `or disjoint`/`shl` chain; the in-bounds
    // icmp compares exactly that value against the element count. Accept an
    // slt/ult icmp user of the offset that is defined before the store (so the
    // value dominates the new conditional branch we will emit at InsertPt).
    for (User *U : Off->users()) {
      auto *IC = dyn_cast<ICmpInst>(U);
      if (!IC || !IC->getType()->isIntegerTy(1))
        continue;
      if (IC->getPredicate() != ICmpInst::ICMP_SLT &&
          IC->getPredicate() != ICmpInst::ICMP_ULT)
        continue;
      // Defined before the store (entry-block icmp dominates everything; an
      // icmp in the same block must precede the store).
      if (IC->getParent() == &IC->getFunction()->getEntryBlock())
        return IC;
      if (IC->getParent() == InsertPt->getParent() && IC->comesBefore(InsertPt))
        return IC;
    }
  }
  return nullptr;
}

// Hoist a set of conflicting stores (all sharing one per-thread offset, hence
// one in-bounds predicate) into ONE shared predicated block guarded by `Guard`,
// inserted right before `At`:
//
//   <pre>                      br i1 %guard, %store_bb, %cont
//   store_bb: store0; store1; ... ; br %cont
//   cont:     <rest>
//
// The conflicting stores (which may be non-adjacent in the original block) are
// MOVED — in program order — into the new store block, along with the address
// GEPs they depend on. Non-conflicting stores stay in `cont` (unguarded,
// exactly as before), so a store with a DIFFERENT predicate/offset is never
// pulled under this guard. A real conditional branch on the in-bounds predicate
// is what makes the AGX JIT stop treating the warp-0 lane stores as
// unconditionally executed, so it no longer address-coalesces the cross-buffer
// pair. `Guard` is an i1 true for in-bounds threads; semantics are preserved
// exactly (out-of-bounds threads never stored anyway). Returns the continuation
// block for re-scanning.
static BasicBlock *guardStoreGroup(ArrayRef<StoreInst *> Group, Value *Guard) {
  assert(!Group.empty() && "empty store group");
  BasicBlock *Pre = Group.front()->getParent();
  // Anchor the hoist at the EARLIEST store in program order (the safety check
  // in the caller guarantees every store's value is defined before this point).
  Instruction *At = Group.front();
  for (StoreInst *SI : Group)
    if (SI->comesBefore(At))
      At = SI;

  // Split before the first store of the group: [At .. end) -> Tail. Then peel a
  // fresh empty StoreBB between Pre and Tail to hold exactly the group's
  // stores.
  BasicBlock *Tail = Pre->splitBasicBlock(At, Pre->getName() + ".cbssep.tail");
  BasicBlock *StoreBB = BasicBlock::Create(
      Pre->getContext(), Pre->getName() + ".cbssep.st", Pre->getParent(), Tail);

  // Pre: replace the unconditional branch (to Tail) with a guarded branch.
  Instruction *PreTerm = Pre->getTerminator();
  IRBuilder<> B(PreTerm);
  B.CreateCondBr(Guard, StoreBB, Tail);
  PreTerm->eraseFromParent();

  // Move each conflicting store into StoreBB, preserving the group's relative
  // order, together with the pure single-use instructions feeding its POINTER
  // and VALUE operands that currently live in Tail. The address chain (GEP /
  // bitcast) and value glue (e.g. `extractelement` from a vectorized result the
  // mid-end sank next to its store) are side-effect-free and only used by this
  // store, so relocating them is safe; their own operands are defined before
  // FirstSI (the split point) and still dominate. Foreign stores and shared
  // values are left in place. This keeps the move scoped to THIS offset group
  // even when other groups' stores are interleaved between its members.
  // Move shared instructions at most once across the whole group (a duplicated
  // output's value glue feeds several of the group's stores).
  SmallPtrSet<Instruction *, 16> Moved;
  IRBuilder<> SB(StoreBB);
  for (StoreInst *SI : Group) {
    // Collect the pure operand chain (pointer + value) living in Tail by a
    // small DFS, then move it in dependency order (operands before users).
    // Multi-use instructions are pulled too — the caller guaranteed every such
    // value's users are stores in this group, which all move into StoreBB, so
    // the def still precedes every use. `Moved` dedups shared glue.
    SmallVector<Instruction *, 8> Ordered;
    SmallPtrSet<Instruction *, 8> Seen;
    std::function<void(Value *)> pull = [&](Value *V) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getParent() != Tail || I->mayHaveSideEffects() ||
          isa<StoreInst>(I) || Moved.count(I) || Seen.count(I))
        return;
      // Only relocate a (possibly multi-use) instruction when EVERY user is
      // also being hoisted: a store in this group, or another instruction
      // already marked to move. Otherwise a user left behind in Tail would see
      // its def sink into the predicated StoreBB and no longer dominate it.
      for (User *U : I->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        bool UserHoisted = (UI && (Moved.count(UI) || Seen.count(UI))) ||
                           (isa<StoreInst>(U) &&
                            llvm::is_contained(Group, cast<StoreInst>(U)));
        if (!UserHoisted)
          return; // a non-hoisted user remains -> leave this def in place.
      }
      Seen.insert(I);
      for (Value *Op : I->operands())
        pull(Op);
      Ordered.push_back(I); // post-order: operands first
    };
    pull(SI->getPointerOperand());
    pull(SI->getValueOperand());
    for (Instruction *I : Ordered) {
      I->moveBefore(*StoreBB, StoreBB->end());
      Moved.insert(I);
    }
    SI->moveBefore(*StoreBB, StoreBB->end());
  }
  // Terminate StoreBB into Tail.
  SB.SetInsertPoint(StoreBB);
  SB.CreateBr(Tail);
  return Tail;
}

// Process one function. Returns true if any store was separated.
static bool separateInFunction(Function &F) {
  bool Changed = false;
  // Iterate over a worklist of blocks; sinking splits a block, so we re-scan.
  SmallVector<BasicBlock *, 16> Worklist;
  for (BasicBlock &BB : F)
    Worklist.push_back(&BB);

  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();

    // Collect device-arg stores in this block, in program order, with their
    // (base, offset).
    struct Rec {
      StoreInst *SI;
      Value *Base;
      Value *Offset;
    };
    SmallVector<Rec, 8> Stores;
    for (Instruction &I : *BB)
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *Base, *Offset;
        if (isDeviceArgStore(SI, Base, Offset))
          Stores.push_back({SI, Base, Offset});
      }
    if (Stores.size() < 2)
      continue;

    // Group conflicting stores BY OFFSET SSA. Two device stores conflict when
    // they share the same per-thread offset value but target DIFFERENT base
    // args (the same per-thread slot in >=2 buffers — the AGX coalescing
    // trigger). Each such offset group gets ITS OWN predicated block guarded by
    // ITS OWN in-bounds mask, so a store under a different predicate is never
    // forced under the wrong guard. Process one group per scan, then re-scan
    // the continuation (which now holds the remaining stores) for the next
    // group.
    DenseMap<Value *, SmallVector<StoreInst *, 4>> ByOffset;
    DenseMap<Value *, bool> CrossBuffer;
    DenseMap<Value *, Value *> AnyBaseForOffset;
    for (auto &R : Stores) {
      if (!R.Offset)
        continue;
      auto &Vec = ByOffset[R.Offset];
      Vec.push_back(R.SI);
      auto It = AnyBaseForOffset.find(R.Offset);
      if (It == AnyBaseForOffset.end())
        AnyBaseForOffset[R.Offset] = R.Base;
      else if (It->second != R.Base)
        CrossBuffer[R.Offset] = true;
    }

    // Pick one offset group that is genuinely cross-buffer, has a real mask,
    // and is safe to hoist. Hoisting moves the whole group to the position of
    // its FIRST store; for the result to dominate-check, every store's stored
    // VALUE (and pointer) must already be available there. A value computed
    // AFTER the first store (e.g. a `fptrunc` the mid-end sank below the store
    // run) would become a use-before-def -> the AIR writer emits an invalid
    // forward record. Such a group is skipped (left un-separated) rather than
    // miscompiled.
    SmallVector<StoreInst *, 4> *Group = nullptr;
    Value *Guard = nullptr;
    for (auto &KV : ByOffset) {
      if (!CrossBuffer.lookup(KV.first))
        continue; // single-buffer offset: not the trigger, leave it.
      Value *G = findMaskForOffset(KV.first, KV.second.front());
      if (!G)
        continue; // no real in-bounds mask: never emit a constant guard.
      // Hoist target = the group's first store (earliest in program order).
      StoreInst *FirstSI = KV.second.front();
      for (StoreInst *SI : KV.second)
        if (SI->comesBefore(FirstSI))
          FirstSI = SI;
      // guardStoreGroup relocates each store along with the pure single-use
      // glue feeding its pointer and value (e.g. a vectorized result's
      // `extractelement`). That keeps dominance EXCEPT when a store's value is
      // defined after FirstSI AND cannot be moved with it — i.e. it is impure
      // or shared (multi-use), so it must stay where it is and would then be
      // used before its def. Detect that case and skip the group (leave it
      // un-separated) rather than emit an invalid forward reference. A value
      // defined before FirstSI, or a movable pure single-use def, is always
      // safe. A store's value defined after FirstSI is fine as long as it is
      // PURE and every one of its users is a store in THIS group — then it (and
      // they) all move together into StoreBB and the def still precedes every
      // use. A duplicated output (out1==out4) makes the shared `extractelement`
      // multi-use, which is safe precisely because all those uses are hoisted.
      // An impure value, or one consumed by something outside the group (a
      // later non-store use, or a foreign store), cannot be relocated -> skip.
      auto allUsersInGroup = [&](Instruction *V) {
        for (User *U : V->users()) {
          auto *SU = dyn_cast<StoreInst>(U);
          if (!SU || !llvm::is_contained(KV.second, SU))
            return false;
        }
        return true;
      };
      bool Safe = true;
      for (StoreInst *SI : KV.second) {
        auto *V = dyn_cast<Instruction>(SI->getValueOperand());
        if (!V || V->getParent() != BB || V->comesBefore(FirstSI))
          continue; // defined before the hoist point (or non-instr) -> safe.
        if (V->mayHaveSideEffects() || !allUsersInGroup(V)) {
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue; // value-after-store would forward-reference; leave it.
      Group = &KV.second;
      Guard = G;
      break;
    }
    if (!Group)
      continue;

    // The group's stores are in program order within BB (ByOffset preserves
    // insertion order per key). Hoist them behind their shared guard.
    BasicBlock *Cont = guardStoreGroup(*Group, Guard);
    Changed = true;
    // Re-scan the continuation for further offset groups.
    Worklist.push_back(Cont);
  }
  return Changed;
}

static bool runImpl(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= separateInFunction(F);
  }
  return Changed;
}

} // namespace

PreservedAnalyses
AIRCrossBufferStoreSeparatePass::run(Module &M, ModuleAnalysisManager &AM) {
  return runImpl(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRCrossBufferStoreSeparateLegacy::runOnModule(Module &M) {
  return runImpl(M);
}

char AIRCrossBufferStoreSeparateLegacy::ID = 0;

INITIALIZE_PASS(AIRCrossBufferStoreSeparateLegacy, DEBUG_TYPE,
                "AIR Cross-Buffer Store Separate", false, false)

ModulePass *llvm::createAIRCrossBufferStoreSeparateLegacyPass() {
  return new AIRCrossBufferStoreSeparateLegacy();
}
