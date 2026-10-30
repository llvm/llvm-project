//===- AIRLowerAtomicRMW.cpp - Lower atomicrmw to AIR intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRLowerAtomicRMW.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Instructions.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "AIRKernelParams.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"

using namespace llvm;

#define DEBUG_TYPE "air-lower-atomic-rmw"

// Lower LLVM atomics to AIR's atomic functions, in the form of MSL 4.0
// (AIR 2.8), which floor's metal_atomic.hpp declares:
//
//   T    air.atomic.<space>.<op>.<sign>T(ptr, T val, i32 order, i32 scope,
//                                         i1 volatile)
//   i32  air.atomic.<space>.cmpxchg.weak.i32(ptr, ptr expected, i32 desired,
//                                             i32 success, i32 failure,
//                                             i32 scope, i1 volatile)
//   void air.atomic.fence(i32 flags, i32 order, i32 scope)
//
// <space> is "global" (device memory) or "local" (threadgroup memory). <sign>
// is "s." or "u." for integer operations whose result depends on the sign
// (min, max) and for add, sub, and, or, xor (which do not care, and use
// "s."); it is empty for xchg and for float operations. Before MSL 4.1 the
// operations themselves are relaxed: an ordered LLVM atomic becomes a relaxed
// one between a release fence and an acquire fence, as Metal.jl does.
//
// cmpxchg.weak returns the value it found, and on an Apple M2 Max never
// failed spuriously; a strong LLVM cmpxchg is lowered to one call and
// succeeds when the value found is the expected one.

namespace {
// MSL memory_order and thread_scope, and the mem_flags of fences.
enum : unsigned { OrderRelaxed = 0, OrderSeqCst = 5 };
enum : unsigned { ScopeThreadgroup = 1, ScopeDevice = 2 };
enum : unsigned { MemDevice = 1, MemThreadgroup = 2 };
} // namespace

static std::string buildAtomicName(bool IsGlobal, AtomicRMWInst::BinOp Op,
                                   bool IsFloat) {
  StringRef OpName, Sign;
  switch (Op) {
  case AtomicRMWInst::Xchg:
    OpName = "xchg";
    break;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::FAdd:
    OpName = "add", Sign = "s.";
    break;
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::FSub:
    OpName = "sub", Sign = "s.";
    break;
  case AtomicRMWInst::Max:
    OpName = "max", Sign = "s.";
    break;
  case AtomicRMWInst::Min:
    OpName = "min", Sign = "s.";
    break;
  case AtomicRMWInst::UMax:
    OpName = "max", Sign = "u.";
    break;
  case AtomicRMWInst::UMin:
    OpName = "min", Sign = "u.";
    break;
  case AtomicRMWInst::And:
    OpName = "and", Sign = "s.";
    break;
  case AtomicRMWInst::Or:
    OpName = "or", Sign = "s.";
    break;
  case AtomicRMWInst::Xor:
    OpName = "xor", Sign = "s.";
    break;
  default:
    return {};
  }
  if (IsFloat) {
    if (OpName != "add" && OpName != "sub" && OpName != "xchg")
      return {};
    Sign = "";
  }
  return ("air.atomic." + Twine(IsGlobal ? "global" : "local") + "." + OpName +
          "." + Sign + (IsFloat ? "f32" : "i32"))
      .str();
}

/// The MSL thread_scope of an atomic on address space AS with LLVM scope SSID.
static unsigned getScope(LLVMContext &Ctx, unsigned AS, SyncScope::ID SSID) {
  if (AS == metal::AS::Threadgroup)
    return ScopeThreadgroup;
  SmallVector<StringRef, 4> Names;
  Ctx.getSyncScopeNames(Names);
  if (SSID < Names.size() && Names[SSID] == "workgroup")
    return ScopeThreadgroup;
  return ScopeDevice;
}

/// The MSL memory_order of an LLVM ordering.
static unsigned mslOrder(AtomicOrdering O) {
  switch (O) {
  case AtomicOrdering::Acquire:
    return 2;
  case AtomicOrdering::Release:
    return 3;
  case AtomicOrdering::AcquireRelease:
    return 4;
  case AtomicOrdering::SequentiallyConsistent:
    return OrderSeqCst;
  default:
    return OrderRelaxed;
  }
}

/// air::AtomicOrderingMD on the AIR atomic \p Call.
static void addOrderingHint(CallInst *Call, ArrayRef<AtomicOrdering> Orders) {
  LLVMContext &Ctx = Call->getContext();
  SmallVector<Metadata *, 2> Ops;
  for (AtomicOrdering O : Orders)
    Ops.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(Ctx), mslOrder(O))));
  Call->setMetadata(air::AtomicOrderingMD, MDNode::get(Ctx, Ops));
}

static bool isOrdered(AtomicOrdering O) {
  return O != AtomicOrdering::NotAtomic && O != AtomicOrdering::Unordered &&
         O != AtomicOrdering::Monotonic;
}

/// air.atomic.fence for an ordered operation of scope Scope.
static void emitFence(IRBuilder<> &B, Module &M, unsigned Scope,
                      unsigned Order = OrderSeqCst) {
  Type *I32 = B.getInt32Ty();
  FunctionCallee F = M.getOrInsertFunction(
      "air.atomic.fence", FunctionType::get(B.getVoidTy(), {I32, I32, I32},
                                            false));
  unsigned Flags =
      Scope == ScopeThreadgroup ? MemThreadgroup : MemDevice | MemThreadgroup;
  B.CreateCall(F, {B.getInt32(Flags), B.getInt32(Order), B.getInt32(Scope)});
}

// ── Atomics Metal has no function for (Metal pipeline only) ─────────────────
//
// Metal's atomics (on an M2) are 32-bit. The others are emulated, in device
// memory (threadgroup memory: see run()):
//
// * 32-bit types other than i32 and float (<2 x half>, <2 x bfloat>, ...):
//   the compare-exchange loop below, on the bits as i32.
// * 8- and 16-bit: a compare-exchange loop on the aligned 32-bit word that
//   holds them, which also retries when the rest of the word changed.
// * 64- and 128-bit: under a spin lock, one 32-bit word of a table in device
//   memory picked by the address. The table is a buffer parameter added to
//   the kernel, "air_atomic_locks" at location index 30
//   (kAtomicLockTableIndex), of kAtomicLockCount zeroed words that the host
//   binds; a lock is released before the kernel ends, so the table stays
//   zero. A thread that does not get the lock leaves the loop iteration
//   without waiting, and the one holding it does its operation and releases
//   it in the same iteration: threads of one SIMD-group spinning on a lock
//   held by another thread of the group cannot keep it from running (no
//   independent thread scheduling is needed). The words of the value are read
//   and written with 32-bit atomics, so they are coherent across
//   threadgroups. Plain (non-atomic) accesses are not ordered with the lock,
//   and an atomic load of the value may see it half written.

namespace {
struct AtomicEmulation {
  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  Type *I32, *I64;
  DenseMap<Function *, Argument *> LockTables;

  AtomicEmulation(Module &M)
      : M(M), Ctx(M.getContext()), DL(M.getDataLayout()),
        I32(Type::getInt32Ty(Ctx)), I64(Type::getInt64Ty(Ctx)) {}

  static Value *pointerOf(Instruction *I) {
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
      return RMW->getPointerOperand();
    return cast<AtomicCmpXchgInst>(I)->getPointerOperand();
  }
  static Type *valueTypeOf(Instruction *I) {
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
      return RMW->getValOperand()->getType();
    return cast<AtomicCmpXchgInst>(I)->getNewValOperand()->getType();
  }

  // V (of type T) as an integer of its size, and back.
  Value *toInt(IRBuilder<> &B, Value *V) {
    Type *IT = B.getIntNTy(DL.getTypeSizeInBits(V->getType()));
    if (V->getType() == IT)
      return V;
    if (V->getType()->isPointerTy())
      return B.CreatePtrToInt(V, IT);
    return B.CreateBitCast(V, IT);
  }
  Value *fromInt(IRBuilder<> &B, Value *V, Type *T) {
    if (V->getType() == T)
      return V;
    if (T->isPointerTy())
      return B.CreateIntToPtr(V, T);
    return B.CreateBitCast(V, T);
  }

  // The address Off bytes into Ptr, as a pointer of its own (each written
  // with its own pointee type, see gepsOfIntToPtrAsIntegers).
  Value *addressAt(IRBuilder<> &B, Value *Ptr, Value *Off) {
    Value *A = B.CreatePtrToInt(Ptr, I64);
    if (Off)
      A = B.CreateAdd(A, Off);
    return B.CreateIntToPtr(A, Ptr->getType());
  }

  // Gives each function with a locked operation (and its callers, up to the
  // kernels) the lock table as a parameter.
  void addLockTables(ArrayRef<Instruction *> Locked) {
    SmallVector<AIRKernelParamRequest, 4> Requests;
    SmallPtrSet<Function *, 4> Seen;
    Type *Ptr = PointerType::get(Ctx, metal::AS::Device);
    for (Instruction *I : Locked)
      if (Seen.insert(I->getFunction()).second)
        Requests.push_back({I->getFunction(), kAtomicLockTableName, Ptr,
                            kAtomicLockTableName});
    AIRKernelParams Params = addKernelParams(M, Requests);
    for (auto &[F, Args] : Params.Args)
      LockTables[F] = Args.lookup(kAtomicLockTableName);
    // A pre-baked !air.kernel describes the kernel's parameter; without one
    // AIRSystemValues does (at the same location index).
    NamedMDNode *Kernels = M.getNamedMetadata("air.kernel");
    if (!Kernels)
      return;
    auto C32 = [&](unsigned V) {
      return ConstantAsMetadata::get(ConstantInt::get(I32, V));
    };
    for (auto &[KF, Key, Table] : Params.Added) {
      MDNode *Arg = MDNode::get(
          Ctx, {C32(Table->getArgNo()), MDString::get(Ctx, "air.buffer"),
                MDString::get(Ctx, "air.location_index"),
                C32(kAtomicLockTableIndex), C32(1),
                MDString::get(Ctx, "air.read_write"),
                MDString::get(Ctx, "air.address_space"), C32(1),
                MDString::get(Ctx, "air.arg_type_size"), C32(4),
                MDString::get(Ctx, "air.arg_type_align_size"), C32(4),
                MDString::get(Ctx, "air.arg_type_name"),
                MDString::get(Ctx, "uint"), MDString::get(Ctx, "air.arg_name"),
                MDString::get(Ctx, kAtomicLockTableName)});
      for (MDNode *K : Kernels->operands()) {
        auto *F =
            K->getNumOperands() == 3
                ? mdconst::dyn_extract_or_null<Function>(K->getOperand(0))
                : nullptr;
        auto *Args = F == KF ? dyn_cast<MDNode>(K->getOperand(2)) : nullptr;
        if (!Args)
          continue;
        SmallVector<Metadata *, 8> Ops(Args->operands());
        Ops.push_back(Arg);
        K->replaceOperandWith(2, MDNode::get(Ctx, Ops));
      }
    }
  }

  // The new value of an emulated operation on Old: an atomicrmw's, or a
  // compare-exchange's (the new value where Old is the expected one).
  Value *newValue(IRBuilder<> &B, Instruction *I, Value *Old) {
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
      return buildAtomicRMWValue(RMW->getOperation(), B, Old,
                                 RMW->getValOperand());
    auto *CX = cast<AtomicCmpXchgInst>(I);
    Value *Eq = B.CreateICmpEQ(toInt(B, Old), toInt(B, CX->getCompareOperand()));
    return B.CreateSelect(Eq, CX->getNewValOperand(), Old);
  }

  void replace(Instruction *I, IRBuilder<> &B, Value *Old, Value *Success) {
    Value *R = Old;
    if (isa<AtomicCmpXchgInst>(I)) {
      R = B.CreateInsertValue(PoisonValue::get(I->getType()), Old, 0);
      R = B.CreateInsertValue(R, Success, 1);
    }
    I->replaceAllUsesWith(R);
    I->eraseFromParent();
  }

  // 8- and 16-bit: a compare-exchange loop on the containing word.
  void subWord(Instruction *I) {
    Type *T = valueTypeOf(I);
    unsigned Bits = DL.getTypeSizeInBits(T);
    Type *IT = IntegerType::get(Ctx, Bits);
    Value *Ptr = pointerOf(I);
    auto *CX = dyn_cast<AtomicCmpXchgInst>(I);
    auto *RMW = dyn_cast<AtomicRMWInst>(I);
    AtomicOrdering Order =
        RMW ? RMW->getOrdering() : CX->getSuccessOrdering();
    SyncScope::ID SSID = RMW ? RMW->getSyncScopeID() : CX->getSyncScopeID();
    bool Volatile = RMW ? RMW->isVolatile() : CX->isVolatile();

    BasicBlock *Entry = I->getParent();
    Function *F = Entry->getParent();
    BasicBlock *Exit = SplitBlock(Entry, I);
    BasicBlock *Loop = BasicBlock::Create(Ctx, "atomic16.loop", F, Exit);
    BasicBlock *Try = BasicBlock::Create(Ctx, "atomic16.try", F, Exit);
    Entry->getTerminator()->setSuccessor(0, Loop);

    IRBuilder<> EB(Entry->getTerminator());
    Value *Addr = EB.CreatePtrToInt(Ptr, I64);
    Value *WordPtr =
        EB.CreateIntToPtr(EB.CreateAnd(Addr, ~uint64_t(3)), Ptr->getType());
    Value *Shift =
        EB.CreateShl(EB.CreateTrunc(EB.CreateAnd(Addr, 3), I32), 3, "shift");
    Value *Keep = EB.CreateNot(
        EB.CreateShl(ConstantInt::get(I32, maskTrailingOnes<uint32_t>(Bits)),
                     Shift));
    Value *Init = EB.CreateAlignedLoad(I32, WordPtr, Align(4), true);

    IRBuilder<> LB(Loop);
    PHINode *Old32 = LB.CreatePHI(I32, 2);
    Old32->addIncoming(Init, Entry);
    Value *OldN = fromInt(LB, LB.CreateTrunc(LB.CreateLShr(Old32, Shift), IT),
                          T);
    if (CX)
      LB.CreateCondBr(LB.CreateICmpEQ(toInt(LB, OldN),
                                      toInt(LB, CX->getCompareOperand())),
                      Try, Exit);
    else
      LB.CreateBr(Try);

    IRBuilder<> TB(Try);
    Value *NewN = toInt(TB, CX ? CX->getNewValOperand() : newValue(TB, I, OldN));
    Value *New32 = TB.CreateOr(TB.CreateAnd(Old32, Keep),
                               TB.CreateShl(TB.CreateZExt(NewN, I32), Shift));
    AtomicCmpXchgInst *W = TB.CreateAtomicCmpXchg(
        WordPtr, Old32, New32, Align(4), Order, AtomicOrdering::Monotonic,
        SSID);
    W->setVolatile(Volatile);
    Old32->addIncoming(TB.CreateExtractValue(W, 0), Try);
    Value *Done = TB.CreateExtractValue(W, 1);
    TB.CreateCondBr(Done, Exit, Loop);

    IRBuilder<> XB(&*Exit->getFirstInsertionPt());
    PHINode *Res = XB.CreatePHI(T, 2);
    Res->addIncoming(OldN, Try);
    Value *Success = nullptr;
    if (CX) {
      Res->addIncoming(OldN, Loop);
      PHINode *Ok = XB.CreatePHI(XB.getInt1Ty(), 2);
      Ok->addIncoming(XB.getTrue(), Try);
      Ok->addIncoming(XB.getFalse(), Loop);
      Success = Ok;
    }
    XB.SetInsertPoint(I);
    replace(I, XB, Res, Success);
  }

  // 64- and 128-bit: under a lock of the table.
  void locked(Instruction *I) {
    Type *T = valueTypeOf(I);
    unsigned Words = DL.getTypeSizeInBits(T) / 32;
    auto *RMW = dyn_cast<AtomicRMWInst>(I);
    SyncScope::ID SSID =
        RMW ? RMW->getSyncScopeID() : cast<AtomicCmpXchgInst>(I)->getSyncScopeID();

    BasicBlock *Entry = I->getParent();
    Function *F = Entry->getParent();
    Argument *Table = LockTables.lookup(F);
    Value *Ptr = pointerOf(I);
    BasicBlock *Exit = SplitBlock(Entry, I);
    BasicBlock *Loop = BasicBlock::Create(Ctx, "atomiclock.loop", F, Exit);
    BasicBlock *Crit = BasicBlock::Create(Ctx, "atomiclock.held", F, Exit);
    BasicBlock *Latch = BasicBlock::Create(Ctx, "atomiclock.latch", F, Exit);
    Entry->getTerminator()->setSuccessor(0, Loop);

    IRBuilder<> EB(Entry->getTerminator());
    Value *Slot = EB.CreateAnd(
        EB.CreateTrunc(EB.CreateLShr(EB.CreatePtrToInt(Ptr, I64), 3), I32),
        kAtomicLockCount - 1);
    Value *Lock = addressAt(
        EB, Table, EB.CreateZExt(EB.CreateShl(Slot, 2), I64));

    // Take the lock (device scope: the table is shared by every thread).
    IRBuilder<> LB(Loop);
    AtomicCmpXchgInst *Take = LB.CreateAtomicCmpXchg(
        Lock, LB.getInt32(0), LB.getInt32(1), Align(4),
        AtomicOrdering::SequentiallyConsistent, AtomicOrdering::Monotonic,
        SyncScope::System);
    Take->setVolatile(true);
    LB.CreateCondBr(LB.CreateExtractValue(Take, 1), Crit, Latch);

    // Holding it: read, update and write the value, then release the lock.
    IRBuilder<> CB(Crit);
    SmallVector<Value *, 4> WordPtrs;
    Value *OldInt = nullptr;
    IntegerType *IT = CB.getIntNTy(Words * 32);
    for (unsigned K = 0; K < Words; ++K) {
      WordPtrs.push_back(
          addressAt(CB, Ptr, K ? CB.getInt64(4 * K) : nullptr));
      AtomicRMWInst *R = CB.CreateAtomicRMW(
          AtomicRMWInst::Or, WordPtrs[K], CB.getInt32(0), Align(4),
          AtomicOrdering::Monotonic, SSID);
      R->setVolatile(true);
      Value *W = CB.CreateShl(CB.CreateZExt(R, IT), 32 * K);
      OldInt = OldInt ? CB.CreateOr(OldInt, W) : W;
    }
    Value *Old = fromInt(CB, OldInt, T);
    Value *NewInt = toInt(CB, newValue(CB, I, Old));
    for (unsigned K = 0; K < Words; ++K) {
      AtomicRMWInst *R = CB.CreateAtomicRMW(
          AtomicRMWInst::Xchg, WordPtrs[K],
          CB.CreateTrunc(CB.CreateLShr(NewInt, 32 * K), I32), Align(4),
          AtomicOrdering::Monotonic, SSID);
      R->setVolatile(true);
    }
    AtomicRMWInst *Release = CB.CreateAtomicRMW(
        AtomicRMWInst::Xchg, Lock, CB.getInt32(0), Align(4),
        AtomicOrdering::SequentiallyConsistent, SyncScope::System);
    Release->setVolatile(true);
    CB.CreateBr(Latch);

    IRBuilder<> TB(Latch);
    PHINode *Done = TB.CreatePHI(TB.getInt1Ty(), 2);
    Done->addIncoming(TB.getFalse(), Loop);
    Done->addIncoming(TB.getTrue(), Crit);
    PHINode *Res = TB.CreatePHI(T, 2);
    Res->addIncoming(PoisonValue::get(T), Loop);
    Res->addIncoming(Old, Crit);
    TB.CreateCondBr(Done, Exit, Loop);

    IRBuilder<> XB(I);
    Value *Success = nullptr;
    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I))
      Success = XB.CreateICmpEQ(toInt(XB, Res),
                                toInt(XB, CX->getCompareOperand()));
    replace(I, XB, Res, Success);
  }

  // 32-bit compare-exchange of a type other than i32 (a pointer in a 32-bit
  // address space): on the bits as i32.
  void cmpxchgAsI32(AtomicCmpXchgInst *CX) {
    IRBuilder<> B(CX);
    Type *T = CX->getNewValOperand()->getType();
    AtomicCmpXchgInst *N = B.CreateAtomicCmpXchg(
        CX->getPointerOperand(), toInt(B, CX->getCompareOperand()),
        toInt(B, CX->getNewValOperand()), CX->getAlign(),
        CX->getSuccessOrdering(), CX->getFailureOrdering(),
        CX->getSyncScopeID());
    N->setVolatile(CX->isVolatile());
    N->setWeak(CX->isWeak());
    replace(CX, B, fromInt(B, B.CreateExtractValue(N, 0), T),
            B.CreateExtractValue(N, 1));
  }

  bool run() {
    SmallVector<Instruction *, 8> SubWord, Locked;
    SmallVector<AtomicCmpXchgInst *, 4> CX32;
    for (Function &F : M)
      for (Instruction &I : instructions(F)) {
        if (!isa<AtomicRMWInst, AtomicCmpXchgInst>(I))
          continue;
        Type *T = valueTypeOf(&I);
        if (T->isIntegerTy(32) || T->isFloatTy())
          continue;
        unsigned Bits = DL.getTypeSizeInBits(T);
        // Device memory only: the threadgroup forms (the words addressed
        // through ptrtoint / inttoptr of a threadgroup pointer) hung an M2
        // Max (a 64-bit add and a 16-bit add on threadgroup globals, 4
        // threads); lowerAtomicRMW reports them.
        if (pointerOf(&I)->getType()->getPointerAddressSpace() !=
                metal::AS::Device &&
            Bits != 32)
          continue;
        if (Bits == 8 || Bits == 16)
          SubWord.push_back(&I);
        else if (Bits == 64 || Bits == 128)
          Locked.push_back(&I);
        else if (Bits == 32 && isa<AtomicCmpXchgInst>(I))
          CX32.push_back(cast<AtomicCmpXchgInst>(&I));
        // 32-bit read-modify-write operations of other types are the
        // compare-exchange loops of lowerAtomicRMW.
      }
    for (Instruction *I : SubWord)
      subWord(I);
    if (!Locked.empty())
      addLockTables(Locked);
    for (Instruction *I : Locked)
      locked(I);
    for (AtomicCmpXchgInst *CX : CX32)
      cmpxchgAsI32(CX);
    return !SubWord.empty() || !Locked.empty() || !CX32.empty();
  }
};
} // namespace

static bool lowerAtomicRMW(Module &M, bool OrderingHints) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  bool Changed = false;

  // 32-bit read-modify-write operations without an AIR function (nand,
  // uinc_wrap / udec_wrap: CUDA's atomicInc / atomicDec, float min / max) as
  // compare-exchange loops, which are lowered below. So are float add and sub
  // in threadgroup memory: an M2 Max rejects air.atomic.local.add.f32
  // ("Unsupported local float atomic operation for given target"). Only for
  // Metal: the portable form keeps the operation for consumers that have it
  // (AGX).
  if (!OrderingHints) {
    Changed |= AtomicEmulation(M).run();
    SmallVector<AtomicRMWInst *, 4> Loops;
    for (Function &F : M)
      for (BasicBlock &BB : F)
        for (Instruction &I : BB) {
          Type *T = nullptr;
          if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
            T = RMW->getValOperand()->getType();
          else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
            T = CX->getNewValOperand()->getType();
          // Metal's atomics are 32-bit: there is no AIR function to lower a
          // wider (or narrower) one to, and the AIR writer would abort.
          if (T && M.getDataLayout().getTypeSizeInBits(T) != 32)
            reportFatalUsageError(
                ("Metal has no " +
                 Twine(M.getDataLayout().getTypeSizeInBits(T)) +
                 "-bit atomic operations outside device memory (in " +
                 F.getName() + ")")
                    .str()
                    .c_str());
          if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
            unsigned AS =
                RMW->getPointerOperand()->getType()->getPointerAddressSpace();
            bool Native = T->isIntegerTy(32) || T->isFloatTy();
            bool LocalFloatArith =
                T->isFloatTy() && AS == metal::AS::Threadgroup &&
                (RMW->getOperation() == AtomicRMWInst::FAdd ||
                 RMW->getOperation() == AtomicRMWInst::FSub);
            if (!Native || LocalFloatArith ||
                buildAtomicName(AS != metal::AS::Threadgroup,
                                RMW->getOperation(), T->isFloatTy())
                    .empty())
              Loops.push_back(RMW);
          }
        }
    for (AtomicRMWInst *RMW : Loops) {
      Type *T = RMW->getValOperand()->getType();
      Value *Ptr = RMW->getPointerOperand();
      BasicBlock *Entry = RMW->getParent();
      BasicBlock *Exit = SplitBlock(Entry, RMW);
      BasicBlock *Loop = BasicBlock::Create(Ctx, "atomicrmw.loop",
                                            Entry->getParent(), Exit);
      Entry->getTerminator()->setSuccessor(0, Loop);
      IRBuilder<> EB(Entry->getTerminator());
      Value *Init = EB.CreateAlignedLoad(I32, Ptr, RMW->getAlign(), true);
      IRBuilder<> LB(Loop);
      PHINode *Old = LB.CreatePHI(I32, 2);
      Old->addIncoming(Init, Entry);
      Value *OldV = T != I32 ? LB.CreateBitCast(Old, T) : Old;
      Value *New = buildAtomicRMWValue(RMW->getOperation(), LB, OldV,
                                       RMW->getValOperand());
      if (T != I32)
        New = LB.CreateBitCast(New, I32);
      AtomicCmpXchgInst *CX = LB.CreateAtomicCmpXchg(
          Ptr, Old, New, RMW->getAlign(), RMW->getOrdering(),
          AtomicOrdering::Monotonic, RMW->getSyncScopeID());
      CX->setVolatile(RMW->isVolatile());
      Value *Loaded = LB.CreateExtractValue(CX, 0);
      Old->addIncoming(Loaded, Loop);
      LB.CreateCondBr(LB.CreateExtractValue(CX, 1), Exit, Loop);
      IRBuilder<> XB(RMW);
      RMW->replaceAllUsesWith(T != I32 ? XB.CreateBitCast(Loaded, T)
                                       : Loaded);
      RMW->eraseFromParent();
      Changed = true;
    }
  }

  SmallVector<Instruction *, 16> Atomics;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isa<AtomicRMWInst, AtomicCmpXchgInst, FenceInst>(I)) {
          // An atomic through a generic pointer (the address-space hint,
          // llvm/IR/AIR.h) has no AIR function; the portable form keeps it.
          Value *Ptr = nullptr;
          if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
            Ptr = RMW->getPointerOperand();
          else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
            Ptr = CX->getPointerOperand();
          if (OrderingHints && Ptr &&
              Ptr->getType()->getPointerAddressSpace() == air::GenericHintAS)
            continue;
          Atomics.push_back(&I);
        }

  for (Instruction *I : Atomics) {
    IRBuilder<> B(I);

    if (auto *Fence = dyn_cast<FenceInst>(I)) {
      // A fence within one thread orders nothing for other threads.
      // The portable form keeps the fence's ordering (an acquire or release
      // fence is cheaper than a full one on a consumer that has them); Metal
      // takes sequentially consistent fences.
      if (Fence->getSyncScopeID() != SyncScope::SingleThread)
        emitFence(B, M, getScope(Ctx, /*AS=*/0, Fence->getSyncScopeID()),
                  OrderingHints ? mslOrder(Fence->getOrdering())
                                : unsigned(OrderSeqCst));
      Fence->eraseFromParent();
      Changed = true;
      continue;
    }

    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
      Value *Ptr = CX->getPointerOperand();
      unsigned AS = Ptr->getType()->getPointerAddressSpace();
      if (!CX->getCompareOperand()->getType()->isIntegerTy(32))
        continue; // The writer reports what is left.
      unsigned Scope = getScope(Ctx, AS, CX->getSyncScopeID());
      bool Ordered = isOrdered(CX->getSuccessOrdering());
      // The expected value lives in a private alloca, in the entry block.
      IRBuilder<> EB(&*CX->getFunction()->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Expected = EB.CreateAlloca(I32, nullptr, "cmpxchg.expected");
      if (Ordered)
        emitFence(B, M, Scope);
      B.CreateStore(CX->getCompareOperand(), Expected);
      std::string Name =
          ("air.atomic." +
           Twine(AS == metal::AS::Threadgroup ? "local" : "global") +
           ".cmpxchg.weak.i32")
              .str();
      FunctionCallee F = M.getOrInsertFunction(
          Name, FunctionType::get(I32,
                                  {Ptr->getType(), Expected->getType(), I32,
                                   I32, I32, I32, I1},
                                  false));
      CallInst *Old = B.CreateCall(
          F, {Ptr, Expected, CX->getNewValOperand(), B.getInt32(OrderRelaxed),
              B.getInt32(OrderRelaxed), B.getInt32(Scope),
              B.getInt1(!OrderingHints || CX->isVolatile())});
      if (Ordered) {
        emitFence(B, M, Scope);
        if (OrderingHints)
          addOrderingHint(Old, {CX->getSuccessOrdering(),
                                CX->getFailureOrdering()});
      }
      Value *Success = B.CreateICmpEQ(Old, CX->getCompareOperand());
      Value *Result = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
      Result = B.CreateInsertValue(Result, Success, 1);
      CX->replaceAllUsesWith(Result);
      CX->eraseFromParent();
      Changed = true;
      continue;
    }

    auto *RMW = cast<AtomicRMWInst>(I);
    unsigned AddrSpace =
        RMW->getPointerOperand()->getType()->getPointerAddressSpace();
    bool IsGlobal = AddrSpace != metal::AS::Threadgroup;

    Type *ValTy = RMW->getValOperand()->getType();
    bool IsFloat = ValTy->isFloatTy();
    if (!(ValTy->isFloatTy() || ValTy->isIntegerTy(32)))
      continue;
    std::string Name = buildAtomicName(IsGlobal, RMW->getOperation(), IsFloat);
    if (Name.empty())
      continue;

    unsigned Scope = getScope(Ctx, AddrSpace, RMW->getSyncScopeID());
    bool Ordered = isOrdered(RMW->getOrdering());
    auto *PtrTy = PointerType::get(Ctx, AddrSpace);
    FunctionType *FTy =
        FunctionType::get(ValTy, {PtrTy, ValTy, I32, I32, I1}, false);
    FunctionCallee FC = M.getOrInsertFunction(Name, FTy);

    if (Ordered)
      emitFence(B, M, Scope);
    CallInst *Result = B.CreateCall(
        FC, {RMW->getPointerOperand(), RMW->getValOperand(),
             B.getInt32(OrderRelaxed), B.getInt32(Scope),
             B.getInt1(!OrderingHints || RMW->isVolatile())},
        RMW->getName());
    if (Ordered) {
      emitFence(B, M, Scope);
      if (OrderingHints)
        addOrderingHint(Result, RMW->getOrdering());
    }
    RMW->replaceAllUsesWith(Result);
    RMW->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses AIRLowerAtomicRMWPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return lowerAtomicRMW(M, /*OrderingHints=*/false) ? PreservedAnalyses::none()
                           : PreservedAnalyses::all();
}

bool AIRLowerAtomicRMWLegacy::runOnModule(Module &M) {
  return lowerAtomicRMW(M, OrderingHints);
}

char AIRLowerAtomicRMWLegacy::ID = 0;

INITIALIZE_PASS(AIRLowerAtomicRMWLegacy, DEBUG_TYPE, "AIR Lower AtomicRMW",
                false, false)

ModulePass *llvm::createAIRLowerAtomicRMWLegacyPass(bool OrderingHints) {
  return new AIRLowerAtomicRMWLegacy(OrderingHints);
}
