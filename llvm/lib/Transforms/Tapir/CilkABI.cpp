//===- CilkABI.cpp - Lower Tapir into Cilk runtime system calls -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cilk ABI to converts Tapir instructions to calls
// into the Cilk runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Timer.h"
#include "llvm/Transforms/Tapir/CilkRTSCilkFor.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "cilkabi"

STATISTIC(LoopsUsingRuntimeCilkFor,
          "Number of Tapir loops implemented using runtime cilk_for");

static cl::opt<bool> fastCilk(
    "fast-cilk", cl::init(false), cl::Hidden,
    cl::desc("Attempt faster Cilk call implementation"));

static cl::opt<bool> DebugABICalls(
    "debug-abi-calls", cl::init(false), cl::Hidden,
    cl::desc("Insert ABI calls for debugging"));

static cl::opt<bool> ArgStruct(
    "cilk-use-arg-struct", cl::init(false), cl::Hidden,
    cl::desc("Use a struct to store arguments for detached tasks"));

static const char TimerGroupName[] = DEBUG_TYPE;
static const char TimerGroupDescription[] = "CilkABI";

enum {
  __CILKRTS_ABI_VERSION = 1
};

enum {
  CILK_FRAME_STOLEN           =    0x01,
  CILK_FRAME_UNSYNCHED        =    0x02,
  CILK_FRAME_DETACHED         =    0x04,
  CILK_FRAME_EXCEPTION_PROBED =    0x08,
  CILK_FRAME_EXCEPTING        =    0x10,
  CILK_FRAME_LAST             =    0x80,
  CILK_FRAME_EXITING          =  0x0100,
  CILK_FRAME_SUSPENDED        =  0x8000,
  CILK_FRAME_UNWINDING        = 0x10000
};

#define CILK_FRAME_VERSION (__CILKRTS_ABI_VERSION << 24)
#define CILK_FRAME_VERSION_MASK  0xFF000000
#define CILK_FRAME_FLAGS_MASK    0x00FFFFFF
#define CILK_FRAME_VERSION_VALUE(_flags) (((_flags) & CILK_FRAME_VERSION_MASK) >> 24)
#define CILK_FRAME_MBZ  (~ (CILK_FRAME_STOLEN           |       \
                            CILK_FRAME_UNSYNCHED        |       \
                            CILK_FRAME_DETACHED         |       \
                            CILK_FRAME_EXCEPTION_PROBED |       \
                            CILK_FRAME_EXCEPTING        |       \
                            CILK_FRAME_LAST             |       \
                            CILK_FRAME_EXITING          |       \
                            CILK_FRAME_SUSPENDED        |       \
                            CILK_FRAME_UNWINDING        |       \
                            CILK_FRAME_VERSION_MASK))

#define CILKRTS_FUNC(name) Get__cilkrts_##name()

TapirTarget::ArgStructMode CilkABI::getArgStructMode() const {
  if (ArgStruct)
    return ArgStructMode::Dynamic;
  return ArgStructMode::None;
}

void CilkABI::addHelperAttributes(Function &Helper) {
  // Use a fast calling convention for the helper.
  Helper.setCallingConv(CallingConv::Fast);
  // Inlining the helper function is not legal.
  Helper.removeFnAttr(Attribute::AlwaysInline);
  Helper.addFnAttr(Attribute::NoInline);
  // If the helper uses an argument structure, then it is not a write-only
  // function.
  if (getArgStructMode() != ArgStructMode::None) {
    Helper.removeFnAttr(Attribute::WriteOnly);
    Helper.removeFnAttr(Attribute::ArgMemOnly);
    Helper.removeFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
  }
  // Note that the address of the helper is unimportant.
  Helper.setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  // The helper is private to this module.
  Helper.setLinkage(GlobalValue::PrivateLinkage);
}

CilkABI::CilkABI(Module &M) : TapirTarget(M) {
  LLVMContext &C = M.getContext();
  Type *VoidPtrTy = Type::getInt8PtrTy(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *Int16Ty = Type::getInt16Ty(C);

  // Get or create local definitions of Cilk RTS structure types.
  PedigreeTy = StructType::lookupOrCreate(C, "struct.__cilkrts_pedigree");
  StackFrameTy = StructType::lookupOrCreate(C, "struct.__cilkrts_stack_frame");
  WorkerTy = StructType::lookupOrCreate(C, "struct.__cilkrts_worker");

  if (PedigreeTy->isOpaque())
    PedigreeTy->setBody(Int64Ty, PointerType::getUnqual(PedigreeTy));
  if (StackFrameTy->isOpaque()) {
    Type *PedigreeUnionTy = StructType::get(PedigreeTy);
    StackFrameTy->setBody(Int32Ty, // flags
                          Int32Ty, // size
                          PointerType::getUnqual(StackFrameTy), // call_parent
                          PointerType::getUnqual(WorkerTy), // worker
                          VoidPtrTy, // except_data
                          ArrayType::get(VoidPtrTy, 5), // ctx
                          Int32Ty, // mxcsr
                          Int16Ty, // fpcsr
                          Int16Ty, // reserved
                          // union { spawn_helper_pedigree, parent_pedigree }
                          PedigreeUnionTy
                          );
  }
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  if (WorkerTy->isOpaque())
    WorkerTy->setBody(PointerType::getUnqual(StackFramePtrTy), // tail
                      PointerType::getUnqual(StackFramePtrTy), // head
                      PointerType::getUnqual(StackFramePtrTy), // exc
                      PointerType::getUnqual(StackFramePtrTy), // protected_tail
                      PointerType::getUnqual(StackFramePtrTy), // ltq_limit
                      Int32Ty, // self
                      VoidPtrTy, // g
                      VoidPtrTy, // l
                      VoidPtrTy, // reducer_map
                      StackFramePtrTy, // current_stack_frame
                      VoidPtrTy, // saved_protected_tail
                      VoidPtrTy, // sysdep
                      PedigreeTy // pedigree
                      );
}

// Accessors for opaque Cilk RTS functions
FunctionCallee CilkABI::Get__cilkrts_get_nworkers() {
  if (CilkRTSGetNworkers)
    return CilkRTSGetNworkers;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  AL = AL.addAttribute(C, AttributeList::FunctionIndex,
                       Attribute::ReadNone);
  // AL = AL.addAttribute(C, AttributeSet::FunctionIndex,
  //                      Attribute::InaccessibleMemOnly);
  AL = AL.addAttribute(C, AttributeList::FunctionIndex,
                       Attribute::NoUnwind);
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), {}, false);
  CilkRTSGetNworkers = M.getOrInsertFunction("__cilkrts_get_nworkers", FTy, AL);
  return CilkRTSGetNworkers;
}

FunctionCallee CilkABI::Get__cilkrts_init() {
  if (CilkRTSInit)
    return CilkRTSInit;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  CilkRTSInit = M.getOrInsertFunction("__cilkrts_init", VoidTy);

  return CilkRTSInit;
}

FunctionCallee CilkABI::Get__cilkrts_leave_frame() {
  if (CilkRTSLeaveFrame)
    return CilkRTSLeaveFrame;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSLeaveFrame = M.getOrInsertFunction("__cilkrts_leave_frame", VoidTy,
                                            StackFramePtrTy);

  return CilkRTSLeaveFrame;
}

FunctionCallee CilkABI::Get__cilkrts_rethrow() {
  if (CilkRTSRethrow)
    return CilkRTSRethrow;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSRethrow = M.getOrInsertFunction("__cilkrts_rethrow", VoidTy,
                                         StackFramePtrTy);

  return CilkRTSRethrow;
}

FunctionCallee CilkABI::Get__cilkrts_sync() {
  if (CilkRTSSync)
    return CilkRTSSync;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSSync = M.getOrInsertFunction("__cilkrts_sync", VoidTy,
                                      StackFramePtrTy);

  return CilkRTSSync;
}

FunctionCallee CilkABI::Get__cilkrts_get_tls_worker() {
  if (CilkRTSGetTLSWorker)
    return CilkRTSGetTLSWorker;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSGetTLSWorker = M.getOrInsertFunction("__cilkrts_get_tls_worker",
                                              WorkerPtrTy);

  return CilkRTSGetTLSWorker;
}

FunctionCallee CilkABI::Get__cilkrts_get_tls_worker_fast() {
  if (CilkRTSGetTLSWorkerFast)
    return CilkRTSGetTLSWorkerFast;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSGetTLSWorkerFast = M.getOrInsertFunction(
      "__cilkrts_get_tls_worker_fast", WorkerPtrTy);

  return CilkRTSGetTLSWorkerFast;
}

FunctionCallee CilkABI::Get__cilkrts_bind_thread_1() {
  if (CilkRTSBindThread1)
    return CilkRTSBindThread1;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSBindThread1 = M.getOrInsertFunction("__cilkrts_bind_thread_1",
                                             WorkerPtrTy);

  return CilkRTSBindThread1;
}

/// Helper methods for storing to and loading from struct fields.
static Value *GEP(IRBuilder<> &B, Value *Base, int Field) {
  // return B.CreateStructGEP(cast<PointerType>(Base->getType()),
  //                          Base, field);
  return B.CreateConstInBoundsGEP2_32(nullptr, Base, 0, Field);
}

static unsigned GetAlignment(const DataLayout &DL, StructType *STy, int Field) {
  return DL.getPrefTypeAlignment(STy->getElementType(Field));
}

static void StoreSTyField(IRBuilder<> &B, const DataLayout &DL, StructType *STy,
                          Value *Val, Value *Dst, int Field,
                          bool isVolatile = false,
                          AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  StoreInst *S = B.CreateAlignedStore(Val, GEP(B, Dst, Field),
                                      GetAlignment(DL, STy, Field), isVolatile);
  S->setOrdering(Ordering);
}

static Value *LoadSTyField(
    IRBuilder<> &B, const DataLayout &DL, StructType *STy, Value *Src,
    int Field, bool isVolatile = false,
    AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  LoadInst *L =  B.CreateAlignedLoad(GEP(B, Src, Field),
                                     GetAlignment(DL, STy, Field), isVolatile);
  L->setOrdering(Ordering);
  return L;
}

/// Emit inline assembly code to save the floating point state, for x86 Only.
void CilkABI::EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
  LLVMContext &C = B.getContext();
  FunctionType *FTy =
    FunctionType::get(Type::getVoidTy(C),
                      {PointerType::getUnqual(Type::getInt32Ty(C)),
                       PointerType::getUnqual(Type::getInt16Ty(C))},
                      false);

  Value *Asm = InlineAsm::get(FTy,
                              "stmxcsr $0\n\t" "fnstcw $1",
                              "*m,*m,~{dirflag},~{fpsr},~{flags}",
                              /*sideeffects*/ true);

  Value *Args[2] = {
    GEP(B, SF, StackFrameFields::mxcsr),
    GEP(B, SF, StackFrameFields::fpcsr)
  };

  B.CreateCall(Asm, Args);
}

/// Helper to find a function with the given name, creating it if it doesn't
/// already exist. Returns false if the function was inserted, indicating that
/// the body of the function has yet to be defined.
static bool GetOrCreateFunction(Module &M, const StringRef FnName,
                                FunctionType *FTy, Function *&Fn) {
  // If the function already exists then let the caller know.
  if ((Fn = M.getFunction(FnName)))
    return true;

  // Otherwise we have to create it.
  Fn = cast<Function>(M.getOrInsertFunction(FnName, FTy).getCallee());

  // Let the caller know that the function is incomplete and the body still
  // needs to be added.
  return false;
}

/// Emit a call to the CILK_SETJMP function.
CallInst *CilkABI::EmitCilkSetJmp(IRBuilder<> &B, Value *SF) {
  LLVMContext &Ctx = M.getContext();

  // We always want to save the floating point state too
  Triple T(M.getTargetTriple()); 
  if (T.getArch() == Triple::x86 || T.getArch() == Triple::x86_64)
    EmitSaveFloatingPointState(B, SF);

  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int8PtrTy = Type::getInt8PtrTy(Ctx);

  // Get the buffer to store program state
  // Buffer is a void**.
  Value *Buf = GEP(B, SF, StackFrameFields::ctx);

  // Store the frame pointer in the 0th slot
  Value *FrameAddr =
    B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
                 ConstantInt::get(Int32Ty, 0));

  Value *FrameSaveSlot = GEP(B, Buf, 0);
  B.CreateStore(FrameAddr, FrameSaveSlot, /*isVolatile=*/true);

  // Store stack pointer in the 2nd slot
  Value *StackAddr = B.CreateCall(
      Intrinsic::getDeclaration(&M, Intrinsic::stacksave));

  Value *StackSaveSlot = GEP(B, Buf, 2);
  B.CreateStore(StackAddr, StackSaveSlot, /*isVolatile=*/true);

  Buf = B.CreateBitCast(Buf, Int8PtrTy);

  // Call LLVM's EH setjmp, which is lightweight.
  Value* F = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_setjmp);

  CallInst *SetjmpCall = B.CreateCall(F, Buf);
  SetjmpCall->setCanReturnTwice();

  return SetjmpCall;
}

/// Get or create a LLVM function for __cilkrts_pop_frame.  It is equivalent to
/// the following C code:
///
/// __cilkrts_pop_frame(__cilkrts_stack_frame *sf) {
///   sf->worker->current_stack_frame = sf->call_parent;
///   sf->call_parent = nullptr;
/// }
Function *CilkABI::Get__cilkrts_pop_frame() {
  // Get or create the __cilkrts_pop_frame function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilkrts_pop_frame",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_pop_frame.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // sf->worker->current_stack_frame = sf->call_parent;
  StoreSTyField(B, DL, WorkerTy,
                LoadSTyField(B, DL, StackFrameTy, SF,
                             StackFrameFields::call_parent,
                             /*isVolatile=*/false,
                             AtomicOrdering::NotAtomic),
                LoadSTyField(B, DL, StackFrameTy, SF,
                             StackFrameFields::worker,
                             /*isVolatile=*/false,
                             AtomicOrdering::Acquire),
                WorkerFields::current_stack_frame,
                /*isVolatile=*/false,
                AtomicOrdering::Release);

  // sf->call_parent = nullptr;
  StoreSTyField(B, DL, StackFrameTy,
                Constant::getNullValue(PointerType::getUnqual(StackFrameTy)),
                SF, StackFrameFields::call_parent, /*isVolatile=*/false,
                AtomicOrdering::Release);

  B.CreateRetVoid();

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// Get or create a LLVM function for __cilkrts_detach.  It is equivalent to the
/// following C code:
///
/// void __cilkrts_detach(struct __cilkrts_stack_frame *sf) {
///   struct __cilkrts_worker *w = sf->worker;
///   struct __cilkrts_stack_frame *parent = sf->call_parent;
///   struct __cilkrts_stack_frame *volatile *tail = w->tail;
///
///   sf->spawn_helper_pedigree = w->pedigree;
///   parent->parent_pedigree = w->pedigree;
///
///   w->pedigree.rank = 0;
///   w->pedigree.next = &sf->spawn_helper_pedigree;
///
///   StoreStore_fence();
///
///   *tail++ = parent;
///   w->tail = tail;
///
///   sf->flags |= CILK_FRAME_DETACHED;
/// }
Function *CilkABI::Get__cilkrts_detach() {
  // Get or create the __cilkrts_detach function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilkrts_detach",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_detach.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // struct __cilkrts_worker *w = sf->worker;
  Value *W = LoadSTyField(B, DL, StackFrameTy, SF,
                          StackFrameFields::worker, /*isVolatile=*/false,
                          AtomicOrdering::NotAtomic);

  // __cilkrts_stack_frame *parent = sf->call_parent;
  Value *Parent = LoadSTyField(B, DL, StackFrameTy, SF,
                               StackFrameFields::call_parent,
                               /*isVolatile=*/false,
                               AtomicOrdering::NotAtomic);

  // __cilkrts_stack_frame *volatile *tail = w->tail;
  Value *Tail = LoadSTyField(B, DL, WorkerTy, W,
                             WorkerFields::tail, /*isVolatile=*/false,
                             AtomicOrdering::Acquire);

  // sf->spawn_helper_pedigree = w->pedigree;
  Value *WorkerPedigree = LoadSTyField(B, DL, WorkerTy, W,
                                       WorkerFields::pedigree);
  Value *NewHelperPedigree = B.CreateInsertValue(
      LoadSTyField(B, DL, StackFrameTy, SF,
                   StackFrameFields::parent_pedigree), WorkerPedigree, { 0 });
  StoreSTyField(B, DL, StackFrameTy, NewHelperPedigree, SF,
                StackFrameFields::parent_pedigree);
  // parent->parent_pedigree = w->pedigree;
  Value *NewParentPedigree = B.CreateInsertValue(
      LoadSTyField(B, DL, StackFrameTy, Parent,
                   StackFrameFields::parent_pedigree), WorkerPedigree, { 0 });
  StoreSTyField(B, DL, StackFrameTy, NewParentPedigree, Parent,
                StackFrameFields::parent_pedigree);

  // w->pedigree.rank = 0;
  {
    StructType *STy = PedigreeTy;
    Type *Ty = STy->getElementType(PedigreeFields::rank);
    StoreSTyField(B, DL, STy, ConstantInt::get(Ty, 0),
                  GEP(B, W, WorkerFields::pedigree), PedigreeFields::rank,
                  /*isVolatile=*/false, AtomicOrdering::Release);
  }

  // w->pedigree.next = &sf->spawn_helper_pedigree;
  StoreSTyField(B, DL, PedigreeTy,
                GEP(B, GEP(B, SF, StackFrameFields::parent_pedigree), 0),
                GEP(B, W, WorkerFields::pedigree), PedigreeFields::next,
                /*isVolatile=*/false, AtomicOrdering::Release);

  // StoreStore_fence();
  B.CreateFence(AtomicOrdering::Release);

  // *tail++ = parent;
  B.CreateStore(Parent, Tail, /*isVolatile=*/true);
  Tail = B.CreateConstGEP1_32(Tail, 1);

  // w->tail = tail;
  StoreSTyField(B, DL, WorkerTy, Tail, W, WorkerFields::tail,
                /*isVolatile=*/false, AtomicOrdering::Release);

  // sf->flags |= CILK_FRAME_DETACHED;
  {
    Value *F = LoadSTyField(B, DL, StackFrameTy, SF,
                            StackFrameFields::flags, /*isVolatile=*/false,
                            AtomicOrdering::Acquire);
    F = B.CreateOr(F, ConstantInt::get(F->getType(), CILK_FRAME_DETACHED));
    StoreSTyField(B, DL, StackFrameTy, F, SF,
                  StackFrameFields::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);
  }

  B.CreateRetVoid();

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// Get or create a LLVM function for __cilk_sync.  Calls to this function is
/// always inlined, as it saves the current stack/frame pointer values. This
/// function must be marked as returns_twice to allow it to be inlined, since
/// the call to setjmp is marked returns_twice.
///
/// It is equivalent to the following C code:
///
/// void __cilk_sync(struct __cilkrts_stack_frame *sf) {
///   if (sf->flags & CILK_FRAME_UNSYNCHED) {
///     sf->parent_pedigree = sf->worker->pedigree;
///     SAVE_FLOAT_STATE(*sf);
///     if (!CILK_SETJMP(sf->ctx))
///       __cilkrts_sync(sf);
///     else if (sf->flags & CILK_FRAME_EXCEPTING)
///       __cilkrts_rethrow(sf);
///   }
///   ++sf->worker->pedigree.rank;
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
Function *CilkABI::GetCilkSyncFn(bool instrument) {
  // Get or create the __cilk_sync function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilk_sync",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilk_sync.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.sync.test", Fn);
  BasicBlock *SaveState = BasicBlock::Create(Ctx, "cilk.sync.savestate", Fn);
  BasicBlock *SyncCall = BasicBlock::Create(Ctx, "cilk.sync.runtimecall", Fn);
  BasicBlock *Excepting = BasicBlock::Create(Ctx, "cilk.sync.excepting", Fn);
  BasicBlock *Rethrow = BasicBlock::Create(Ctx, "cilk.sync.rethrow", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (instrument)
    //   // cilk_sync_begin
    //   B.CreateCall(CILK_CSI_FUNC(sync_begin, M), SF);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags, /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Flags = B.CreateAnd(Flags,
                        ConstantInt::get(Flags->getType(),
                                         CILK_FRAME_UNSYNCHED));
    Value *Zero = ConstantInt::get(Flags->getType(), 0);
    Value *Unsynced = B.CreateICmpEQ(Flags, Zero);
    B.CreateCondBr(Unsynced, Exit, SaveState);
  }

  // SaveState
  {
    IRBuilder<> B(SaveState);

    // sf.parent_pedigree = sf.worker->pedigree;
    Value *NewParentPedigree = B.CreateInsertValue(
        LoadSTyField(B, DL, StackFrameTy, SF,
                     StackFrameFields::parent_pedigree),
        LoadSTyField(B, DL, WorkerTy,
                     LoadSTyField(B, DL, StackFrameTy, SF,
                                  StackFrameFields::worker,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire),
                     WorkerFields::pedigree), { 0 });
    StoreSTyField(B, DL, StackFrameTy, NewParentPedigree, SF,
                  StackFrameFields::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Excepting);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(sf);
    B.CreateCall(CILKRTS_FUNC(sync), SF);
    B.CreateBr(Exit);
  }

  // Excepting
  {
    IRBuilder<> B(Excepting);
    if (Rethrow) {
      // if (sf->flags & CILK_FRAME_EXCEPTING)
      Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                  StackFrameFields::flags,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire);
      Flags = B.CreateAnd(Flags,
                          ConstantInt::get(Flags->getType(),
                                           CILK_FRAME_EXCEPTING));
      Value *Zero = ConstantInt::get(Flags->getType(), 0);
      Value *CanExcept = B.CreateICmpEQ(Flags, Zero);
      B.CreateCondBr(CanExcept, Exit, Rethrow);
    } else {
      B.CreateBr(Exit);
    }
  }

  // Rethrow
  if (Rethrow) {
    IRBuilder<> B(Rethrow);
    // __cilkrts_rethrow(sf);
    B.CreateCall(CILKRTS_FUNC(rethrow), SF)->setDoesNotReturn();
    B.CreateUnreachable();
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Worker = LoadSTyField(B, DL, StackFrameTy, SF,
                                 StackFrameFields::worker,
                                 /*isVolatile=*/false,
                                 AtomicOrdering::Acquire);
    Value *Pedigree = GEP(B, Worker, WorkerFields::pedigree);
    Value *Rank = GEP(B, Pedigree, PedigreeFields::rank);
    unsigned RankAlignment = GetAlignment(DL, PedigreeTy,
                                          PedigreeFields::rank);
    B.CreateAlignedStore(B.CreateAdd(
                             B.CreateAlignedLoad(Rank, RankAlignment),
                             ConstantInt::get(
                                 Rank->getType()->getPointerElementType(), 1)),
                         Rank, RankAlignment);
    // if (instrument)
    //   // cilk_sync_end
    //   B.CreateCall(CILK_CSI_FUNC(sync_end, M), SF);

    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);

  return Fn;
}

/// Get or create a LLVM function for __cilk_sync_nothrow.  Calls to this
/// function is always inlined, as it saves the current stack/frame pointer
/// values. This function must be marked as returns_twice to allow it to be
/// inlined, since the call to setjmp is marked returns_twice.
///
/// It is equivalent to the following C code:
///
/// void __cilk_sync_nothrow(struct __cilkrts_stack_frame *sf) {
///   if (sf->flags & CILK_FRAME_UNSYNCHED) {
///     sf->parent_pedigree = sf->worker->pedigree;
///     SAVE_FLOAT_STATE(*sf);
///     if (!CILK_SETJMP(sf->ctx))
///       __cilkrts_sync(sf);
///   }
///   ++sf->worker->pedigree.rank;
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
Function *CilkABI::GetCilkSyncNothrowFn(bool instrument) {
  // Get or create the __cilk_sync_nothrow function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilk_sync_nothrow",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilk_sync_nothrow.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.sync.test", Fn);
  BasicBlock *SaveState = BasicBlock::Create(Ctx, "cilk.sync.savestate", Fn);
  BasicBlock *SyncCall = BasicBlock::Create(Ctx, "cilk.sync.runtimecall", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (instrument)
    //   // cilk_sync_begin
    //   B.CreateCall(CILK_CSI_FUNC(sync_begin, M), SF);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags, /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Flags = B.CreateAnd(Flags,
                        ConstantInt::get(Flags->getType(),
                                         CILK_FRAME_UNSYNCHED));
    Value *Zero = ConstantInt::get(Flags->getType(), 0);
    Value *Unsynced = B.CreateICmpEQ(Flags, Zero);
    B.CreateCondBr(Unsynced, Exit, SaveState);
  }

  // SaveState
  {
    IRBuilder<> B(SaveState);

    // sf.parent_pedigree = sf.worker->pedigree;
    Value *NewParentPedigree = B.CreateInsertValue(
        LoadSTyField(B, DL, StackFrameTy, SF,
                     StackFrameFields::parent_pedigree),
        LoadSTyField(B, DL, WorkerTy,
                     LoadSTyField(B, DL, StackFrameTy, SF,
                                  StackFrameFields::worker,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire),
                     WorkerFields::pedigree), { 0 });
    StoreSTyField(B, DL, StackFrameTy, NewParentPedigree, SF,
                  StackFrameFields::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Exit);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(sf);
    B.CreateCall(CILKRTS_FUNC(sync), SF);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Worker = LoadSTyField(B, DL, StackFrameTy, SF,
                                 StackFrameFields::worker,
                                 /*isVolatile=*/false,
                                 AtomicOrdering::Acquire);
    Value *Pedigree = GEP(B, Worker, WorkerFields::pedigree);
    Value *Rank = GEP(B, Pedigree, PedigreeFields::rank);
    unsigned RankAlignment = GetAlignment(DL, PedigreeTy,
                                          PedigreeFields::rank);
    B.CreateAlignedStore(B.CreateAdd(
                             B.CreateAlignedLoad(Rank, RankAlignment),
                             ConstantInt::get(
                                 Rank->getType()->getPointerElementType(), 1)),
                         Rank, RankAlignment);
    // if (instrument)
    //   // cilk_sync_end
    //   B.CreateCall(CILK_CSI_FUNC(sync_end, M), SF);

    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);

  return Fn;
}

/// Get or create a LLVM function for __cilk_sync.  Calls to this function is
/// always inlined, as it saves the current stack/frame pointer values. This
/// function must be marked as returns_twice to allow it to be inlined, since
/// the call to setjmp is marked returns_twice.
///
/// It is equivalent to the following C code:
///
/// void *__cilk_catch_exception(struct __cilkrts_stack_frame *sf, void *Exn) {
///   if (sf->flags & CILK_FRAME_UNSYNCHED) {
///     if (!CILK_SETJMP(sf->ctx)) {
///       sf->except_data = Exn;
///       sf->flags |= CILK_FRAME_EXCEPTING;
///       __cilkrts_sync(sf);
///     }
///     sf->flags &= ~CILK_FRAME_EXCEPTING
///     Exn = sf->except_data;
///   }
///   ++sf->worker->pedigree.rank;
///   return Exn;
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
Function *CilkABI::GetCilkCatchExceptionFn(Type *ExnTy) {
  // Get or create the __cilk_catch_exception function.
  LLVMContext &Ctx = M.getContext();

  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilk_catch_exception",
                          FunctionType::get(ExnTy,
                                            {StackFramePtrTy, ExnTy},
                                            false), Fn))
    return Fn;

  // Create the body of __cilk_catch_exeption
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args++;
  Value *Exn = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.catch.test", Fn);
  BasicBlock *SetJmp = BasicBlock::Create(Ctx, "cilk.catch.setjmp", Fn);
  BasicBlock *SyncCall = BasicBlock::Create(Ctx, "cilk.catch.runtimecall", Fn);
  BasicBlock *Catch = BasicBlock::Create(Ctx, "cilk.catch.catch", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.catch.end", Fn);

  Value *NewExn;

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags, /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Flags = B.CreateAnd(Flags,
                        ConstantInt::get(Flags->getType(),
                                         CILK_FRAME_UNSYNCHED));
    Value *Zero = ConstantInt::get(Flags->getType(), 0);
    Value *Unsynced = B.CreateICmpEQ(Flags, Zero);
    B.CreateCondBr(Unsynced, Exit, SetJmp);
  }

  // SetJmp
  {
    IRBuilder<> B(SetJmp);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Catch);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // sf->except_data = Exn;
    // sf->flags = sf->flags | CILK_FRAME_EXCEPTING;
    StoreSTyField(B, DL, StackFrameTy, Exn, SF,
                  StackFrameFields::except_data, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags,
                                /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Flags = B.CreateOr(Flags, ConstantInt::get(Flags->getType(),
                                               CILK_FRAME_EXCEPTING));
    StoreSTyField(B, DL, StackFrameTy, Flags, SF,
                  StackFrameFields::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);

    // __cilkrts_sync(sf);
    B.CreateCall(CILKRTS_FUNC(sync), SF);
    B.CreateBr(Catch);
  }

  // Catch
  {
    IRBuilder<> B(Catch);
    // sf->flags = sf->flags & ~CILK_FRAME_EXCEPTING;
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags,
                                /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Flags = B.CreateAnd(Flags, ConstantInt::get(Flags->getType(),
                                                ~CILK_FRAME_EXCEPTING));
    StoreSTyField(B, DL, StackFrameTy, Flags, SF,
                  StackFrameFields::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);

    // Exn = sf->except_data;
    NewExn = LoadSTyField(B, DL, StackFrameTy, SF,
                          StackFrameFields::except_data, /*isVolatile=*/false,
                          AtomicOrdering::Acquire);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    PHINode *ExnPN = B.CreatePHI(ExnTy, 2);
    ExnPN->addIncoming(Exn, Entry);
    ExnPN->addIncoming(NewExn, Catch);

    // ++sf.worker->pedigree.rank;
    Value *Worker = LoadSTyField(B, DL, StackFrameTy, SF,
                                 StackFrameFields::worker,
                                 /*isVolatile=*/false,
                                 AtomicOrdering::Acquire);
    Value *Pedigree = GEP(B, Worker, WorkerFields::pedigree);
    Value *Rank = GEP(B, Pedigree, PedigreeFields::rank);
    unsigned RankAlignment = GetAlignment(DL, PedigreeTy,
                                          PedigreeFields::rank);
    B.CreateAlignedStore(B.CreateAdd(
                             B.CreateAlignedLoad(Rank, RankAlignment),
                             ConstantInt::get(
                                 Rank->getType()->getPointerElementType(), 1)),
                         Rank, RankAlignment);

    B.CreateRet(ExnPN);
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);

  return Fn;
}

/// Get or create a LLVM function for __cilkrts_enter_frame.  It is equivalent
/// to the following C code:
///
/// void __cilkrts_enter_frame_1(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     if (w == 0) { /* slow path, rare */
///         w = __cilkrts_bind_thread_1();
///         sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
///     } else {
///         sf->flags = CILK_FRAME_VERSION;
///     }
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
Function *CilkABI::Get__cilkrts_enter_frame_1() {
  // Get or create the __cilkrts_enter_frame_1 function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilkrts_enter_frame_1",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_enter_frame_1.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  BasicBlock *SlowPath = BasicBlock::Create(Ctx, "slowpath", Fn);
  BasicBlock *FastPath = BasicBlock::Create(Ctx, "fastpath", Fn);
  BasicBlock *Cont = BasicBlock::Create(Ctx, "cont", Fn);

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  StructType *SFTy = StackFrameTy;

  // Block  (Entry)
  CallInst *W = nullptr;
  {
    IRBuilder<> B(Entry);
    // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
    if (fastCilk)
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast));
    else
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker));

    // if (w == 0)
    Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    B.CreateCondBr(Cond, SlowPath, FastPath);
  }
  // Block  (SlowPath)
  CallInst *Wslow = nullptr;
  {
    IRBuilder<> B(SlowPath);
    // w = __cilkrts_bind_thread_1();
    Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1));
    // sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
    Type *Ty = SFTy->getElementType(StackFrameFields::flags);
    StoreSTyField(B, DL, StackFrameTy,
                  ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
                  SF, StackFrameFields::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    B.CreateBr(Cont);
  }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    // sf->flags = CILK_FRAME_VERSION;
    Type *Ty = SFTy->getElementType(StackFrameFields::flags);
    StoreSTyField(B, DL, StackFrameTy,
                  ConstantInt::get(Ty, CILK_FRAME_VERSION),
                  SF, StackFrameFields::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    B.CreateBr(Cont);
  }
  // Block  (Cont)
  {
    IRBuilder<> B(Cont);
    Value *Wfast = W;
    PHINode *W  = B.CreatePHI(WorkerPtrTy, 2);
    W->addIncoming(Wslow, SlowPath);
    W->addIncoming(Wfast, FastPath);

    // sf->call_parent = w->current_stack_frame;
    StoreSTyField(B, DL, StackFrameTy,
                  LoadSTyField(B, DL, WorkerTy, W,
                               WorkerFields::current_stack_frame,
                               /*isVolatile=*/false,
                               AtomicOrdering::Acquire),
                  SF, StackFrameFields::call_parent, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // sf->worker = w;
    StoreSTyField(B, DL, StackFrameTy, W, SF,
                  StackFrameFields::worker, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // w->current_stack_frame = sf;
    StoreSTyField(B, DL, WorkerTy, SF, W,
                  WorkerFields::current_stack_frame, /*isVolatile=*/false,
                  AtomicOrdering::Release);

    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// Get or create a LLVM function for __cilkrts_enter_frame_fast.  It is
/// equivalent to the following C code:
///
/// void __cilkrts_enter_frame_fast_1(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     sf->flags = CILK_FRAME_VERSION;
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
Function *CilkABI::Get__cilkrts_enter_frame_fast_1() {
  // Get or create the __cilkrts_enter_frame_fast_1 function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilkrts_enter_frame_fast_1",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_enter_frame_fast_1.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);

  IRBuilder<> B(Entry);
  Value *W;

  // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
  // if (fastCilk)
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast));
  // else
  //   W = B.CreateCall(CILKRTS_FUNC(get_tls_worker));

  StructType *SFTy = StackFrameTy;
  Type *Ty = SFTy->getElementType(StackFrameFields::flags);

  // sf->flags = CILK_FRAME_VERSION;
  StoreSTyField(B, DL, StackFrameTy,
                ConstantInt::get(Ty, CILK_FRAME_VERSION),
                SF, StackFrameFields::flags, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // sf->call_parent = w->current_stack_frame;
  StoreSTyField(B, DL, StackFrameTy,
                LoadSTyField(B, DL, WorkerTy, W,
                             WorkerFields::current_stack_frame,
                             /*isVolatile=*/false,
                             AtomicOrdering::Acquire),
                SF, StackFrameFields::call_parent, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // sf->worker = w;
  StoreSTyField(B, DL, StackFrameTy, W, SF,
                StackFrameFields::worker, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // w->current_stack_frame = sf;
  StoreSTyField(B, DL, WorkerTy, SF, W,
                WorkerFields::current_stack_frame, /*isVolatile=*/false,
                AtomicOrdering::Release);

  B.CreateRetVoid();

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

// /// Get or create a LLVM function for __cilk_parent_prologue.
// /// It is equivalent to the following C code:
// ///
// /// void __cilk_parent_prologue(__cilkrts_stack_frame *sf) {
// ///   __cilkrts_enter_frame_1(sf);
// /// }
// static Function *GetCilkParentPrologue(Module &M) {
//   Function *Fn = 0;

//   if (GetOrCreateFunction<cilk_func>("__cilk_parent_prologue", M, Fn))
//     return Fn;

//   // If we get here we need to add the function body
//   LLVMContext &Ctx = M.getContext();

//   Function::arg_iterator args = Fn->arg_begin();
//   Value *SF = &*args;

//   BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
//   IRBuilder<> B(Entry);

//   // __cilkrts_enter_frame_1(sf)
//   B.CreateCall(CILKRTS_FUNC(enter_frame_1), SF);

//   B.CreateRetVoid();

//   Fn->addFnAttr(Attribute::InlineHint);

//   return Fn;
// }

/// Get or create a LLVM function for __cilk_parent_epilogue.  It is equivalent
/// to the following C code:
///
/// void __cilk_parent_epilogue(__cilkrts_stack_frame *sf) {
///   __cilkrts_pop_frame(sf);
///   if (sf->flags != CILK_FRAME_VERSION)
///     __cilkrts_leave_frame(sf);
/// }
Function *CilkABI::GetCilkParentEpilogueFn(bool instrument) {
  // Get or create the __cilk_parent_epilogue function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilk_parent_epilogue",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilk_parent_epilogue.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn),
    *B1 = BasicBlock::Create(Ctx, "body", Fn),
    *Exit  = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (instrument)
    //   // cilk_leave_begin
    //   B.CreateCall(CILK_CSI_FUNC(leave_begin, M), SF);

    // __cilkrts_pop_frame(sf)
    B.CreateCall(CILKRTS_FUNC(pop_frame), SF);

    // if (sf->flags != CILK_FRAME_VERSION)
    Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
                                StackFrameFields::flags, /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Value *Cond = B.CreateICmpNE(
        Flags, ConstantInt::get(Flags->getType(), CILK_FRAME_VERSION));
    B.CreateCondBr(Cond, B1, Exit);
  }

  // B1
  {
    IRBuilder<> B(B1);

    // __cilkrts_leave_frame(sf);
    B.CreateCall(CILKRTS_FUNC(leave_frame), SF);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);
    // if (instrument)
    //   // cilk_leave_end
    //   B.CreateCall(CILK_CSI_FUNC(leave_end, M));
    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const StringRef stack_frame_name = "__cilkrts_sf";

/// Create the __cilkrts_stack_frame for the spawning function.
AllocaInst *CilkABI::CreateStackFrame(Function &F) {
  const DataLayout &DL = M.getDataLayout();
  Type *SFTy = StackFrameTy;

  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *SF = B.CreateAlloca(SFTy, DL.getAllocaAddrSpace(),
                                  /*ArraySize*/nullptr,
                                  /*Name*/stack_frame_name);
  SF->setAlignment(8);

  return SF;
}

Value *CilkABI::GetOrInitCilkStackFrame(Function &F, bool Helper,
                                        bool instrument) {
  if (DetachCtxToStackFrame.count(&F))
    return DetachCtxToStackFrame[&F];

  AllocaInst *SF = CreateStackFrame(F);
  DetachCtxToStackFrame[&F] = SF;
  BasicBlock::iterator InsertPt = ++SF->getIterator();
  IRBuilder<> IRB(&(F.getEntryBlock()), InsertPt);

  // if (instrument) {
  //   Type *Int8PtrTy = IRB.getInt8PtrTy();
  //   Value *ThisFn = ConstantExpr::getBitCast(&F, Int8PtrTy);
  //   Value *ReturnAddress =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M,
  //                                              Intrinsic::returnaddress),
  //                    IRB.getInt32(0));
  //   StackSave =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M,
  //                                              Intrinsic::stacksave));
  //   if (Helper) {
  //     Value *begin_args[3] = { SF, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *M),
  //                    begin_args);
  //   } else {
  //     Value *begin_args[4] = { IRB.getInt32(0), SF, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_begin, *M), begin_args);
  //   }
  // }
  Value *Args[1] = { SF };
  if (Helper || fastCilk)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1), Args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1), Args);

  // if (instrument) {
  //   Value* end_args[2] = { SF, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  // }

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogueFn(instrument), Args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      // /*
      //   sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
      //   sf.except_data = Exn;
      // */
      // IRBuilder<> B(RI);
      // Value *Exn = AtExit->CreateExtractValue(RI->getValue(),
      //                                         ArrayRef<unsigned>(0));
      // Value *Flags = LoadSTyField(*AtExit, DL, StackFrameTy, SF,
      //                             StackFrameFields::flags,
      //                             /*isVolatile=*/false,
      //                             AtomicOrdering::Acquire);
      // Flags = AtExit->CreateOr(Flags,
      //                          ConstantInt::get(Flags->getType(),
      //                                           CILK_FRAME_EXCEPTING));
      // StoreSTyField(*AtExit, DL, StackFrameTy, Flags, SF,
      //               StackFrameFields::flags, /*isVolatile=*/false,
      //               AtomicOrdering::Release);
      // StoreSTyField(*AtExit, DL, StackFrameTy, Exn, SF,
      //               StackFrameFields::except_data, /*isVolatile=*/false,
      //               AtomicOrdering::Release);
      /*
        __cilkrts_pop_frame(&sf);
        if (sf->flags)
          __cilkrts_leave_frame(&sf);
      */
      AtExit->CreateCall(GetCilkParentEpilogueFn(instrument), Args, "");
    }
  }

  return SF;
}

bool CilkABI::makeFunctionDetachable(Function &Extracted, bool instrument) {
  /*
    __cilkrts_stack_frame sf;
    __cilkrts_enter_frame_fast_1(&sf);
    __cilkrts_detach();
    *x = f(y);
  */

  const DataLayout& DL = M.getDataLayout();
  AllocaInst *SF = CreateStackFrame(Extracted);
  DetachCtxToStackFrame[&Extracted] = SF;
  assert(SF && "Error creating Cilk stack frame in helper.");
  Value *Args[1] = { SF };

  // Scan function to see if it detaches.
  LLVM_DEBUG({
      bool SimpleHelper = !canDetach(&Extracted);
      if (!SimpleHelper)
        dbgs() << "NOTE: Detachable helper function itself detaches.\n";
    });

  BasicBlock::iterator InsertPt = ++SF->getIterator();
  IRBuilder<> IRB(&(Extracted.getEntryBlock()), InsertPt);

  // if (instrument) {
  //   Type *Int8PtrTy = IRB.getInt8PtrTy();
  //   Value *ThisFn = ConstantExpr::getBitCast(&Extracted, Int8PtrTy);
  //   Value *ReturnAddress =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::returnaddress),
  //                    IRB.getInt32(0));
  //   StackSave =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stacksave));
  //   Value *begin_args[3] = { SF, ThisFn, ReturnAddress };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *M), begin_args);
  // }

  IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1), Args);

  // if (instrument) {
  //   Value *end_args[2] = { SF, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  // }

  // __cilkrts_detach()
  {
    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_begin, *M), args);

    IRB.CreateCall(CILKRTS_FUNC(detach), Args);

    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_end, *M));
  }

  EscapeEnumerator EE(Extracted, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogueFn(instrument), Args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      /*
        sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
        sf.except_data = Exn;
      */
      IRBuilder<> B(RI);
      Value *Exn = AtExit->CreateExtractValue(RI->getValue(), { 0 });
      Value *Flags = LoadSTyField(*AtExit, DL, StackFrameTy, SF,
                                  StackFrameFields::flags,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire);
      Flags = AtExit->CreateOr(Flags,
                               ConstantInt::get(Flags->getType(),
                                                CILK_FRAME_EXCEPTING));
      StoreSTyField(*AtExit, DL, StackFrameTy, Flags, SF,
                    StackFrameFields::flags, /*isVolatile=*/false,
                    AtomicOrdering::Release);
      StoreSTyField(*AtExit, DL, StackFrameTy, Exn, SF,
                    StackFrameFields::except_data, /*isVolatile=*/false,
                    AtomicOrdering::Release);
      /*
        __cilkrts_pop_frame(&sf);
        if (sf->flags)
          __cilkrts_leave_frame(&sf);
      */
      AtExit->CreateCall(GetCilkParentEpilogueFn(instrument), Args, "");
    }
  }

  return true;
}

/// Lower a call to get the grainsize of this Tapir loop.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
Value *CilkABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  IRBuilder<> Builder(GrainsizeCall);

  // Get 8 * workers
  Value *Workers = Builder.CreateCall(CILKRTS_FUNC(get_nworkers));
  Value *WorkersX8 = Builder.CreateIntCast(
      Builder.CreateMul(Workers, ConstantInt::get(Workers->getType(), 8)),
      Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) =
  //           (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, WorkersX8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       WorkersX8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Value *Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void CilkABI::lowerSync(SyncInst &SI) {
  Function &Fn = *(SI.getFunction());

  Value *SF = GetOrInitCilkStackFrame(Fn, /*Helper*/false, false);
  Value *args[] = { SF };
  assert(args[0] && "sync used in function without frame!");
  CallInst *CI;

  if (Fn.doesNotThrow())
    CI = CallInst::Create(GetCilkSyncNothrowFn(), args, "",
                          /*insert before*/&SI);
  else
    CI = CallInst::Create(GetCilkSyncFn(), args, "",
                          /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
  // Mark this function as stealable.
  Fn.addFnAttr(Attribute::Stealable);
}

void CilkABI::processOutlinedTask(Function &F) {
  NamedRegionTimer NRT("processOutlinedTask", "Process outlined task",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);
  makeFunctionDetachable(F, false);
}

void CilkABI::processSpawner(Function &F) {
  NamedRegionTimer NRT("processSpawner", "Process spawner",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);
  GetOrInitCilkStackFrame(F, /*Helper=*/false, false);

  // Mark this function as stealable.
  F.addFnAttr(Attribute::Stealable);
}

void CilkABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  NamedRegionTimer NRT("processSubTaskCall", "Process subtask call",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();

  LLVM_DEBUG(dbgs() << "CilkABI::processSubTaskCall: " << *ReplCall << "\n");

  Function &F = *ReplCall->getFunction();
  assert(DetachCtxToStackFrame.count(&F) &&
         "No frame found for spawning task.");
  Value *SF = DetachCtxToStackFrame[&F];

  if (InvokeInst *II = dyn_cast<InvokeInst>(ReplCall)) {
    LandingPadInst *LPI = II->getLandingPadInst();
    IRBuilder<> B(&*II->getUnwindDest()->getFirstInsertionPt());
    Value *Exn = B.CreateExtractValue(LPI, { 0 });
    Value *NewExn = B.CreateCall(GetCilkCatchExceptionFn(Exn->getType()),
                                 { SF, Exn });
    B.CreateInsertValue(LPI, NewExn, { 0 });
  }

  // Split the basic block containing the detach replacement just before the
  // start of the detach-replacement instructions.
  BasicBlock *DetBlock = ReplStart->getParent();
  BasicBlock *CallBlock = SplitBlock(DetBlock, ReplStart, &DT);

  // Emit a Cilk setjmp at the end of the block preceding the split-off detach
  // replacement.
  Instruction *SetJmpPt = DetBlock->getTerminator();
  IRBuilder<> B(SetJmpPt);
  Value *SetJmpRes = EmitCilkSetJmp(B, SF);

  // Get the ordinary continuation of the detach.
  BasicBlock *CallCont;
  if (InvokeInst *II = dyn_cast<InvokeInst>(ReplCall))
    CallCont = II->getNormalDest();
  else // isa<CallInst>(CallSite)
    CallCont = CallBlock->getSingleSuccessor();

  // Insert a conditional branch, based on the result of the setjmp, to either
  // the detach replacement or the continuation.
  SetJmpRes = B.CreateICmpEQ(SetJmpRes,
                             ConstantInt::get(SetJmpRes->getType(), 0));
  B.CreateCondBr(SetJmpRes, CallBlock, CallCont);
  // Add DetBlock as a predecessor for all Phi nodes in CallCont.  These Phi
  // nodes receive the same value from DetBlock as from CallBlock.
  for (PHINode &Phi : CallCont->phis())
    Phi.addIncoming(Phi.getIncomingValueForBlock(CallBlock), DetBlock);
  SetJmpPt->eraseFromParent();

  // If we're not using dynamic argument structs, then no further processing is
  // needed.
  if (ArgStructMode::Dynamic != getArgStructMode())
    return;

  // Create a separate spawn-helper function to allocate and populate the
  // argument struct.

  // Inputs to the spawn helper
  ValueSet SHInputSet = TOI.InputSet;
  ValueSet SHInputs;
  fixupInputSet(*Parent, SHInputSet, SHInputs);
  LLVM_DEBUG({
      dbgs() << "SHInputSet:\n";
      for (Value *V : SHInputSet)
        dbgs() << "\t" << *V << "\n";
      dbgs() << "SHInputs:\n";
      for (Value *V : SHInputs)
        dbgs() << "\t" << *V << "\n";
    });
  ValueSet Outputs;  // Should be empty.
  // Only one block needs to be cloned into the spawn helper
  std::vector<BasicBlock *> BlocksToClone;
  BlocksToClone.push_back(CallBlock);
  SmallVector<ReturnInst *, 1> Returns;  // Ignore returns cloned.
  ValueToValueMapTy VMap;
  Twine NameSuffix = ".shelper";
  Function *SpawnHelper =
      CreateHelper(SHInputs, Outputs, BlocksToClone, CallBlock, DetBlock,
                   CallCont, VMap, &M, Parent->getSubprogram() != nullptr,
                   Returns, NameSuffix.str(), nullptr, nullptr, nullptr,
                   UnwindDest);

  assert(Returns.empty() && "Returns cloned when creating SpawnHelper.");

  // Use a fast calling convention for the helper.
  SpawnHelper->setCallingConv(CallingConv::Fast);
  // Add attributes to new helper function.
  SpawnHelper->addFnAttr(Attribute::NoInline);
  if (!UnwindDest) {
    SpawnHelper->addFnAttr(Attribute::NoUnwind);
    SpawnHelper->addFnAttr(Attribute::UWTable);
  }
  // Note that the address of the helper is unimportant.
  SpawnHelper->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  // The helper is private to this module.
  SpawnHelper->setLinkage(GlobalValue::PrivateLinkage);

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(Parent, SHInputs, VMap, ReplCall, nullptr, nullptr);

  // Move allocas in the newly cloned block to the entry block of the helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> Ends;
    // Ends.push_back(cast<BasicBlock>(VMap[CallBlock])->getTerminator());
    Ends.push_back(cast<BasicBlock>(VMap[CallCont])->getTerminator());
    if (isa<InvokeInst>(ReplCall))
      Ends.push_back(cast<BasicBlock>(VMap[UnwindDest])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedBlock = cast<BasicBlock>(VMap[CallBlock]);
    MoveStaticAllocasInBlock(&SpawnHelper->getEntryBlock(), ClonedBlock, Ends);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  // Insert a call to the spawn helper.
  SmallVector<Value *, 8> SHInputVec;
  for (Value *V : SHInputs)
    SHInputVec.push_back(V);
  SplitEdge(DetBlock, CallBlock);
  B.SetInsertPoint(CallBlock->getTerminator());
  if (isa<InvokeInst>(ReplCall)) {
    InvokeInst *SpawnHelperCall = InvokeInst::Create(SpawnHelper, CallCont,
                                                     UnwindDest, SHInputVec);
    SpawnHelperCall->setDebugLoc(ReplCall->getDebugLoc());
    SpawnHelperCall->setCallingConv(SpawnHelper->getCallingConv());
    // The invoke of the spawn helper can replace the terminator in CallBlock.
    ReplaceInstWithInst(CallBlock->getTerminator(), SpawnHelperCall);
  } else {
    CallInst *SpawnHelperCall = B.CreateCall(SpawnHelper, SHInputVec);
    SpawnHelperCall->setDebugLoc(ReplCall->getDebugLoc());
    SpawnHelperCall->setCallingConv(SpawnHelper->getCallingConv());
    SpawnHelperCall->setDoesNotThrow();
    // Branch around CallBlock.  Its contents are now dead.
    ReplaceInstWithInst(CallBlock->getTerminator(),
                        BranchInst::Create(CallCont));
  }
}

// Helper function to inline calls to compiler-generated Cilk Plus runtime
// functions when possible.  This inlining is necessary to properly implement
// some Cilk runtime "calls," such as __cilkrts_detach().
static inline void inlineCilkFunctions(Function &F) {
  bool Changed;
  do {
    Changed = false;
    for (Instruction &I : instructions(F))
      if (CallInst *Call = dyn_cast<CallInst>(&I))
        if (Function *Fn = Call->getCalledFunction())
          if (Fn->getName().startswith("__cilk")) {
            InlineFunctionInfo IFI;
            if (InlineFunction(Call, IFI)) {
              if (Fn->hasNUses(0))
                Fn->eraseFromParent();
              Changed = true;
              break;
            }
          }
  } while (Changed);

  if (verifyFunction(F, &errs()))
    llvm_unreachable("Tapir->CilkABI lowering produced bad IR!");
}

void CilkABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  LLVM_DEBUG(dbgs() << "CilkABI processing function " << F.getName() << "\n");
  if (fastCilk && F.getName() == "main") {
    IRBuilder<> B(F.getEntryBlock().getTerminator());
    B.CreateCall(CILKRTS_FUNC(init));
  }
}

void CilkABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  NamedRegionTimer NRT("postProcessFunction", "Post-process function",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);
  if (!DebugABICalls)
    inlineCilkFunctions(F);
}

void CilkABI::postProcessHelper(Function &F) {
  NamedRegionTimer NRT("postProcessHelper", "Post-process helper",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);
  if (!DebugABICalls)
    inlineCilkFunctions(F);
}

LoopOutlineProcessor *CilkABI::getLoopOutlineProcessor(
    const TapirLoopInfo *TL) const {
  if (UseRuntimeCilkFor)
    return new RuntimeCilkFor(M);
  return nullptr;
}
