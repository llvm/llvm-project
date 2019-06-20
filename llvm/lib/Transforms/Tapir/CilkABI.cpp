//===- CilkABI.cpp - Lower Tapir into Cilk runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cilk ABI to converts Tapir instructions to calls
// into the Cilk runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/ADT/Statistic.h"
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
#include "llvm/ADT/Triple.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "cilkabi"

STATISTIC(LoopsConvertedToCilkABI,
          "Number of Tapir loops converted to use the Cilk ABI for loops");

static cl::opt<bool> fastCilk(
    "fast-cilk", cl::init(false), cl::Hidden,
    cl::desc("Attempt faster Cilk call implementation"));

static cl::opt<bool> DebugABICalls(
    "debug-abi-calls", cl::init(false), cl::Hidden,
    cl::desc("Insert ABI calls for debugging"));

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
Function *CilkABI::Get__cilkrts_get_nworkers() {
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
  CilkRTSGetNworkers = cast<Function>(
      M.getOrInsertFunction("__cilkrts_get_nworkers", FTy, AL));
  return CilkRTSGetNworkers;
}

Function *CilkABI::Get__cilkrts_init() {
  if (CilkRTSInit)
    return CilkRTSInit;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  CilkRTSInit = cast<Function>(
      M.getOrInsertFunction("__cilkrts_init", VoidTy));

  return CilkRTSInit;
}

Function *CilkABI::Get__cilkrts_leave_frame() {
  if (CilkRTSLeaveFrame)
    return CilkRTSLeaveFrame;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSLeaveFrame = cast<Function>(
      M.getOrInsertFunction("__cilkrts_leave_frame", VoidTy, StackFramePtrTy));

  return CilkRTSLeaveFrame;
}

Function *CilkABI::Get__cilkrts_rethrow() {
  if (CilkRTSRethrow)
    return CilkRTSRethrow;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSRethrow = cast<Function>(
      M.getOrInsertFunction("__cilkrts_rethrow", VoidTy, StackFramePtrTy));

  return CilkRTSRethrow;
}

Function *CilkABI::Get__cilkrts_sync() {
  if (CilkRTSSync)
    return CilkRTSSync;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSSync = cast<Function>(
      M.getOrInsertFunction("__cilkrts_sync", VoidTy, StackFramePtrTy));

  return CilkRTSSync;
}

Function *CilkABI::Get__cilkrts_get_tls_worker() {
  if (CilkRTSGetTLSWorker)
    return CilkRTSGetTLSWorker;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSGetTLSWorker = cast<Function>(
      M.getOrInsertFunction("__cilkrts_get_tls_worker", WorkerPtrTy));

  return CilkRTSGetTLSWorker;
}

Function *CilkABI::Get__cilkrts_get_tls_worker_fast() {
  if (CilkRTSGetTLSWorkerFast)
    return CilkRTSGetTLSWorkerFast;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSGetTLSWorkerFast = cast<Function>(
      M.getOrInsertFunction("__cilkrts_get_tls_worker_fast", WorkerPtrTy));

  return CilkRTSGetTLSWorkerFast;
}

Function *CilkABI::Get__cilkrts_bind_thread_1() {
  if (CilkRTSBindThread1)
    return CilkRTSBindThread1;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSBindThread1 = cast<Function>(
      M.getOrInsertFunction("__cilkrts_bind_thread_1", WorkerPtrTy));

  return CilkRTSBindThread1;
}

Function *CilkABILoopSpawning::Get__cilkrts_cilk_for_32() {
  if (CilkRTSCilkFor32)
    return CilkRTSCilkFor32;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *VoidPtrTy = Type::getInt8PtrTy(C);
  Type *CountTy = Type::getInt32Ty(C);
  FunctionType *BodyTy = FunctionType::get(VoidTy,
                                           {VoidPtrTy, CountTy, CountTy},
                                           false);
  FunctionType *FTy =
    FunctionType::get(VoidTy,
                      {PointerType::getUnqual(BodyTy), VoidPtrTy, CountTy,
                       Type::getInt32Ty(C)}, false);
  CilkRTSCilkFor32 = cast<Function>(
      M.getOrInsertFunction("__cilkrts_cilk_for_32", FTy));

  return CilkRTSCilkFor32;
}

Function *CilkABILoopSpawning::Get__cilkrts_cilk_for_64() {
  if (CilkRTSCilkFor64)
    return CilkRTSCilkFor64;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *VoidPtrTy = Type::getInt8PtrTy(C);
  Type *CountTy = Type::getInt64Ty(C);
  FunctionType *BodyTy = FunctionType::get(VoidTy,
                                           {VoidPtrTy, CountTy, CountTy},
                                           false);
  FunctionType *FTy =
    FunctionType::get(VoidTy,
                      {PointerType::getUnqual(BodyTy), VoidPtrTy, CountTy,
                       Type::getInt32Ty(C)}, false);
  CilkRTSCilkFor64 = cast<Function>(
      M.getOrInsertFunction("__cilkrts_cilk_for_64", FTy));

  return CilkRTSCilkFor64;
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
  Fn = cast<Function>(M.getOrInsertFunction(FnName, FTy));

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
  makeFunctionDetachable(F, false);
}

void CilkABI::processSpawner(Function &F) {
  GetOrInitCilkStackFrame(F, /*Helper=*/false, false);

  // Mark this function as stealable.
  F.addFnAttr(Attribute::Stealable);
}

void CilkABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;

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

void CilkABI::preProcessFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "CilkABI processing function " << F.getName() << "\n");
  if (fastCilk && F.getName() == "main") {
    IRBuilder<> B(F.getEntryBlock().getTerminator());
    B.CreateCall(CILKRTS_FUNC(init));
  }
}

void CilkABI::postProcessFunction(Function &F) {
  if (!DebugABICalls)
    inlineCilkFunctions(F);
}

void CilkABI::postProcessHelper(Function &F) {
  if (!DebugABICalls)
    inlineCilkFunctions(F);
}


/// Replace the latch of the loop to check that IV is always less than or equal
/// to the limit.
///
/// This method assumes that the loop has a single loop latch.
Value *CilkABILoopSpawning::canonicalizeLoopLatch(PHINode *IV, Value *Limit) {
  Loop *L = OrigLoop;

  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition = Builder.CreateICmpULT(
      Builder.CreateAdd(IV, ConstantInt::get(IV->getType(), 1)), Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  Builder.CreateCondBr(NewCondition, Header, ExitBlock);

  // Erase the old conditional branch.
  Value *OldCond = LatchBr->getCondition();
  LatchBr->eraseFromParent();
  if (!OldCond->hasNUsesOrMore(1))
    if (Instruction *OldCondInst = dyn_cast<Instruction>(OldCond))
      OldCondInst->eraseFromParent();

  return NewCondition;
}

/// Top-level call to convert a Tapir loop to be processed using an appropriate
/// Cilk ABI call.
bool CilkABILoopSpawning::processLoop() {
  Loop *L = OrigLoop;

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();

  LLVM_DEBUG({
      LoopBlocksDFS DFS(L);
      DFS.perform(LI);
      dbgs() << "Blocks in loop (from DFS):\n";
      for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO()))
        dbgs() << *BB;
    });

  using namespace ore;

  // Check the exit blocks of the loop.
  if (!ExitBlock) {
    LLVM_DEBUG(dbgs() <<
               "LS loop does not contain valid exit block after latch.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "InvalidLatchExit",
                                        L->getStartLoc(),
                                        Header)
             << "invalid latch exit");
    return false;
  }

  // Get the unwind destination of the detach in the header.
  BasicBlock *DetachUnwind = nullptr;
  Value *SyncRegion = nullptr;
  {
    DetachInst *DI = cast<DetachInst>(Header->getTerminator());
    SyncRegion = DI->getSyncRegion();
    if (DI->hasUnwindDest())
      DetachUnwind = DI->getUnwindDest();
  }
  // Get special exits from this loop.
  SmallVector<BasicBlock *, 4> EHExits;
  getEHExits(L, ExitBlock, DetachUnwind, SyncRegion, EHExits);

  // Check the exit blocks of the loop.
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  SmallPtrSet<BasicBlock *, 4> HandledExits;
  for (BasicBlock *BB : EHExits)
    HandledExits.insert(BB);
  for (BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    if (Exit == DetachUnwind) continue;
    if (!HandledExits.count(Exit)) {
      LLVM_DEBUG(dbgs() << "LS loop contains a bad exit block " << *Exit);
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "BadExit",
                                          L->getStartLoc(),
                                          Header)
               << "bad exit block found");
      return false;
    }
  }

  Function *F = Header->getParent();
  Module *M = F->getParent();

  LLVM_DEBUG(dbgs() << "LS loop header:" << *Header);
  LLVM_DEBUG(dbgs() << "LS loop latch:" << *Latch);
  LLVM_DEBUG(dbgs() <<
             "LS SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");

  /// Get loop limit.
  const SCEV *BETC = SE.getExitCount(L, Latch);
  const SCEV *Limit = SE.getAddExpr(BETC, SE.getOne(BETC->getType()));
  LLVM_DEBUG(dbgs() << "LS Loop limit: " << *Limit << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getExitCount(L, Latch);
  // LLVM_DEBUG(dbgs() << "LS predicated loop limit: " << *PLimit << "\n");
  // emitAnalysis(LoopSpawningReport()
  //              << "computed loop limit " << *Limit << "\n");
  if (SE.getCouldNotCompute() == Limit) {
    LLVM_DEBUG(dbgs() << "SE could not compute loop limit.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "UnknownLoopLimit",
                                        L->getStartLoc(),
                                        Header)
             << "could not compute limit");
    return false;
  }
  // ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "LoopLimit", L->getStartLoc(),
  //                                     Header)
  //          << "loop limit: " << NV("Limit", Limit));
  /// Determine the type of the canonical IV.
  Type *CanonicalIVTy = Limit->getType();
  {
    const DataLayout &DL = M->getDataLayout();
    for (PHINode &PN : Header->phis()) {
      if (PN.getType()->isFloatingPointTy()) continue;
      CanonicalIVTy = getWiderType(DL, PN.getType(), CanonicalIVTy);
    }
    Limit = SE.getNoopOrAnyExtend(Limit, CanonicalIVTy);
  }
  /// Clean up the loop's induction variables.
  PHINode *CanonicalIV = canonicalizeIVs(CanonicalIVTy);
  if (!CanonicalIV) {
    LLVM_DEBUG(dbgs() << "Could not get canonical IV.\n");
    // emitAnalysis(LoopSpawningReport()
    //              << "Could not get a canonical IV.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoCanonicalIV",
                                        L->getStartLoc(),
                                        Header)
             << "could not find or create canonical IV");
    return false;
  }
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

  // Remove all IV's other can CanonicalIV.
  // First, check that we can do this.
  bool CanRemoveIVs = true;
  for (PHINode &PN : Header->phis()) {
    if (CanonicalIV == &PN) continue;
    const SCEV *S = SE.getSCEV(&PN);
    if (SE.getCouldNotCompute() == S) {
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Could not compute the scalar evolution.\n");
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoSCEV", &PN)
               << "could not compute scalar evolution of "
               << NV("PHINode", &PN));
      CanRemoveIVs = false;
    }
  }

  if (!CanRemoveIVs) {
    LLVM_DEBUG(dbgs() << "Could not compute scalar evolutions for all IV's.\n");
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  // We now have everything we need to extract the loop.  It's time to
  // do some surgery.

  SCEVExpander Exp(SE, M->getDataLayout(), "ls");

  // Remove the IV's (other than CanonicalIV) and replace them with
  // their stronger forms.
  //
  // TODO?: We can probably adapt this process such that we don't require all
  // IV's to be canonical.
  {
    SmallVector<PHINode*, 8> IVsToRemove;
    for (PHINode &PN : Header->phis()) {
      if (&PN == CanonicalIV) continue;
      const SCEV *S = SE.getSCEV(&PN);
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
      PN.replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(&PN);
    }
    for (PHINode *PN : IVsToRemove)
      PN->eraseFromParent();
  }

  // All remaining IV's should be canonical.  Collect them.
  //
  // TODO?: We can probably adapt this process such that we don't require all
  // IV's to be canonical.
  SmallVector<PHINode*, 8> IVs;
  bool AllCanonical = true;
  for (PHINode &PN : Header->phis()) {
    LLVM_DEBUG({
        const SCEVAddRecExpr *PNSCEV =
          dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(&PN));
        assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr");
        assert(PNSCEV->getStart()->isZero() &&
               "PHINode SCEV does not start at 0");
        dbgs() << "LS step recurrence for SCEV " << *PNSCEV << " is "
               << *(PNSCEV->getStepRecurrence(SE)) << "\n";
        assert(PNSCEV->getStepRecurrence(SE)->isOne() &&
               "PHINode SCEV step is not 1");
      });
    if (ConstantInt *C =
        dyn_cast<ConstantInt>(PN.getIncomingValueForBlock(Preheader))) {
      if (C->isZero()) {
        LLVM_DEBUG({
            if (&PN != CanonicalIV) {
              const SCEVAddRecExpr *PNSCEV =
                dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(&PN));
              dbgs() <<
                "Saving the canonical IV " << PN << " (" << *PNSCEV << ")\n";
            }
          });
        if (&PN != CanonicalIV)
          ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "SaveIV", &PN)
                   << "saving the canonical the IV "
                   << NV("PHINode", &PN));
        IVs.push_back(&PN);
      }
    } else {
      AllCanonical = false;
      LLVM_DEBUG(dbgs() <<
                 "Remaining non-canonical PHI Node found: " << PN << "\n");
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Found a remaining non-canonical IV.\n");
      ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "NonCanonicalIV", &PN)
               << "found a remaining noncanonical IV");
    }
  }
  if (!AllCanonical)
    return false;

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      Preheader->getTerminator());
  LLVM_DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

  // Canonicalize the loop latch.
  Instruction *NewCond =
    cast<Instruction>(canonicalizeLoopLatch(CanonicalIV, LimitVar));

  /// Clone the loop into a new function.

  // Get the inputs and outputs for the Loop blocks.
  SetVector<Value *> Inputs, Outputs;
  SetVector<Value *> BodyInputs, BodyOutputs;
  ValueToValueMapTy VMap, InputMap;
  std::vector<BasicBlock *> LoopBlocks;
  AllocaInst *Closure;
  SetVector<Value *> HelperInputs;
  SmallVector<Instruction *, 8> StructInputLoads;
  bool NeedSeparateEndArg;

  // Get the correct CilkForABI call.
  Function *CilkForABI;
  if (LimitVar->getType()->isIntegerTy(32))
    CilkForABI = CILKRTS_FUNC(cilk_for_32);
  else if (LimitVar->getType()->isIntegerTy(64))
    CilkForABI = CILKRTS_FUNC(cilk_for_64);
  else
    llvm_unreachable("Invalid integer type for Tapir loop limit.");

  // Get the sync region containing this Tapir loop.
  const Instruction *InputSyncRegion;
  {
    const DetachInst *DI = cast<DetachInst>(Header->getTerminator());
    InputSyncRegion = cast<Instruction>(DI->getSyncRegion());
  }

  // Add start iteration, end iteration, and grainsize to inputs.
  {
    LoopBlocks = L->getBlocks();

    // Add unreachable and exception-handling exits to the set of loop blocks to
    // clone.
    LLVM_DEBUG({
        dbgs() << "Handled exits of loop:";
        for (BasicBlock *HE : HandledExits)
          dbgs() << *HE;
        dbgs() << "\n";
      });
    for (BasicBlock *HE : HandledExits)
      LoopBlocks.push_back(HE);

    // Get the inputs and outputs for the loop body.
    {
      SmallPtrSet<BasicBlock *, 32> Blocks;
      for (BasicBlock *BB : LoopBlocks)
        Blocks.insert(BB);
      findInputsOutputs(Blocks, BodyInputs, BodyOutputs, &HandledExits, DT);
    }

    // Add argument for start of CanonicalIV.
    LLVM_DEBUG({
        Value *CanonicalIVInput =
          CanonicalIV->getIncomingValueForBlock(Preheader);
        // CanonicalIVInput should be the constant 0.
        assert(isa<Constant>(CanonicalIVInput) &&
               "Input to canonical IV from preheader is not constant.");
      });
    Argument *StartArg = new Argument(CanonicalIV->getType(),
                                      CanonicalIV->getName()+".start");
    Inputs.insert(StartArg);
    InputMap[CanonicalIV] = StartArg;

    // Determine if the end argument and loop limit are distinct entities, i.e.,
    // because the loop limit is a constant (and the end argument is guaranteed
    // to be a parameter), or because the loop limit is separately used in the
    // loop body.
    NeedSeparateEndArg = (isa<Constant>(LimitVar) ||
                          isUsedInLoopBody(LimitVar, LoopBlocks, NewCond));
    // Add argument for end.
    if (NeedSeparateEndArg) {
      Argument *EndArg = new Argument(LimitVar->getType(), "end");
      Inputs.insert(EndArg);
      InputMap[LimitVar] = EndArg;
    } else {
      Inputs.insert(LimitVar);
      InputMap[LimitVar] = LimitVar;
    }

    // Put all of the inputs together, and clear redundant inputs from
    // the set for the loop body.
    SmallVector<Value *, 8> BodyInputsToRemove;
    SmallVector<Value *, 8> StructInputs;
    SmallVector<Type *, 8> StructIT;
    for (Value *V : BodyInputs)
      if (V == InputSyncRegion)
        BodyInputsToRemove.push_back(V);
      else if (!Inputs.count(V)) {
        Inputs.insert(V);
        StructInputs.push_back(V);
        StructIT.push_back(V->getType());
      } else
        BodyInputsToRemove.push_back(V);
    for (Value *V : BodyInputsToRemove)
      BodyInputs.remove(V);
    LLVM_DEBUG({
        for (Value *V : BodyInputs)
          dbgs() << "Remaining body input: " << *V << "\n";
      });
    for (Value *V : BodyOutputs)
      dbgs() << "EL output: " << *V << "\n";
    assert(BodyOutputs.empty() &&
           "All results from parallel loop should be passed by memory already.");

    StructType *ST = StructType::get(F->getContext(), StructIT);
    LLVM_DEBUG(dbgs() << "Closure struct type " << *ST << "\n");
    {
      BasicBlock *AllocaInsertBlk = GetDetachedCtx(Preheader);
      IRBuilder<> Builder(&*AllocaInsertBlk->getFirstInsertionPt());
      Closure = Builder.CreateAlloca(ST);
    }
    IRBuilder<> B(Preheader->getTerminator());
    IRBuilder<> B2(Header->getFirstNonPHIOrDbgOrLifetime());
    for (unsigned i = 0; i < StructInputs.size(); ++i) {
      B.CreateStore(StructInputs[i], B.CreateConstGEP2_32(ST, Closure, 0, i));
    }
    for (unsigned i = 0; i < StructInputs.size(); ++i) {
      auto STGEP = cast<Instruction>(B2.CreateConstGEP2_32(ST, Closure, 0, i));
      auto STLoad = B2.CreateLoad(STGEP);
      // Save these two instructions, so they can be moved later.
      StructInputLoads.push_back(STGEP);
      StructInputLoads.push_back(STLoad);

      // Update all uses of the struct inputs in the loop body.
      auto UI = StructInputs[i]->use_begin(), E = StructInputs[i]->use_end();
      for (; UI != E;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (!Usr || !L->contains(Usr->getParent()))
          continue;
        U.set(STLoad);
      }
    }
    LLVM_DEBUG(dbgs() << "New preheader:" << *Preheader << "\n");
    LLVM_DEBUG(dbgs() << "New header:" << *Header << "\n");
    HelperInputs.insert(Closure);
    HelperInputs.insert(StartArg);
    HelperInputs.insert(cast<Value>(InputMap[LimitVar]));
  }
  LLVM_DEBUG({
      for (Value *V : Inputs)
        dbgs() << "EL input: " << *V << "\n";
      for (Value *V : Outputs)
        dbgs() << "EL output: " << *V << "\n";
      for (Value *V : HelperInputs)
        dbgs() << "Helper input: " << *V << "\n";
    });

  Function *Helper;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.

    // LowerDbgDeclare(*(Header->getParent()));

    Helper = CreateHelper(HelperInputs, Outputs, LoopBlocks,
                          Header, Preheader, ExitBlock,
                          VMap, M,
                          F->getSubprogram() != nullptr, Returns, ".ls",
                          nullptr, &HandledExits, &HandledExits, DetachUnwind,
                          InputSyncRegion, nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning loop.");

    // Use a fast calling convention for the helper.
    //Helper->setCallingConv(CallingConv::Fast);
    // Helper->setCallingConv(Header->getParent()->getCallingConv());
  }

  // Add a sync to the helper's return.
  BasicBlock *HelperHeader = cast<BasicBlock>(VMap[Header]);
  {
    assert(isa<ReturnInst>(cast<BasicBlock>(VMap[ExitBlock])->getTerminator()));
    SyncInst *NewSync =
      insertSyncBeforeEscape(cast<BasicBlock>(VMap[ExitBlock]),
                             cast<Instruction>(VMap[InputSyncRegion]), DT, LI);
    // Set debug info of new sync to match that of terminator of the header of
    // the cloned loop.
    NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());
  }

  // Add syncs to the helper's resume blocks.
  if (DetachUnwind) {
    assert(
        isa<ResumeInst>(cast<BasicBlock>(VMap[DetachUnwind])->getTerminator()));
    SyncInst *NewSync =
      insertSyncBeforeEscape(cast<BasicBlock>(VMap[DetachUnwind]),
                             cast<Instruction>(VMap[InputSyncRegion]), DT, LI);
    NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());

  }
  for (BasicBlock *BB : HandledExits) {
    if (!isDetachedRethrow(BB->getTerminator(), InputSyncRegion)) continue;
    assert(isa<ResumeInst>(cast<BasicBlock>(VMap[BB])->getTerminator()));
    SyncInst *NewSync =
      insertSyncBeforeEscape(cast<BasicBlock>(VMap[BB]),
                             cast<Instruction>(VMap[InputSyncRegion]), DT, LI);
    NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());
  }

  BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
  PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

  // Rewrite the cloned IV's to start at the start iteration argument.
  {
    // Rewrite clone of canonical IV to start at the start iteration
    // argument.
    Argument *NewCanonicalIVStart = cast<Argument>(VMap[InputMap[CanonicalIV]]);
    {
      int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned canonical IV does not inherit a constant value from cloned preheader.");
      NewCanonicalIV->setIncomingValue(NewPreheaderIdx, NewCanonicalIVStart);
    }

    // Rewrite other cloned IV's to start at their value at the start
    // iteration.
    const SCEV *StartIterSCEV = SE.getSCEV(NewCanonicalIVStart);
    LLVM_DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
    for (PHINode *IV : IVs) {
      if (CanonicalIV == IV) continue;

      // Get the value of the IV at the start iteration.
      LLVM_DEBUG(dbgs() << "IV " << *IV);
      const SCEV *IVSCEV = SE.getSCEV(IV);
      LLVM_DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
      const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
      const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
      LLVM_DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
            " to " << *IVAtIter << "\n");

      // NOTE: Expanded code should not refer to other IV's.
      Value *IVStart = Exp.expandCodeFor(IVAtIter, IVAtIter->getType(),
                                         NewPreheader->getTerminator());


      // Set the value that the cloned IV inherits from the cloned preheader.
      PHINode *NewIV = cast<PHINode>(VMap[IV]);
      int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned IV does not inherit a constant value from cloned preheader.");
      NewIV->setIncomingValue(NewPreheaderIdx, IVStart);
    }

    // Remap the newly added instructions in the new preheader to use
    // values local to the helper.
    for (Instruction &II : *NewPreheader)
      RemapInstruction(&II, VMap, RF_IgnoreMissingLocals,
                       /*TypeMapper=*/nullptr, /*Materializer=*/nullptr);
  }

  // If the loop limit and end iteration are distinct, then rewrite the loop
  // latch condition to use the end-iteration argument.
  if (NeedSeparateEndArg) {
    CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
    IRBuilder<> Builder(HelperCond);
    Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                 VMap[InputMap[LimitVar]]);
    HelperCond->replaceAllUsesWith(NewHelperCond);
    HelperCond->eraseFromParent();
  }

  BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
  // Move the loads from the Helper's struct input to the Helper's entry block.
  for (Instruction *STLoad : StructInputLoads) {
    Instruction *NewSTLoad = cast<Instruction>(VMap[STLoad]);
    NewSTLoad->moveBefore(NewPreheader->getTerminator());
  }

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Update allocas in cloned loop body.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (BasicBlock *Pred : predecessors(Latch)) {
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (L->contains(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }
    // The cloned loop should be serialized by this point.
    BasicBlock *ClonedLoopBodyEntry =
      cast<BasicBlock>(VMap[Header])->getSingleSuccessor();
    assert(ClonedLoopBodyEntry &&
           "Head of cloned loop body has multiple successors.");
    bool ContainsDynamicAllocas =
      MoveStaticAllocasInBlock(&Helper->getEntryBlock(), ClonedLoopBodyEntry,
                               ReattachPoints);

    // If the cloned loop contained dynamic alloca instructions, wrap the cloned
    // loop with llvm.stacksave/llvm.stackrestore intrinsics.
    if (ContainsDynamicAllocas) {
      Module *M = Helper->getParent();
      // Get the two intrinsics we care about.
      Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
      Function *StackRestore =
        Intrinsic::getDeclaration(M,Intrinsic::stackrestore);

      // Insert the llvm.stacksave.
      CallInst *SavedPtr =
        IRBuilder<>(&*ClonedLoopBodyEntry, ClonedLoopBodyEntry->begin())
        .CreateCall(StackSave, {}, "savedstack");

      // Insert a call to llvm.stackrestore before the reattaches in the
      // original Tapir loop.
      for (Instruction *ExitPoint : ReattachPoints)
        IRBuilder<>(ExitPoint).CreateCall(StackRestore, SavedPtr);
    }
  }

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(F, HelperInputs, VMap,
                          Preheader->getTerminator(), AC, DT);

  // Add call to new helper function in original function.
  {
    LLVM_DEBUG(dbgs() << "CilkForABI function " << *CilkForABI << "\n");
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs;
    IRBuilder<> Builder(Preheader->getTerminator());
    // Add a pointer to the helper function.
    {
      Value *HelperPtr = Builder.CreatePointerCast(
          Helper, CilkForABI->getFunctionType()->getParamType(0));
      TopCallArgs.push_back(HelperPtr);
    }
    // Add a pointer to the context.
    {
      Value *ClosurePtr = Builder.CreatePointerCast(
          Closure, CilkForABI->getFunctionType()->getParamType(1));
      TopCallArgs.push_back(ClosurePtr);
    }
    // Add loop limit.
    TopCallArgs.push_back(LimitVar);
    // Add grainsize.
    if (!SpecifiedGrainsize)
      TopCallArgs.push_back(
          ConstantInt::get(IntegerType::getInt32Ty(F->getContext()), 0));
    else
      TopCallArgs.push_back(
          ConstantInt::get(IntegerType::getInt32Ty(F->getContext()),
                           SpecifiedGrainsize));
    LLVM_DEBUG({
        dbgs() << "TopCallArgs:\n";
        for (Value *Arg : TopCallArgs)
          dbgs() << "\t" << *Arg << "\n";
      });
    // Create call instruction.
    if (!DetachUnwind) {
      IRBuilder<> Builder(Preheader->getTerminator());
      CallInst *TopCall = Builder.CreateCall(
          CilkForABI, ArrayRef<Value *>(TopCallArgs));
      // Use a fast calling convention for the helper.
      TopCall->setCallingConv(CallingConv::Fast);
      // TopCall->setCallingConv(Helper->getCallingConv());
      TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
      // // Update CG graph with the call we just added.
      // CG[F]->addCalledFunction(TopCall, CG[Helper]);
    } else {
      BasicBlock *CallDest = SplitBlock(Preheader, Preheader->getTerminator(),
                                        DT, LI);
      InvokeInst *TopCall = InvokeInst::Create(CilkForABI, CallDest, DetachUnwind,
                                               ArrayRef<Value *>(TopCallArgs));
      // Update PHI nodes in DetachUnwind
      for (PHINode &P : DetachUnwind->phis()) {
        int j = P.getBasicBlockIndex(Header);
        assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
        LLVM_DEBUG({
            if (Instruction *I = dyn_cast<Instruction>(P.getIncomingValue(j)))
              assert(I->getParent() != Header &&
                     "DetachUnwind PHI node uses value from header!");
          });
        P.addIncoming(P.getIncomingValue(j), Preheader);
      }
      // Update the dominator tree by informing it about the new edge from the
      // preheader to the detach unwind destination.
      if (DT)
        DT->insertEdge(Preheader, DetachUnwind);
      ReplaceInstWithInst(Preheader->getTerminator(), TopCall);
      // Use a fast calling convention for the helper.
      TopCall->setCallingConv(CallingConv::Fast);
      // TopCall->setCallingConv(Helper->getCallingConv());
      TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
      // // Update CG graph with the call we just added.
      // CG[F]->addCalledFunction(TopCall, CG[Helper]);
    }
  }
  // Remove sync of loop in parent.
  {
    // Get the sync region for this loop's detached iterations.
    DetachInst *HeadDetach = cast<DetachInst>(Header->getTerminator());
    Value *SyncRegion = HeadDetach->getSyncRegion();
    // Check the Tapir instructions contained in this sync region.  Look for a
    // single sync instruction among those Tapir instructions.  Meanwhile,
    // verify that the only detach instruction in this sync region is the detach
    // in theloop header.  If these conditions are met, then we assume that the
    // sync applies to this loop.  Otherwise, something more complicated is
    // going on, and we give up.
    SyncInst *LoopSync = nullptr;
    bool SingleSyncJustForLoop = true;
    for (User *U : SyncRegion->users()) {
      // Skip the detach in the loop header.
      if (HeadDetach == U) continue;
      // Remember the first sync instruction we find.  If we find multiple sync
      // instructions, then something nontrivial is going on.
      if (SyncInst *SI = dyn_cast<SyncInst>(U)) {
        if (!LoopSync)
          LoopSync = SI;
        else
          SingleSyncJustForLoop = false;
      }
      // If we find a detach instruction that is not the loop header's, then
      // something nontrivial is going on.
      if (isa<DetachInst>(U))
        SingleSyncJustForLoop = false;
    }
    if (LoopSync && SingleSyncJustForLoop)
      // Replace the sync with a branch.
      ReplaceInstWithInst(LoopSync,
                          BranchInst::Create(LoopSync->getSuccessor(0)));
    else if (!LoopSync)
      LLVM_DEBUG(dbgs() << "No sync found for this loop.\n");
    else
      LLVM_DEBUG(dbgs() <<
                 "No single sync found that only affects this loop.\n");
  }

  ++LoopsConvertedToCilkABI;

  unlinkLoop();

  return Helper;
}
