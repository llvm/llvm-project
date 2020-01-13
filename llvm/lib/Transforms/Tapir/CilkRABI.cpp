//===- CilkRABI.cpp - Interface to the CilkR runtime system ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CilkR ABI to converts Tapir instructions to calls
// into the CilkR runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkRABI.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilkrabi"

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

CilkRABI::CilkRABI(Module &M) : TapirTarget(M) {
  LLVMContext &C = M.getContext();
  Type *VoidPtrTy = Type::getInt8PtrTy(C);
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *Int16Ty = Type::getInt16Ty(C);

  // Get or create local definitions of Cilk RTS structure types.
  StackFrameTy = StructType::lookupOrCreate(C, "struct.__cilkrts_stack_frame");
  WorkerTy = StructType::lookupOrCreate(C, "struct.__cilkrts_worker");

  if (StackFrameTy->isOpaque())
    StackFrameTy->setBody(Int32Ty, // flags
                          PointerType::getUnqual(StackFrameTy), // call_parent
                          PointerType::getUnqual(WorkerTy), // worker
                          // VoidPtrTy, // except_data
                          ArrayType::get(VoidPtrTy, 5), // ctx
                          Int32Ty, // mxcsr
                          Int16Ty, // fpcsr
                          Int16Ty, // reserved
                          Int32Ty // magic
                          );
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  if (WorkerTy->isOpaque())
    WorkerTy->setBody(PointerType::getUnqual(StackFramePtrTy), // tail
                      PointerType::getUnqual(StackFramePtrTy), // head
                      PointerType::getUnqual(StackFramePtrTy), // exc
                      PointerType::getUnqual(StackFramePtrTy), // ltq_limit
                      Int32Ty, // self
                      VoidPtrTy, // g
                      VoidPtrTy, // l
                      StackFramePtrTy // current_stack_frame
                      // VoidPtrTy // reducer_map
                      );
}

// Accessors for opaque Cilk RTS functions

FunctionCallee CilkRABI::Get__cilkrts_get_nworkers() {
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

FunctionCallee CilkRABI::Get__cilkrts_leave_frame() {
  if (CilkRTSLeaveFrame)
    return CilkRTSLeaveFrame;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSLeaveFrame = M.getOrInsertFunction("__cilkrts_leave_frame", VoidTy,
                                            StackFramePtrTy);

  return CilkRTSLeaveFrame;
}

// FunctionCallee CilkRABI::Get__cilkrts_rethrow() {
//   if (CilkRTSRethrow)
//     return CilkRTSRethrow;

//   LLVMContext &C = M.getContext();
//   Type *VoidTy = Type::getVoidTy(C);
//   PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
//   CilkRTSRethrow = M.getOrInsertFunction("__cilkrts_rethrow", VoidTy,
//                                          StackFramePtrTy);

//   return CilkRTSRethrow;
// }

FunctionCallee CilkRABI::Get__cilkrts_sync() {
  if (CilkRTSSync)
    return CilkRTSSync;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  CilkRTSSync = M.getOrInsertFunction("__cilkrts_sync", VoidTy,
                                      StackFramePtrTy);

  return CilkRTSSync;
}

FunctionCallee CilkRABI::Get__cilkrts_get_tls_worker() {
  if (CilkRTSGetTLSWorker)
    return CilkRTSGetTLSWorker;

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  CilkRTSGetTLSWorker = M.getOrInsertFunction("__cilkrts_get_tls_worker",
                                              WorkerPtrTy);

  return CilkRTSGetTLSWorker;
}

/// Helper methods for storing to and loading from struct fields.
static Value *GEP(IRBuilder<> &B, Value *Base, int field) {
  // return B.CreateStructGEP(cast<PointerType>(Base->getType()),
  //                          Base, field);
  return B.CreateConstInBoundsGEP2_32(nullptr, Base, 0, field);
}

static unsigned GetAlignment(const DataLayout &DL, StructType *STy, int field) {
  return DL.getPrefTypeAlignment(STy->getElementType(field));
}

static void StoreSTyField(IRBuilder<> &B, const DataLayout &DL, StructType *STy,
                          Value *Val, Value *Dst, int field,
                          bool isVolatile = false,
                          AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  StoreInst *S = B.CreateAlignedStore(Val, GEP(B, Dst, field),
                                      GetAlignment(DL, STy, field), isVolatile);
  S->setOrdering(Ordering);
}

static Value *LoadSTyField(
    IRBuilder<> &B, const DataLayout &DL, StructType *STy, Value *Src,
    int field, bool isVolatile = false,
    AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  LoadInst *L =  B.CreateAlignedLoad(GEP(B, Src, field),
                                     GetAlignment(DL, STy, field), isVolatile);
  L->setOrdering(Ordering);
  return L;
}

/// Emit inline assembly code to save the floating point state, for x86 Only.
void CilkRABI::EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
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
CallInst *CilkRABI::EmitCilkSetJmp(IRBuilder<> &B, Value *SF) {
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
Function *CilkRABI::Get__cilkrts_pop_frame() {
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

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

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
///   StoreStore_fence();
///
///   *tail++ = parent;
///   w->tail = tail;
///
///   sf->flags |= CILK_FRAME_DETACHED;
/// }
Function *CilkRABI::Get__cilkrts_detach() {
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

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

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
///     SAVE_FLOAT_STATE(*sf);
///     if (!CILK_SETJMP(sf->ctx))
///       __cilkrts_sync(sf);
///     // else if (sf->flags & CILK_FRAME_EXCEPTING)
///     //   __cilkrts_rethrow(sf);
///   }
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
Function *CilkRABI::GetCilkSyncFn() {
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

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.sync.test", Fn);
  BasicBlock *SaveState = BasicBlock::Create(Ctx, "cilk.sync.savestate", Fn);
  BasicBlock *SyncCall = BasicBlock::Create(Ctx, "cilk.sync.runtimecall", Fn);
  // BasicBlock *Excepting = BasicBlock::Create(Ctx, "cilk.sync.excepting", Fn);
  // BasicBlock *Rethrow = BasicBlock::Create(Ctx, "cilk.sync.rethrow", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

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
    B.CreateCondBr(Unsynced, Exit, SaveState);
  }

  // SaveState
  {
    IRBuilder<> B(SaveState);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    // B.CreateCondBr(C, SyncCall, Excepting);
    B.CreateCondBr(C, SyncCall, Exit);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(&sf);
    B.CreateCall(CILKRTS_FUNC(sync), SF);
    B.CreateBr(Exit);
  }

  // // Excepting
  // {
  //   IRBuilder<> B(Excepting);
  //   if (Rethrow) {
  //     Value *Flags = LoadSTyField(B, DL, StackFrameTy, SF,
  //                                 StackFrameFields::flags,
  //                                 /*isVolatile=*/false,
  //                                 AtomicOrdering::Acquire);
  //     Flags = B.CreateAnd(Flags,
  //                         ConstantInt::get(Flags->getType(),
  //                                          CILK_FRAME_EXCEPTING));
  //     Value *Zero = ConstantInt::get(Flags->getType(), 0);
  //     Value *CanExcept = B.CreateICmpEQ(Flags, Zero);
  //     B.CreateCondBr(CanExcept, Exit, Rethrow);
  //   } else {
  //     B.CreateBr(Exit);
  //   }
  // }

  // // Rethrow
  // if (Rethrow) {
  //   IRBuilder<> B(Rethrow);
  //   B.CreateCall(CILKRTS_FUNC(rethrow), SF)->setDoesNotReturn();
  //   B.CreateUnreachable();
  // }

  // Exit
  {
    IRBuilder<> B(Exit);

    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);

  return Fn;
}

/// Get or create a LLVM function for __cilkrts_enter_frame.  It is equivalent
/// to the following C code:
///
/// void __cilkrts_enter_frame(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     // if (w == 0) { /* slow path, rare */
///     //     w = __cilkrts_bind_thread_1();
///     //     sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
///     // } else {
///         sf->flags = CILK_FRAME_VERSION;
///     // }
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
Function *CilkRABI::Get__cilkrts_enter_frame() {
  // Get or create the __cilkrts_enter_frame function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilk_enter_frame",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_enter_frame.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  // BasicBlock *SlowPath = BasicBlock::Create(Ctx, "slowpath", Fn);
  BasicBlock *FastPath = BasicBlock::Create(Ctx, "fastpath", Fn);
  BasicBlock *Cont = BasicBlock::Create(Ctx, "cont", Fn);

  PointerType *WorkerPtrTy = PointerType::getUnqual(WorkerTy);
  StructType *SFTy = StackFrameTy;

  // Block  (Entry)
  CallInst *W = nullptr;
  {
    IRBuilder<> B(Entry);
    // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker));

    // // if (w == 0)
    // Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    // B.CreateCondBr(Cond, SlowPath, FastPath);
    B.CreateBr(FastPath);
  }
  // // Block  (SlowPath)
  // CallInst *Wslow = nullptr;
  // {
  //   IRBuilder<> B(SlowPath);
  //   // w = __cilkrts_bind_thread_1();
  //   Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1));
  //   // sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
  //   Type *Ty = SFTy->getElementType(StackFrameFields::flags);
  //   StoreSTyField(B, DL, StackFrameTy,
  //                 ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
  //                 SF, StackFrameFields::flags, /*isVolatile=*/false,
  //                 AtomicOrdering::Release);
  //   B.CreateBr(Cont);
  // }
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
    // Value *Wfast = W;
    // PHINode *W  = B.CreatePHI(WorkerPtrTy, 2);
    // W->addIncoming(Wslow, SlowPath);
    // W->addIncoming(Wfast, FastPath);
    Value *Wkr = B.CreatePointerCast(W, WorkerPtrTy);
    // sf->call_parent = w->current_stack_frame;
    StoreSTyField(B, DL, StackFrameTy,
                  LoadSTyField(B, DL, WorkerTy, Wkr,
                               WorkerFields::current_stack_frame,
                               /*isVolatile=*/false,
                               AtomicOrdering::Acquire),
                  SF, StackFrameFields::call_parent, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // sf->worker = w;
    StoreSTyField(B, DL, StackFrameTy, Wkr, SF,
                  StackFrameFields::worker, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // w->current_stack_frame = sf;
    StoreSTyField(B, DL, WorkerTy, SF, Wkr,
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
/// void __cilkrts_enter_frame_fast(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     sf->flags = CILK_FRAME_VERSION;
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
Function *CilkRABI::Get__cilkrts_enter_frame_fast() {
  // Get or create the __cilkrts_enter_frame_fast function.
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *StackFramePtrTy = PointerType::getUnqual(StackFrameTy);
  Function *Fn = nullptr;
  if (GetOrCreateFunction(M, "__cilkrts_enter_frame_fast",
                          FunctionType::get(VoidTy, {StackFramePtrTy}, false),
                          Fn))
    return Fn;

  // Create the body of __cilkrts_enter_frame_fast.
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);

  IRBuilder<> B(Entry);
  Value *W;

  // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
  // if (fastCilk)
  //   W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast));
  // else
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker));

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

/// Get or create a LLVM function for __cilk_parent_epilogue.  It is equivalent
/// to the following C code:
///
/// void __cilk_parent_epilogue(__cilkrts_stack_frame *sf) {
///   __cilkrts_pop_frame(sf);
///   if (sf->flags != CILK_FRAME_VERSION)
///     __cilkrts_leave_frame(sf);
/// }
Function *CilkRABI::GetCilkParentEpilogueFn() {
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

  Function::arg_iterator Args = Fn->arg_begin();
  Value *SF = &*Args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn),
    *B1 = BasicBlock::Create(Ctx, "body", Fn),
    *Exit  = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

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
    B.CreateRetVoid();
  }

  Fn->setLinkage(Function::InternalLinkage);
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const StringRef stack_frame_name = "__cilkrts_sf";

/// Create the __cilkrts_stack_frame for the spawning function.
AllocaInst *CilkRABI::CreateStackFrame(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *SFTy = StackFrameTy;

  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *SF = B.CreateAlloca(SFTy, DL.getAllocaAddrSpace(),
                                  /*ArraySize*/nullptr,
                                  /*Name*/stack_frame_name);
  SF->setAlignment(8);

  return SF;
}

Value* CilkRABI::GetOrInitCilkStackFrame(Function &F, bool Helper = true) {
  if (DetachCtxToStackFrame.count(&F))
    return DetachCtxToStackFrame[&F];

  AllocaInst *SF = CreateStackFrame(F);
  DetachCtxToStackFrame[&F] = SF;
  BasicBlock::iterator InsertPt = ++SF->getIterator();
  IRBuilder<> IRB(&(F.getEntryBlock()), InsertPt);

  Value *Args[1] = { SF };
  if (Helper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast), Args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame), Args);

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogueFn(), Args, "");
  }
  return SF;
}

bool CilkRABI::makeFunctionDetachable(Function &Extracted) {
  /*
    __cilkrts_stack_frame sf;
    __cilkrts_enter_frame_fast(&sf);
    __cilkrts_detach();
    *x = f(y);
  */

  // const DataLayout& DL = M->getDataLayout();
  AllocaInst *SF = CreateStackFrame(Extracted);
  DetachCtxToStackFrame[&Extracted] = SF;
  assert(SF && "No Cilk stack frame for Cilk function.");
  Value *Args[1] = { SF };

  // Scan function to see if it detaches.
  LLVM_DEBUG({
      bool SimpleHelper = !canDetach(&Extracted);
      if (!SimpleHelper)
        dbgs() << "NOTE: Detachable helper function itself detaches.\n";
    });

  BasicBlock::iterator InsertPt = ++SF->getIterator();
  IRBuilder<> IRB(&(Extracted.getEntryBlock()), InsertPt);

  IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast), Args);

  // Call __cilkrts_detach
  {
    IRB.CreateCall(CILKRTS_FUNC(detach), Args);
  }

  EscapeEnumerator EE(Extracted, "cilkrabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogueFn(), Args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      // TODO: Handle exceptions.
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
      AtExit->CreateCall(GetCilkParentEpilogueFn(), Args, "");
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
Value *CilkRABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
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

void CilkRABI::lowerSync(SyncInst &SI) {
  Function &Fn = *(SI.getParent()->getParent());

  Value *SF = GetOrInitCilkStackFrame(Fn, /*Helper*/false);
  Value *Args[] = { SF };
  assert(Args[0] && "sync used in function without frame!");
  CallInst *CI = CallInst::Create(GetCilkSyncFn(), Args, "",
                                  /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
  // Mark this function as stealable.
  Fn.addFnAttr(Attribute::Stealable);
}

void CilkRABI::processOutlinedTask(Function &F) {
  makeFunctionDetachable(F);
}

void CilkRABI::processSpawner(Function &F) {
  GetOrInitCilkStackFrame(F, /*Helper=*/false);

  // Mark this function as stealable.
  F.addFnAttr(Attribute::Stealable);
}

void CilkRABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;

  Function &F = *ReplCall->getParent()->getParent();
  Value *SF = DetachCtxToStackFrame[&F];
  assert(SF && "No frame found for spawning task");

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

void CilkRABI::preProcessFunction(Function &F, TaskInfo &TI,
                                  bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  if (F.getName() == "main")
    F.setName("cilk_main");
}

void CilkRABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any postprocessing when outlining Tapir loops.
    return;

  inlineCilkFunctions(F);
}

void CilkRABI::postProcessHelper(Function &F) {
  inlineCilkFunctions(F);
}
