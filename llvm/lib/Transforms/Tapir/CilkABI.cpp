//===- CilkABI.cpp - Lower Tapir into Cilk runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CilkABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Cilk
// runtime system.  This interface does the low-level dirty work of passes
// such as LowerToCilk.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilkabi"

/// Helper typedefs for cilk struct TypeBuilders.
typedef llvm::TypeBuilder<__cilkrts_stack_frame, false> StackFrameBuilder;
typedef llvm::TypeBuilder<__cilkrts_worker, false> WorkerBuilder;
typedef llvm::TypeBuilder<__cilkrts_pedigree, false> PedigreeBuilder;

/// Helper methods for storing to and loading from struct fields.
static Value *GEP(IRBuilder<> &B, Value *Base, int field) {
  // return B.CreateStructGEP(cast<PointerType>(Base->getType()),
  //                          Base, field);
  return B.CreateConstInBoundsGEP2_32(nullptr, Base, 0, field);
}

static void StoreField(IRBuilder<> &B, Value *Val, Value *Dst, int field,
                       bool isVolatile = false) {
  B.CreateStore(Val, GEP(B, Dst, field), isVolatile);
}

static Value *LoadField(IRBuilder<> &B, Value *Src, int field,
                        bool isVolatile = false) {
  return B.CreateLoad(GEP(B, Src, field), isVolatile);
}

/// \brief Emit inline assembly code to save the floating point
/// state, for x86 Only.
static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
  typedef void (AsmPrototype)(uint32_t*, uint16_t*);
  llvm::FunctionType *FTy =
    TypeBuilder<AsmPrototype, false>::get(B.getContext());

  Value *Asm = InlineAsm::get(FTy,
                              "stmxcsr $0\n\t" "fnstcw $1",
                              "*m,*m,~{dirflag},~{fpsr},~{flags}",
                              /*sideeffects*/ true);

  Value * args[2] = {
    GEP(B, SF, StackFrameBuilder::mxcsr),
    GEP(B, SF, StackFrameBuilder::fpcsr)
  };

  B.CreateCall(Asm, args);
}

/// \brief Helper to find a function with the given name, creating it if it
/// doesn't already exist. If the function needed to be created then return
/// false, signifying that the caller needs to add the function body.
template <typename T>
static bool GetOrCreateFunction(const char *FnName, Module& M,
                                Function *&Fn,
                                Function::LinkageTypes Linkage =
                                Function::InternalLinkage,
                                bool DoesNotThrow = true) {
  LLVMContext &Ctx = M.getContext();

  Fn = M.getFunction(FnName);

  // if the function already exists then let the
  // caller know that it is complete
  if (Fn)
    return true;

  // Otherwise we have to create it
  FunctionType *FTy = TypeBuilder<T, false>::get(Ctx);
  Fn = Function::Create(FTy, Linkage, FnName, &M);

  // Set nounwind if it does not throw.
  if (DoesNotThrow)
    Fn->setDoesNotThrow();

  // and let the caller know that the function is incomplete
  // and the body still needs to be added
  return false;
}

/// \brief Emit a call to the CILK_SETJMP function.
static CallInst *EmitCilkSetJmp(IRBuilder<> &B, Value *SF, Module& M) {
  LLVMContext &Ctx = M.getContext();

  // We always want to save the floating point state too
  EmitSaveFloatingPointState(B, SF);

  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int8PtrTy = Type::getInt8PtrTy(Ctx);

  // Get the buffer to store program state
  // Buffer is a void**.
  Value *Buf = GEP(B, SF, StackFrameBuilder::ctx);

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

/// \brief Get or create a LLVM function for __cilkrts_pop_frame.
/// It is equivalent to the following C code
///
/// __cilkrts_pop_frame(__cilkrts_stack_frame *sf) {
///   sf->worker->current_stack_frame = sf->call_parent;
///   sf->call_parent = 0;
/// }
static Function *Get__cilkrts_pop_frame(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_pop_frame", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // sf->worker->current_stack_frame = sf.call_parent;
  StoreField(B,
             LoadField(B, SF, StackFrameBuilder::call_parent,
                       /*isVolatile=*/true),
             LoadField(B, SF, StackFrameBuilder::worker,
                       /*isVolatile=*/true),
             WorkerBuilder::current_stack_frame,
             /*isVolatile=*/true);

  // sf->call_parent = 0;
  StoreField(B,
             Constant::getNullValue(
                 TypeBuilder<__cilkrts_stack_frame*, false>::get(Ctx)),
             SF, StackFrameBuilder::call_parent, /*isVolatile=*/true);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_detach.
/// It is equivalent to the following C code
///
/// void __cilkrts_detach(struct __cilkrts_stack_frame *sf) {
///   struct __cilkrts_worker *w = sf->worker;
///   struct __cilkrts_stack_frame *volatile *tail = w->tail;
///
///   sf->spawn_helper_pedigree = w->pedigree;
///   sf->call_parent->parent_pedigree = w->pedigree;
///
///   w->pedigree.rank = 0;
///   w->pedigree.next = &sf->spawn_helper_pedigree;
///
///   *tail++ = sf->call_parent;
///   w->tail = tail;
///
///   sf->flags |= CILK_FRAME_DETACHED;
/// }
static Function *Get__cilkrts_detach(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_detach", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // struct __cilkrts_worker *w = sf->worker;
  Value *W = LoadField(B, SF, StackFrameBuilder::worker,
                       /*isVolatile=*/true);

  // __cilkrts_stack_frame *volatile *tail = w->tail;
  Value *Tail = LoadField(B, W, WorkerBuilder::tail,
                          /*isVolatile=*/true);

  // sf->spawn_helper_pedigree = w->pedigree;
  StoreField(B,
             LoadField(B, W, WorkerBuilder::pedigree),
             SF, StackFrameBuilder::parent_pedigree);

  // sf->call_parent->parent_pedigree = w->pedigree;
  StoreField(B,
             LoadField(B, W, WorkerBuilder::pedigree),
             LoadField(B, SF, StackFrameBuilder::call_parent),
             StackFrameBuilder::parent_pedigree);

  // w->pedigree.rank = 0;
  {
    StructType *STy = PedigreeBuilder::get(Ctx);
    llvm::Type *Ty = STy->getElementType(PedigreeBuilder::rank);
    StoreField(B,
               ConstantInt::get(Ty, 0),
               GEP(B, W, WorkerBuilder::pedigree),
               PedigreeBuilder::rank);
  }

  // w->pedigree.next = &sf->spawn_helper_pedigree;
  StoreField(B,
             GEP(B, SF, StackFrameBuilder::parent_pedigree),
             GEP(B, W, WorkerBuilder::pedigree),
             PedigreeBuilder::next);

  // *tail++ = sf->call_parent;
  B.CreateStore(LoadField(B, SF, StackFrameBuilder::call_parent,
                          /*isVolatile=*/true),
                Tail, /*isVolatile=*/true);
  Tail = B.CreateConstGEP1_32(Tail, 1);

  // w->tail = tail;
  StoreField(B, Tail, W, WorkerBuilder::tail, /*isVolatile=*/true);

  // sf->flags |= CILK_FRAME_DETACHED;
  {
    Value *F = LoadField(B, SF, StackFrameBuilder::flags, /*isVolatile=*/true);
    F = B.CreateOr(F, ConstantInt::get(F->getType(), CILK_FRAME_DETACHED));
    StoreField(B, F, SF, StackFrameBuilder::flags, /*isVolatile=*/true);
  }

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_sync.
/// Calls to this function is always inlined, as it saves
/// the current stack/frame pointer values. This function must be marked
/// as returns_twice to allow it to be inlined, since the call to setjmp
/// is marked returns_twice.
///
/// It is equivalent to the following C code
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
static Function *GetCilkSyncFn(Module &M, bool instrument = false) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilk_sync", M, Fn,
                                     Function::InternalLinkage,
                                     /*doesNotThrow*/false))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.sync.test", Fn);
  BasicBlock *SaveState = BasicBlock::Create(Ctx, "cilk.sync.savestate", Fn);
  BasicBlock *SyncCall = BasicBlock::Create(Ctx, "cilk.sync.runtimecall", Fn);
  BasicBlock *Excepting = BasicBlock::Create(Ctx, "cilk.sync.excepting", Fn);
  // TODO: Detect whether exceptions are needed.
  BasicBlock *Rethrow = BasicBlock::Create(Ctx, "cilk.sync.rethrow", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    if (instrument)
      // cilk_sync_begin
      B.CreateCall(CILK_CSI_FUNC(sync_begin, M), SF);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
                             /*isVolatile=*/true);
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
    StoreField(B,
               LoadField(B, LoadField(B, SF, StackFrameBuilder::worker,
                                      /*isVolatile=*/true),
                         WorkerBuilder::pedigree),
               SF, StackFrameBuilder::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF, M);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Excepting);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(&sf);
    B.CreateCall(CILKRTS_FUNC(sync, M), SF);
    B.CreateBr(Exit);
  }

  // Excepting
  {
    IRBuilder<> B(Excepting);
    if (Rethrow) {
      Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
                               /*isVolatile=*/true);
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
    B.CreateCall(CILKRTS_FUNC(rethrow, M), SF)->setDoesNotReturn();
    B.CreateUnreachable();
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Rank = LoadField(B, SF, StackFrameBuilder::worker,
                            /*isVolatile=*/true);
    Rank = GEP(B, Rank, WorkerBuilder::pedigree);
    Rank = GEP(B, Rank, PedigreeBuilder::rank);
    B.CreateStore(B.CreateAdd(
                      B.CreateLoad(Rank),
                      ConstantInt::get(Rank->getType()->getPointerElementType(),
                                       1)),
                  Rank);
    if (instrument)
      // cilk_sync_end
      B.CreateCall(CILK_CSI_FUNC(sync_end, M), SF);

    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);
  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame.
/// It is equivalent to the following C code
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
static Function *Get__cilkrts_enter_frame_1(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame_1", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  BasicBlock *SlowPath = BasicBlock::Create(Ctx, "slowpath", Fn);
  BasicBlock *FastPath = BasicBlock::Create(Ctx, "fastpath", Fn);
  BasicBlock *Cont = BasicBlock::Create(Ctx, "cont", Fn);

  llvm::PointerType *WorkerPtrTy =
    TypeBuilder<__cilkrts_worker*, false>::get(Ctx);
  StructType *SFTy = StackFrameBuilder::get(Ctx);

  // Block  (Entry)
  CallInst *W = nullptr;
  {
    IRBuilder<> B(Entry);
    if (fastCilk)
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast, M));
    else
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

    Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    B.CreateCondBr(Cond, SlowPath, FastPath);
  }
  // Block  (SlowPath)
  CallInst *Wslow = nullptr;
  {
    IRBuilder<> B(SlowPath);
    Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1, M));
    llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
               ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
               SF, StackFrameBuilder::flags, /*isVolatile=*/true);
    B.CreateBr(Cont);
  }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
               ConstantInt::get(Ty, CILK_FRAME_VERSION),
               SF, StackFrameBuilder::flags, /*isVolatile=*/true);
    B.CreateBr(Cont);
  }
  // Block  (Cont)
  {
    IRBuilder<> B(Cont);
    Value *Wfast = W;
    PHINode *W  = B.CreatePHI(WorkerPtrTy, 2);
    W->addIncoming(Wslow, SlowPath);
    W->addIncoming(Wfast, FastPath);

    StoreField(B,
               LoadField(B, W, WorkerBuilder::current_stack_frame,
                         /*isVolatile=*/true),
               SF, StackFrameBuilder::call_parent,
               /*isVolatile=*/true);

    StoreField(B, W, SF, StackFrameBuilder::worker, /*isVolatile=*/true);
    StoreField(B, SF, W, WorkerBuilder::current_stack_frame,
               /*isVolatile=*/true);

    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame_fast.
/// It is equivalent to the following C code
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
static Function *Get__cilkrts_enter_frame_fast_1(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame_fast_1", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);

  IRBuilder<> B(Entry);
  Value *W;

  if (fastCilk)
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast, M));
  else
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

  StructType *SFTy = StackFrameBuilder::get(Ctx);
  llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);

  StoreField(B,
             ConstantInt::get(Ty, CILK_FRAME_VERSION),
             SF, StackFrameBuilder::flags, /*isVolatile=*/true);
  StoreField(B,
             LoadField(B, W, WorkerBuilder::current_stack_frame,
                       /*isVolatile=*/true),
             SF, StackFrameBuilder::call_parent,
             /*isVolatile=*/true);
  StoreField(B, W, SF, StackFrameBuilder::worker, /*isVolatile=*/true);
  StoreField(B, SF, W, WorkerBuilder::current_stack_frame, /*isVolatile=*/true);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

// /// \brief Get or create a LLVM function for __cilk_parent_prologue.
// /// It is equivalent to the following C code
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
//   B.CreateCall(CILKRTS_FUNC(enter_frame_1, M), SF);

//   B.CreateRetVoid();

//   Fn->addFnAttr(Attribute::InlineHint);

//   return Fn;
// }

/// \brief Get or create a LLVM function for __cilk_parent_epilogue.
/// It is equivalent to the following C code
///
/// void __cilk_parent_epilogue(__cilkrts_stack_frame *sf) {
///   __cilkrts_pop_frame(sf);
///   if (sf->flags != CILK_FRAME_VERSION)
///     __cilkrts_leave_frame(sf);
/// }
static Function *GetCilkParentEpilogue(Module &M, bool instrument = false) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilk_parent_epilogue", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn),
    *B1 = BasicBlock::Create(Ctx, "body", Fn),
    *Exit  = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    if (instrument)
      // cilk_leave_begin
      B.CreateCall(CILK_CSI_FUNC(leave_begin, M), SF);

    // __cilkrts_pop_frame(sf)
    B.CreateCall(CILKRTS_FUNC(pop_frame, M), SF);

    // if (sf->flags != CILK_FRAME_VERSION)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
                             /*isVolatile=*/true);
    Value *Cond = B.CreateICmpNE(Flags,
                                 ConstantInt::get(Flags->getType(),
                                                  CILK_FRAME_VERSION));
    B.CreateCondBr(Cond, B1, Exit);
  }

  // B1
  {
    IRBuilder<> B(B1);

    // __cilkrts_leave_frame(sf);
    B.CreateCall(CILKRTS_FUNC(leave_frame, M), SF);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);
    if (instrument)
      // cilk_leave_end
      B.CreateCall(CILK_CSI_FUNC(leave_end, M));
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const StringRef stack_frame_name = "__cilkrts_sf";
static const StringRef worker8_name = "__cilkrts_wc8";

// static llvm::Value *LookupStackFrame(Function &F) {
//   return F.getValueSymbolTable()->lookup(stack_frame_name);
// }

/// \brief Create the __cilkrts_stack_frame for the spawning function.
static AllocaInst *CreateStackFrame(Function &F) {
  // assert(!LookupStackFrame(F) && "already created the stack frame");

  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *SFTy = StackFrameBuilder::get(Ctx);

  Instruction *I = F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime();

  AllocaInst *SF = new AllocaInst(SFTy, DL.getAllocaAddrSpace(),
                                  /*size*/nullptr, 8,
                                  /*name*/stack_frame_name, /*insert before*/I);
  if (!I)
    F.getEntryBlock().getInstList().push_back(SF);

  return SF;
}

Value* GetOrInitCilkStackFrame(Function& F,
                               ValueToValueMapTy &DetachCtxToStackFrame,
                               bool Helper = true, bool instrument = false) {
  // Value* V = LookupStackFrame(F);
  Value *V = DetachCtxToStackFrame[&F];
  if (V) return V;

  AllocaInst* alloc = CreateStackFrame(F);
  DetachCtxToStackFrame[&F] = alloc;
  BasicBlock::iterator II = F.getEntryBlock().getFirstInsertionPt();
  AllocaInst* curinst;
  do {
    curinst = dyn_cast<llvm::AllocaInst>(II);
    II++;
  } while (curinst != alloc);
  Value *StackSave;
  IRBuilder<> IRB(&(F.getEntryBlock()), II);

  if (instrument) {
    Type *Int8PtrTy = IRB.getInt8PtrTy();
    Value *ThisFn = ConstantExpr::getBitCast(&F, Int8PtrTy);
    Value *ReturnAddress =
      IRB.CreateCall(Intrinsic::getDeclaration(F.getParent(),
                                               Intrinsic::returnaddress),
                     IRB.getInt32(0));
    StackSave =
      IRB.CreateCall(Intrinsic::getDeclaration(F.getParent(),
                                               Intrinsic::stacksave));
    if (Helper) {
      Value *begin_args[3] = { alloc, ThisFn, ReturnAddress };
      IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *F.getParent()),
                     begin_args);
    } else {
      Value *begin_args[4] = { IRB.getInt32(0), alloc, ThisFn, ReturnAddress };
      IRB.CreateCall(CILK_CSI_FUNC(enter_begin, *F.getParent()), begin_args);
    }
  }
  Value *args[1] = { alloc };
  if (Helper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *F.getParent()), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1, *F.getParent()), args);
  /* inst->insertAfter(alloc); */

  if (instrument) {
    Value* end_args[2] = { alloc, StackSave };
    IRB.CreateCall(CILK_CSI_FUNC(enter_end, *F.getParent()), end_args);
  }

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*F.getParent(), instrument),
                         args, "");
  }

  // // The function exits are unified before lowering.
  // ReturnInst *retInst = nullptr;
  // for (BasicBlock &BB : F) {
  //   TerminatorInst* TI = BB.getTerminator();
  //   if (!TI) continue;
  //   if (ReturnInst* RI = llvm::dyn_cast<ReturnInst>(TI)) {
  //     assert(!retInst && "Multiple returns found.");
  //     retInst = RI;
  //   }
  // }

  // assert(retInst && "No returns found.");
  // CallInst::Create(GetCilkParentEpilogue(*F.getParent(), instrument), args, "",
  //                  retInst);
  return alloc;
}

static inline
bool makeFunctionDetachable(Function &extracted,
                            ValueToValueMapTy &DetachCtxToStackFrame,
                            bool instrument = false) {
  Module *M = extracted.getParent();
  // LLVMContext& Context = extracted.getContext();
  // const DataLayout& DL = M->getDataLayout();
  /*
    __cilkrts_stack_frame sf;
    __cilkrts_enter_frame_fast_1(&sf);
    __cilkrts_detach();
    *x = f(y);
  */

  Value *SF = CreateStackFrame(extracted);
  DetachCtxToStackFrame[&extracted] = SF;
  assert(SF);
  Value *args[1] = { SF };

  // Scan function to see if it detaches.
  bool SimpleHelper = true;
  for (BasicBlock &BB : extracted) {
    if (isa<DetachInst>(BB.getTerminator())) {
      SimpleHelper = false;
      break;
    }
  }
  if (!SimpleHelper)
    DEBUG(dbgs() << "Detachable helper function itself detaches.\n");

  BasicBlock::iterator II = extracted.getEntryBlock().getFirstInsertionPt();
  AllocaInst* curinst;
  do {
    curinst = dyn_cast<llvm::AllocaInst>(II);
    II++;
  } while (curinst != SF);
  Value *StackSave;
  IRBuilder<> IRB(&(extracted.getEntryBlock()), II);

  if (instrument) {
    Type *Int8PtrTy = IRB.getInt8PtrTy();
    Value *ThisFn = ConstantExpr::getBitCast(&extracted, Int8PtrTy);
    Value *ReturnAddress =
      IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::returnaddress),
                     IRB.getInt32(0));
    StackSave =
      IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stacksave));
    if (SimpleHelper) {
      Value *begin_args[3] = { SF, ThisFn, ReturnAddress };
      IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *M), begin_args);
    } else {
      Value *begin_args[4] = { IRB.getInt32(0), SF, ThisFn, ReturnAddress };
      IRB.CreateCall(CILK_CSI_FUNC(enter_begin, *M), begin_args);
    }
  }

  if (SimpleHelper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *M), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1, *M), args);

  if (instrument) {
    Value *end_args[2] = { SF, StackSave };
    IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  }

  // Call __cilkrts_detach
  {
    if (instrument)
      IRB.CreateCall(CILK_CSI_FUNC(detach_begin, *M), args);

    IRB.CreateCall(CILKRTS_FUNC(detach, *M), args);

    if (instrument)
      IRB.CreateCall(CILK_CSI_FUNC(detach_end, *M));
  }

  EscapeEnumerator EE(extracted, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*M, instrument), args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      /*
        sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
        sf.except_data = Exn;
      */
      IRBuilder<> B(RI);
      Value *Exn = AtExit->CreateExtractValue(RI->getValue(),
                                              ArrayRef<unsigned>(0));
      Value *Flags = LoadField(*AtExit, SF, StackFrameBuilder::flags,
                               /*isVolatile=*/true);
      Flags = AtExit->CreateOr(Flags,
                               ConstantInt::get(Flags->getType(),
                                                CILK_FRAME_EXCEPTING));
      StoreField(*AtExit, Exn, SF, StackFrameBuilder::except_data);
      /*
        __cilkrts_pop_frame(&sf);
        if (sf->flags)
          __cilkrts_leave_frame(&sf);
      */
      AtExit->CreateCall(GetCilkParentEpilogue(*M, instrument), args, "");
      // CallInst::Create(GetCilkParentEpilogue(*M, instrument), args, "", RI);
    }
  }

  // // Handle returns
  // ReturnInst* Ret = nullptr;
  // for (BasicBlock &BB : extracted) {
  //   TerminatorInst* TI = BB.getTerminator();
  //   if (!TI) continue;
  //   if (ReturnInst* RI = dyn_cast<ReturnInst>(TI)) {
  //     assert(Ret == nullptr && "Multiple return");
  //     Ret = RI;
  //   }
  // }
  // assert(Ret && "No return from extract function");

  // /*
  //    __cilkrts_pop_frame(&sf);
  //    if (sf->flags)
  //      __cilkrts_leave_frame(&sf);
  // */
  // CallInst::Create(GetCilkParentEpilogue(*M, instrument), args, "", Ret);

  // // Handle resumes
  // for (BasicBlock &BB : extracted) {
  //   if (!isa<ResumeInst>(BB.getTerminator()))
  //     continue;
  //   ResumeInst *RI = cast<ResumeInst>(BB.getTerminator());
  //   /*
  //     sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
  //     sf.except_data = Exn;
  //    */
  //   IRBuilder<> B(RI);
  //   Value *Exn = B.CreateExtractValue(RI->getValue(), ArrayRef<unsigned>(0));
  //   Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
  //                            /*isVolatile=*/true);
  //   Flags = B.CreateOr(Flags,
  //                      ConstantInt::get(Flags->getType(),
  //                                       CILK_FRAME_EXCEPTING));
  //   StoreField(B, Exn, SF, StackFrameBuilder::except_data);
  //   /*
  //     __cilkrts_pop_frame(&sf);
  //     if (sf->flags)
  //       __cilkrts_leave_frame(&sf);
  //   */
  //   CallInst::Create(GetCilkParentEpilogue(*M, instrument), args, "", RI);
  // }

  return true;
}

//##############################################################################

/// \brief Get/Create the worker count for the spawning function.
Value* llvm::cilk::GetOrCreateWorker8(Function &F) {
  // Value* W8 = F.getValueSymbolTable()->lookup(worker8_name);
  // if (W8) return W8;
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  Value *P0 = B.CreateCall(CILKRTS_FUNC(get_nworkers, *F.getParent()));
  Value *P8 = B.CreateMul(P0, ConstantInt::get(P0->getType(), 8), worker8_name);
  return P8;
}

void llvm::cilk::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame,
                            bool instrument) {
  Function &Fn = *(SI.getParent()->getParent());
  Module &M = *(Fn.getParent());

  Value *SF = GetOrInitCilkStackFrame(Fn, DetachCtxToStackFrame,
                                      /*isFast*/false, instrument);
  Value *args[] = { SF };
  assert( args[0] && "sync used in function without frame!" );
  CallInst *CI = CallInst::Create(GetCilkSyncFn(M, instrument), args, "",
                                  /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
}

bool llvm::cilk::verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                                   bool error) {
  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);

  SmallVector<BasicBlock *, 32> Todo;
  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BasicBlock *, 4> WorkListEH;
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst* Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (ReattachInst* Inst = dyn_cast<ReattachInst>(Term)) {
      //only analyze reattaches going to the same continuation
      if (Inst->getSuccessor(0) != Continue) continue;
      continue;
    } else if (DetachInst* Inst = dyn_cast<DetachInst>(Term)) {
      assert(Inst != &Detach && "Found recursive Detach!");
      Todo.push_back(Inst->getSuccessor(0));
      Todo.push_back(Inst->getSuccessor(1));
      continue;
    } else if (SyncInst* Inst = dyn_cast<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Inst->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ))
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          WorkListEH.push_back(Succ);
        else
          Todo.push_back(Succ);
      }
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      DEBUG(Term->dump());
      DEBUG(Term->getParent()->getParent()->dump());
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
    }
  }
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      for (BasicBlock *Succ : successors(BB))
        WorkListEH.push_back(Succ);
    }
  }
  return true;
}

bool llvm::cilk::populateDetachedCFG(
    const DetachInst &Detach, DominatorTree &DT,
    SmallPtrSetImpl<BasicBlock *> &functionPieces,
    SmallVectorImpl<BasicBlock *> &reattachB,
    SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
    bool replace, bool error) {
  SmallVector<BasicBlock *, 32> Todo;
  SmallVector<BasicBlock *, 4> WorkListEH;

  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst *Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (isa<ReattachInst>(Term)) {
      // only analyze reattaches going to the same continuation
      if (Term->getSuccessor(0) != Continue) continue;
      if (replace) {
        BranchInst* toReplace = BranchInst::Create(Continue);
        ReplaceInstWithInst(Term, toReplace);
        reattachB.push_back(BB);
      }
      continue;
    } else if (isa<DetachInst>(Term)) {
      assert(Term != &Detach && "Found recursive detach!");
      Todo.push_back(Term->getSuccessor(0));
      Todo.push_back(Term->getSuccessor(1));
      continue;
    } else if (isa<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Term->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ)) {
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          ExitBlocks.insert(Succ);
          WorkListEH.push_back(Succ);
        } else {
          Todo.push_back(Succ);
        }
      }
      // We don't bother cloning unreachable exits from the detached CFG at this
      // point.  We're cloning the entire detached CFG anyway when we outline
      // the function.
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      DEBUG(Term->dump());
      DEBUG(Term->getParent()->getParent()->dump());
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
    }
  }

  // Find the exit-handling blocks.
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      // if (isa<ResumeInst>(BB-getTerminator()))
      //   ResumeBlocks.push_back(BB);

      for (BasicBlock *Succ : successors(BB)) {
        ExitBlocks.insert(Succ);
        WorkListEH.push_back(Succ);
      }
    }

    // Visited now contains exception-handling blocks that we want to clone as
    // part of outlining.
    for (BasicBlock *EHBlock : Visited)
      functionPieces.insert(EHBlock);
  }

  return true;
}

//Returns true if success
Function *llvm::cilk::extractDetachBodyToFunction(DetachInst &detach,
                                                  DominatorTree &DT,
                                                  AssumptionCache &AC,
                                                  CallInst **call) {
  BasicBlock *Detacher = detach.getParent();
  Function &F = *(Detacher->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BasicBlock *, 32> reattachB;
  SmallPtrSet<BasicBlock *, 4> ExitBlocks;

  // if (!Spawned->getUniquePredecessor())
  //   dbgs() << *Spawned;
  assert(Spawned->getUniquePredecessor() &&
         "Entry block of detached CFG has multiple predecessors.");
  assert(Spawned->getUniquePredecessor() == Detacher &&
         "Broken CFG.");

  // if (getNumPred(Spawned) > 1) {
  //   dbgs() << "Found multiple predecessors to a detached-CFG entry block "
  //          << Spawned->getName() << ".\n";
  //   BasicBlock* ts = BasicBlock::Create(Spawned->getContext(), Spawned->getName()+".fx", &F, Detacher);
  //   IRBuilder<> b(ts);
  //   b.CreateBr(Spawned);
  //   detach.setSuccessor(0,ts);
  //   llvm::BasicBlock::iterator i = Spawned->begin();
  //   while (auto phi = llvm::dyn_cast<llvm::PHINode>(i)) {
  //     int idx = phi->getBasicBlockIndex(detach.getParent());
  //     phi->setIncomingBlock(idx, ts);
  //     ++i;
  //   }
  //   Spawned = ts;
  // }

  if (!populateDetachedCFG(detach, DT, functionPieces, reattachB,
                           ExitBlocks, true))
    return nullptr;

  // functionPieces.erase(Spawned);
  // std::vector<BasicBlock *> blocks(functionPieces.begin(), functionPieces.end());
  // blocks.insert(blocks.begin(), Spawned);
  // functionPieces.insert(Spawned);

  // Check the spawned block's predecessors.
  for (BasicBlock *BB : functionPieces) {
    int detached_count = 0;
    if (ExitBlocks.count(BB))
      continue;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      if (detached_count == 0 && BB == Spawned && Pred == detach.getParent()) {
        detached_count = 1;
        continue;
      }
      assert(functionPieces.count(Pred) &&
             "Block inside of detached context branched into from outside branch context");
    }
  }

  // Get the inputs and outputs for the detached CFG.
  SetVector<Value *> Inputs, Outputs;
  findInputsOutputs(functionPieces, Inputs, Outputs, &ExitBlocks);
  // extractor.findInputsOutputs(Inputs, Outputs);
  assert(Outputs.empty() &&
         "All results from detached CFG should be passed by memory already.");

  // Clone the detached CFG into a helper function.
  ValueToValueMapTy VMap;
  Function *extracted;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
    std::vector<BasicBlock *> blocks(functionPieces.begin(), functionPieces.end());

    extracted = CreateHelper(Inputs, Outputs, blocks,
                             Spawned, Detacher, Continue,
                             VMap, F.getParent(),
                             F.getSubprogram() != nullptr, Returns, ".cilk",
                             &ExitBlocks, nullptr, nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning detached CFG.");

    // Use a fast calling convention for the helper.
    extracted->setCallingConv(CallingConv::Fast);
    // extracted->setCallingConv(F.getCallingConv());

    extracted->addFnAttr(Attribute::NoInline);
  }

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(&F, Inputs, VMap, &detach, &AC, &DT);

  // Add call to new helper function in original function.
  CallInst *TopCall;
  {
    // Create call instruction.
    IRBuilder<> Builder(&detach);
    TopCall = Builder.CreateCall(extracted, Inputs.getArrayRef());
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    // TopCall->setCallingConv(extracted->getCallingConv());
    TopCall->setDebugLoc(detach.getDebugLoc());
  }
  if (call)
    *call = TopCall;

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (pred_iterator PI = pred_begin(Continue), PE = pred_end(Continue);
         PI != PE; ++PI) {
      BasicBlock *Pred = *PI;
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (functionPieces.count(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[Spawned]);
    MoveStaticAllocasInBlock(&extracted->getEntryBlock(), ClonedDetachedBlock,
                             ReattachPoints);

    // We should not need to add new llvm.stacksave/llvm.stackrestore
    // intrinsics, because calling and returning from the helper will
    // automatically manage the stack.
  }

  return extracted;
}

Function *llvm::cilk::createDetach(DetachInst &detach,
                                   ValueToValueMapTy &DetachCtxToStackFrame,
                                   DominatorTree &DT, AssumptionCache &AC,
                                   bool instrument) {
  BasicBlock *detB = detach.getParent();
  Function &F = *(detB->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  Module *M = F.getParent();
  //replace with branch to succesor
  //entry / cilk.spawn.savestate
  Value *SF = GetOrInitCilkStackFrame(F, DetachCtxToStackFrame,
                                      /*isFast=*/false, instrument);
  // assert(SF && "null stack frame unexpected");

  // dbgs() << *detB << *Spawned << *Continue;

  // if (!Spawned->getUniquePredecessor())
  //   SplitEdge(detB, Spawned, &DT, nullptr);

  // dbgs() << *detB << *(detach.getDetached());

  CallInst *cal = nullptr;
  Function *extracted = extractDetachBodyToFunction(detach, DT, AC, &cal);
  assert(extracted && "could not extract detach body to function");

  // Unlink the detached CFG in the original function.  The heavy lifting of
  // removing the outlined detached-CFG is left to subsequent DCE.
  BranchInst *ContinueBr;
  {
    // Replace the detach with a branch to the continuation.
    ContinueBr = BranchInst::Create(Continue);
    ReplaceInstWithInst(&detach, ContinueBr);

    // Rewrite phis in the detached block.
    BasicBlock::iterator BI = Spawned->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      // int j = P->getBasicBlockIndex(detB);
      // assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
      P->removeIncomingValue(detB);
      ++BI;
    }
  }

  Value *SetJmpRes;
  {
    IRBuilder<> B(cal);

    if (instrument)
      // cilk_spawn_prepare
      B.CreateCall(CILK_CSI_FUNC(spawn_prepare, *M), SF);

    // Need to save state before spawning
    SetJmpRes = EmitCilkSetJmp(B, SF, *M);

    if (instrument)
      // cilk_spawn_or_continue
      B.CreateCall(CILK_CSI_FUNC(spawn_or_continue, *M), SetJmpRes);
  }

  // Conditionally call the new helper function based on the result of the
  // setjmp.
  {
    BasicBlock *CallBlock = SplitBlock(detB, cal, &DT);
    BasicBlock *CallCont = SplitBlock(CallBlock,
                                      CallBlock->getTerminator(), &DT);
    IRBuilder<> B(detB->getTerminator());
    SetJmpRes = B.CreateICmpEQ(SetJmpRes,
                               ConstantInt::get(SetJmpRes->getType(), 0));
    B.CreateCondBr(SetJmpRes, CallBlock, CallCont);
    detB->getTerminator()->eraseFromParent();
  }

  makeFunctionDetachable(*extracted, DetachCtxToStackFrame, instrument);

  return extracted;
}
