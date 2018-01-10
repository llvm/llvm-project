//===- CilkRABI.cpp - Lower Tapir into CilkR runtime system calls ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CilkRABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Cilk
// runtime system.  This interface does the low-level dirty work of passes
// such as LowerToCilk.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkRABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilkrabi"

typedef void *__CILK_JUMP_BUFFER[5];

// typedef CilkRABI::__cilkrts_pedigree __cilkrts_pedigree;
typedef CilkRABI::__cilkrts_stack_frame __cilkrts_stack_frame;
typedef CilkRABI::__cilkrts_worker __cilkrts_worker;

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

typedef void (__cilkrts_init)();

typedef void (__cilkrts_enter_frame)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_enter_frame_fast)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_leave_frame)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_rethrow)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_sync)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_detach)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_pop_frame)(__cilkrts_stack_frame *sf);
typedef int (__cilkrts_get_nworkers)();
typedef __cilkrts_worker *(__cilkrts_get_tls_worker)();
// typedef __cilkrts_worker *(__cilkrts_bind_thread_1)();

typedef void (cilk_func)(__cilkrts_stack_frame *);

#define CILKRTS_FUNC(name, CGF) Get__cilkrts_##name(CGF)

#define DEFAULT_GET_CILKRTS_FUNC(name)                                  \
  static Function *Get__cilkrts_##name(Module& M) {         \
    return cast<Function>(M.getOrInsertFunction(            \
                                          "__cilkrts_"#name,            \
                                          TypeBuilder<__cilkrts_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

//DEFAULT_GET_CILKRTS_FUNC(get_nworkers)
// #pragma GCC diagnostic ignored "-Wunused-function"
static Function *Get__cilkrts_get_nworkers(Module& M) {
  LLVMContext &C = M.getContext();
  AttributeList AL;
  AL = AL.addAttribute(C, AttributeList::FunctionIndex,
                       Attribute::ReadNone);
  // AL = AL.addAttribute(C, AttributeSet::FunctionIndex,
  //                      Attribute::InaccessibleMemOnly);
  AL = AL.addAttribute(C, AttributeList::FunctionIndex,
                       Attribute::NoUnwind);
  Function *F = cast<Function>(
      M.getOrInsertFunction(
          "__cilkrts_get_nworkers",
          TypeBuilder<__cilkrts_get_nworkers, false>::get(C),
          AL));
  return F;
}

// TODO: set up these CILKRTS and CILK_CSI functions in a cleaner
// way so we don't need these pragmas.
// DEFAULT_GET_CILKRTS_FUNC(init)
DEFAULT_GET_CILKRTS_FUNC(sync)
// DEFAULT_GET_CILKRTS_FUNC(rethrow)
DEFAULT_GET_CILKRTS_FUNC(leave_frame)
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker)
// DEFAULT_GET_CILKRTS_FUNC(get_tls_worker_fast)
// DEFAULT_GET_CILKRTS_FUNC(bind_thread_1)

typedef std::map<LLVMContext*, StructType*> TypeBuilderCache;

namespace llvm {
/// Specializations of TypeBuilder for:
///   __cilkrts_pedigree,
///   __cilkrts_worker,
///   __cilkrts_stack_frame
// template <bool X>
// class TypeBuilder<__cilkrts_pedigree, X> {
// public:
//   static StructType *get(LLVMContext &C) {
//     static TypeBuilderCache cache;
//     TypeBuilderCache::iterator I = cache.find(&C);
//     if (I != cache.end())
//       return I->second;
//     StructType *ExistingTy = StructType::lookupOrCreate(C, "struct.__cilkrts_pedigree");
//     cache[&C] = ExistingTy;
//     StructType *NewTy = StructType::create(C);
//     NewTy->setBody(
//         TypeBuilder<uint64_t,            X>::get(C), // rank
//         TypeBuilder<__cilkrts_pedigree*, X>::get(C)  // next
//                 );
//     if (ExistingTy->isOpaque())
//       ExistingTy->setBody(NewTy->elements());
//     else
//       assert(ExistingTy->isLayoutIdentical(NewTy) &&
//              "Conflicting definition of tye struct.__cilkrts_pedigree");
//     return ExistingTy;
//   }
//   enum {
//     rank,
//     next
//   };
// };

template <bool X>
class TypeBuilder<__cilkrts_worker, X> {
public:
  static StructType *get(LLVMContext &C) {
    static TypeBuilderCache cache;
    TypeBuilderCache::iterator I = cache.find(&C);
    if (I != cache.end())
      return I->second;
    // Try looking up this type by name.
    StructType *ExistingTy = StructType::lookupOrCreate(C, "struct.__cilkrts_worker");
    cache[&C] = ExistingTy;
    if (ExistingTy->isOpaque())
      // Define the layout of the __cilkrts_worker struct needed for this pass.
      ExistingTy->setBody(
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // tail
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // head
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // exc
          // TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // protected_tail
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // ltq_limit
          TypeBuilder<int32_t,                 X>::get(C), // self
          TypeBuilder<void*,                   X>::get(C), // g
          TypeBuilder<void*,                   X>::get(C), // l
          // TypeBuilder<void*,                   X>::get(C), // reducer_map
          TypeBuilder<__cilkrts_stack_frame*,  X>::get(C)  // current_stack_frame
          // TypeBuilder<void*,                   X>::get(C), // saved_protected_tail
          // TypeBuilder<void*,                   X>::get(C), // sysdep
          // TypeBuilder<__cilkrts_pedigree,      X>::get(C)  // pedigree
                          );
    else {
      // Verify that the necessary parts of the __cilkrts_worker struct appear
      // in the right places.
      assert((ExistingTy->getElementType(tail) ==
              TypeBuilder<__cilkrts_stack_frame**, X>::get(C)) &&
             "Invalid type for __cilkrts_worker.tail");
      assert((ExistingTy->getElementType(current_stack_frame) ==
              TypeBuilder<__cilkrts_stack_frame*, X>::get(C)) &&
             "Invalid type for __cilkrts_worker.current_stack_frame");
      // assert((ExistingTy->getElementType(pedigree) ==
      //         TypeBuilder<__cilkrts_pedigree, X>::get(C)) &&
      //        "Invalid type for __cilkrts_worker.pedigree");
    }
    return ExistingTy;
  }
  enum {
    tail,
    head,
    exc,
    // protected_tail,
    ltq_limit,
    self,
    g,
    l,
    // reducer_map,
    current_stack_frame,
    // saved_protected_tail,
    // sysdep,
    // pedigree
  };
};

template <bool X>
class TypeBuilder<__cilkrts_stack_frame, X> {
public:
  static StructType *get(LLVMContext &C) {
    static TypeBuilderCache cache;
    TypeBuilderCache::iterator I = cache.find(&C);
    if (I != cache.end())
      return I->second;
    StructType *ExistingTy = StructType::lookupOrCreate(C, "struct.__cilkrts_stack_frame");
    cache[&C] = ExistingTy;
    StructType *NewTy = StructType::create(C);
    NewTy->setBody(
        TypeBuilder<uint32_t,               X>::get(C), // flags
        // TypeBuilder<int32_t,                X>::get(C), // size
        TypeBuilder<__cilkrts_stack_frame*, X>::get(C), // call_parent
        TypeBuilder<__cilkrts_worker*,      X>::get(C), // worker
        // TypeBuilder<void*,                  X>::get(C), // except_data
        TypeBuilder<__CILK_JUMP_BUFFER,     X>::get(C), // ctx
        TypeBuilder<uint32_t,               X>::get(C), // mxcsr
        TypeBuilder<uint16_t,               X>::get(C), // fpcsr
        TypeBuilder<uint16_t,               X>::get(C), // reserved
        // ExistingTy->isOpaque() ?
        // StructType::get(
        //     TypeBuilder<__cilkrts_pedigree, X>::get(C)  // parent_pedigree
        //                 ) :
        // ExistingTy->getStructElementType(parent_pedigree)
        TypeBuilder<uint32_t,               X>::get(C) // magic
                   );
    if (ExistingTy->isOpaque())
      ExistingTy->setBody(NewTy->elements());
    else
      assert(ExistingTy->isLayoutIdentical(NewTy) &&
             "Conflicting definition of tye struct.__cilkrts_stack_frame");
    return ExistingTy;
  }
  enum {
    flags,
    // size,
    call_parent,
    worker,
    // except_data,
    ctx,
    mxcsr,
    fpcsr,
    reserved,
    // parent_pedigree,
    magic,
  };
};
} // end namespace llvm

/// Helper typedefs for cilk struct TypeBuilders.
typedef TypeBuilder<__cilkrts_stack_frame, false> StackFrameBuilder;
typedef TypeBuilder<__cilkrts_worker, false> WorkerBuilder;
// typedef TypeBuilder<__cilkrts_pedigree, false> PedigreeBuilder;

/// Helper methods for storing to and loading from struct fields.
static Value *GEP(IRBuilder<> &B, Value *Base, int field) {
  // return B.CreateStructGEP(cast<PointerType>(Base->getType()),
  //                          Base, field);
  return B.CreateConstInBoundsGEP2_32(nullptr, Base, 0, field);
}

static void StoreField(IRBuilder<> &B, Value *Val, Value *Dst, int field,
                       bool isVolatile = false,
                       AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  Value *FieldDst = GEP(B, Dst, field);
  StoreInst *S = B.CreateStore(Val, FieldDst, isVolatile);
  S->setOrdering(Ordering);
}

static Value *LoadField(IRBuilder<> &B, Value *Src, int field,
                        bool isVolatile = false,
                        AtomicOrdering Ordering = AtomicOrdering::NotAtomic) {
  LoadInst *L = B.CreateLoad(GEP(B, Src, field), isVolatile);
  L->setOrdering(Ordering);
  return L;
}

/// \brief Emit inline assembly code to save the floating point
/// state, for x86 Only.
static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
  typedef void (AsmPrototype)(uint32_t*, uint16_t*);
  FunctionType *FTy =
    TypeBuilder<AsmPrototype, false>::get(B.getContext());

  Value *Asm = InlineAsm::get(FTy,
                              "stmxcsr $0\n\t" "fnstcw $1",
                              "*m,*m,~{dirflag},~{fpsr},~{flags}",
                              /*sideeffects*/ true);

  Value *args[2] = {
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
             LoadField(B, SF, StackFrameBuilder::call_parent),
             LoadField(B, SF, StackFrameBuilder::worker),
             WorkerBuilder::current_stack_frame);

  // sf->call_parent = 0;
  StoreField(B,
             Constant::getNullValue(
                 TypeBuilder<__cilkrts_stack_frame*, false>::get(Ctx)),
             SF, StackFrameBuilder::call_parent);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_detach.
/// It is equivalent to the following C code
///
/// void __cilkrts_detach(struct __cilkrts_stack_frame *sf) {
///   struct __cilkrts_worker *w = sf->worker;
///   struct __cilkrts_stack_frame *parent = sf->call_parent;
///   struct __cilkrts_stack_frame *volatile *tail = w->tail;
///
///   Release_fence();
//
///   *tail++ = parent;
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
  Value *W = LoadField(B, SF, StackFrameBuilder::worker);

    // __cilkrts_stack_frame *parent = sf->call_parent;
  Value *Parent = LoadField(B, SF, StackFrameBuilder::call_parent);

  // __cilkrts_stack_frame *volatile *tail = w->tail;
  Value *Tail = LoadField(B, W, WorkerBuilder::tail);

  // StoreStore_fence();
  B.CreateFence(AtomicOrdering::Release);

  // *tail++ = parent;
  B.CreateStore(Parent, Tail);
  Tail = B.CreateConstGEP1_32(Tail, 1);

  // w->tail = tail;
  StoreField(B, Tail, W, WorkerBuilder::tail);

  // sf->flags |= CILK_FRAME_DETACHED;
  {
    Value *F = LoadField(B, SF, StackFrameBuilder::flags);
    F = B.CreateOr(F, ConstantInt::get(F->getType(), CILK_FRAME_DETACHED));
    StoreField(B, F, SF, StackFrameBuilder::flags);
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
///     SAVE_FLOAT_STATE(*sf);
///     if (!CILK_SETJMP(sf->ctx))
///       __cilkrts_sync(sf);
///     else if (sf->flags & CILK_FRAME_EXCEPTING)
///       __cilkrts_rethrow(sf);
///   }
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
static Function *GetCilkSyncFn(Module &M) {
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
  // BasicBlock *Rethrow = BasicBlock::Create(Ctx, "cilk.sync.rethrow", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
                             false);
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
    // if (Rethrow) {
    //   Value *Flags = LoadField(B, SF, StackFrameBuilder::flags,
    //                            /*isVolatile=*/true);
    //   Flags = B.CreateAnd(Flags,
    //                       ConstantInt::get(Flags->getType(),
    //                                        CILK_FRAME_EXCEPTING));
    //   Value *Zero = ConstantInt::get(Flags->getType(), 0);
    //   Value *CanExcept = B.CreateICmpEQ(Flags, Zero);
    //   B.CreateCondBr(CanExcept, Exit, Rethrow);
    // } else {
      B.CreateBr(Exit);
    // }
  }

  // // Rethrow
  // if (Rethrow) {
  //   IRBuilder<> B(Rethrow);
  //   B.CreateCall(CILKRTS_FUNC(rethrow, M), SF)->setDoesNotReturn();
  //   B.CreateUnreachable();
  // }

  // Exit
  {
    IRBuilder<> B(Exit);

    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);
  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame.
/// It is equivalent to the following C code
///
/// void __cilkrts_enter_frame(struct __cilkrts_stack_frame *sf)
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
static Function *Get__cilkrts_enter_frame(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  // BasicBlock *SlowPath = BasicBlock::Create(Ctx, "slowpath", Fn);
  BasicBlock *FastPath = BasicBlock::Create(Ctx, "fastpath", Fn);
  BasicBlock *Cont = BasicBlock::Create(Ctx, "cont", Fn);

  PointerType *WorkerPtrTy =
    TypeBuilder<__cilkrts_worker*, false>::get(Ctx);
  StructType *SFTy = StackFrameBuilder::get(Ctx);

  // Block  (Entry)
  CallInst *W = nullptr;
  {
    IRBuilder<> B(Entry);
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

    // Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    // B.CreateCondBr(Cond, SlowPath, FastPath);
    B.CreateBr(FastPath);
  }
  // // Block  (SlowPath)
  // CallInst *Wslow = nullptr;
  // {
  //   IRBuilder<> B(SlowPath);
  //   Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1, M));
  //   Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
  //   StoreField(B,
  //              ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
  //              SF, StackFrameBuilder::flags, /*isVolatile=*/true);
  //   B.CreateBr(Cont);
  // }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
               ConstantInt::get(Ty, CILK_FRAME_VERSION),
               SF, StackFrameBuilder::flags);
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
    StoreField(B,
               LoadField(B, Wkr, WorkerBuilder::current_stack_frame),
               SF, StackFrameBuilder::call_parent);

    StoreField(B, Wkr, SF, StackFrameBuilder::worker);
    StoreField(B, SF, Wkr, WorkerBuilder::current_stack_frame);

    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame_fast.
/// It is equivalent to the following C code
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
static Function *Get__cilkrts_enter_frame_fast(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame_fast", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);

  IRBuilder<> B(Entry);
  Value *W;

  // if (fastCilk)
  //   W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast, M));
  // else
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

  StructType *SFTy = StackFrameBuilder::get(Ctx);
  Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);

  StoreField(B,
             ConstantInt::get(Ty, CILK_FRAME_VERSION),
             SF, StackFrameBuilder::flags);
  StoreField(B,
             LoadField(B, W, WorkerBuilder::current_stack_frame),
             SF, StackFrameBuilder::call_parent);
  StoreField(B, W, SF, StackFrameBuilder::worker);
  StoreField(B, SF, W, WorkerBuilder::current_stack_frame);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_parent_epilogue.
/// It is equivalent to the following C code
///
/// void __cilk_parent_epilogue(__cilkrts_stack_frame *sf) {
///   __cilkrts_pop_frame(sf);
///   if (sf->flags != CILK_FRAME_VERSION)
///     __cilkrts_leave_frame(sf);
/// }
static Function *GetCilkParentEpilogue(Module &M) {
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

    // __cilkrts_pop_frame(sf)
    B.CreateCall(CILKRTS_FUNC(pop_frame, M), SF);

    // if (sf->flags != CILK_FRAME_VERSION)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
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
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const StringRef stack_frame_name = "__cilkrts_sf";
static const StringRef worker8_name = "__cilkrts_wc8";

/// \brief Create the __cilkrts_stack_frame for the spawning function.
static AllocaInst *CreateStackFrame(Function &F) {
  // assert(!LookupStackFrame(F) && "already created the stack frame");

  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *SFTy = StackFrameBuilder::get(Ctx);

  Instruction *I = F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime();

  AllocaInst *SF = new AllocaInst(SFTy, DL.getAllocaAddrSpace(),
                                  /*ArraySize=*/nullptr, /*Align=*/8,
                                  /*Name=*/stack_frame_name,
                                  /*InsertBefore=*/I);
  if (!I)
    F.getEntryBlock().getInstList().push_back(SF);

  return SF;
}

Value* GetOrInitCilkStackFrame(Function &F,
                               ValueToValueMapTy &DetachCtxToStackFrame,
                               bool Helper = true) {
  Value *V = DetachCtxToStackFrame[&F];
  if (V) return V;

  AllocaInst* alloc = CreateStackFrame(F);
  DetachCtxToStackFrame[&F] = alloc;
  BasicBlock::iterator II = F.getEntryBlock().getFirstInsertionPt();
  AllocaInst* curinst;
  do {
    curinst = dyn_cast<AllocaInst>(II);
    II++;
  } while (curinst != alloc);
  IRBuilder<> IRB(&(F.getEntryBlock()), II);

  Value *args[1] = { alloc };
  if (Helper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast, *F.getParent()), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame, *F.getParent()), args);
  /* inst->insertAfter(alloc); */

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*F.getParent()),
                         args, "");
  }
  return alloc;
}

static inline
bool makeFunctionDetachable(Function &extracted,
                            ValueToValueMapTy &DetachCtxToStackFrame) {
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
    curinst = dyn_cast<AllocaInst>(II);
    II++;
  } while (curinst != SF);
  // Value *StackSave;
  IRBuilder<> IRB(&(extracted.getEntryBlock()), II);

  if (SimpleHelper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast, *M), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame, *M), args);

  // Call __cilkrts_detach
  {
    IRB.CreateCall(CILKRTS_FUNC(detach, *M), args);
  }

  EscapeEnumerator EE(extracted, "cilkrabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*M), args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      // TODO: Handle exceptions.
      // /*
      //   sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
      //   sf.except_data = Exn;
      // */
      // IRBuilder<> B(RI);
      // Value *Exn = AtExit->CreateExtractValue(RI->getValue(),
      //                                         ArrayRef<unsigned>(0));
      // Value *Flags = LoadField(*AtExit, SF, StackFrameBuilder::flags);
      // Flags = AtExit->CreateOr(Flags,
      //                          ConstantInt::get(Flags->getType(),
      //                                           CILK_FRAME_EXCEPTING));
      // StoreField(*AtExit, Exn, SF, StackFrameBuilder::except_data);
      /*
        __cilkrts_pop_frame(&sf);
        if (sf->flags)
          __cilkrts_leave_frame(&sf);
      */
      AtExit->CreateCall(GetCilkParentEpilogue(*M), args, "");
    }
  }

  return true;
}

//##############################################################################

CilkRABI::CilkRABI() {}

/// \brief Get/Create the worker count for the spawning function.
Value *CilkRABI::GetOrCreateWorker8(Function &F) {
  // Value* W8 = F.getValueSymbolTable()->lookup(worker8_name);
  // if (W8) return W8;
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  Value *P0 = B.CreateCall(CILKRTS_FUNC(get_nworkers, *F.getParent()));
  Value *P8 = B.CreateMul(P0, ConstantInt::get(P0->getType(), 8), worker8_name);
  return P8;
}

void CilkRABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {
  Function &Fn = *(SI.getParent()->getParent());
  Module &M = *(Fn.getParent());

  Value *SF = GetOrInitCilkStackFrame(Fn, DetachCtxToStackFrame,
                                      /*isFast*/false);
  Value *args[] = { SF };
  assert( args[0] && "sync used in function without frame!" );
  CallInst *CI = CallInst::Create(GetCilkSyncFn(M), args, "",
                                  /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
}

Function *CilkRABI::createDetach(DetachInst &detach,
                                 ValueToValueMapTy &DetachCtxToStackFrame,
                                 DominatorTree &DT, AssumptionCache &AC) {
  BasicBlock *detB = detach.getParent();
  Function &F = *(detB->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  Module *M = F.getParent();
  //replace with branch to succesor
  //entry / cilk.spawn.savestate
  Value *SF = GetOrInitCilkStackFrame(F, DetachCtxToStackFrame,
                                      /*isFast=*/false);
  assert(SF && "null stack frame unexpected");

  CallInst *cal = nullptr;
  Function *extracted = extractDetachBodyToFunction(detach, DT, AC, &cal);
  assert(extracted && "could not extract detach body to function");

  // Unlink the detached CFG in the original function.  The heavy lifting of
  // removing the outlined detached-CFG is left to subsequent DCE.

  // Replace the detach with a branch to the continuation.
  BranchInst *ContinueBr = BranchInst::Create(Continue);
  ReplaceInstWithInst(&detach, ContinueBr);

  // Rewrite phis in the detached block.
  {
    BasicBlock::iterator BI = Spawned->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      P->removeIncomingValue(detB);
      ++BI;
    }
  }

  Value *SetJmpRes;
  {
    IRBuilder<> b(cal);
    SetJmpRes = EmitCilkSetJmp(b, SF, *M);
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

  makeFunctionDetachable(*extracted, DetachCtxToStackFrame);

  return extracted;
}

// Helper function to inline calls to compiler-generated Cilk Plus runtime
// functions when possible.  This inlining is necessary to properly implement
// some Cilk runtime "calls," such as __cilkrts_detach().
static inline void inlineCilkFunctions(Function &F) {
  bool inlining = true;
  while (inlining) {
    inlining = false;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
      if (CallInst *cal = dyn_cast<CallInst>(&*I))
        if (Function *fn = cal->getCalledFunction())
          if (fn->getName().startswith("__cilk")) {
            InlineFunctionInfo IFI;
            if (InlineFunction(cal, IFI)) {
              if (fn->getNumUses()==0)
                fn->eraseFromParent();
              inlining = true;
              break;
            }
          }
  }

  if (verifyFunction(F, &errs())) {
    DEBUG(F.dump());
    assert(0);
  }
}

void CilkRABI::preProcessFunction(Function &F) {
  if (F.getName() == "main")
    F.setName("cilk_main");
}

void CilkRABI::postProcessFunction(Function &F) {
  inlineCilkFunctions(F);
}

void CilkRABI::postProcessHelper(Function &F) {
  inlineCilkFunctions(F);
}
