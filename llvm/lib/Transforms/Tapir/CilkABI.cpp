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
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilkabi"

STATISTIC(LoopsConvertedToCilkABI,
          "Number of Tapir loops converted to use the Cilk ABI for loops");

extern cl::opt<bool> fastCilk;

typedef void *__CILK_JUMP_BUFFER[5];

typedef tapir::CilkABI::__cilkrts_pedigree __cilkrts_pedigree;
typedef tapir::CilkABI::__cilkrts_stack_frame __cilkrts_stack_frame;
typedef tapir::CilkABI::__cilkrts_worker __cilkrts_worker;

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


typedef uint32_t cilk32_t;
typedef uint64_t cilk64_t;
typedef void (*__cilk_abi_f32_t)(void *data, cilk32_t low, cilk32_t high);
typedef void (*__cilk_abi_f64_t)(void *data, cilk64_t low, cilk64_t high);

typedef void (__cilkrts_init)();

typedef void (__cilkrts_enter_frame_1)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_enter_frame_fast_1)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_leave_frame)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_rethrow)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_sync)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_detach)(__cilkrts_stack_frame *sf);
typedef void (__cilkrts_pop_frame)(__cilkrts_stack_frame *sf);
typedef int (__cilkrts_get_nworkers)();
typedef __cilkrts_worker *(__cilkrts_get_tls_worker)();
typedef __cilkrts_worker *(__cilkrts_get_tls_worker_fast)();
typedef __cilkrts_worker *(__cilkrts_bind_thread_1)();

typedef void (cilk_func)(__cilkrts_stack_frame *);

typedef void (cilk_enter_begin)(uint32_t, __cilkrts_stack_frame *, void *, void *);
typedef void (cilk_enter_helper_begin)(__cilkrts_stack_frame *, void *, void *);
typedef void (cilk_enter_end)(__cilkrts_stack_frame *, void *);
typedef void (cilk_detach_begin)(__cilkrts_stack_frame *);
typedef void (cilk_detach_end)();
typedef void (cilk_spawn_prepare)(__cilkrts_stack_frame *);
typedef void (cilk_spawn_or_continue)(int);
typedef void (cilk_sync_begin)(__cilkrts_stack_frame *);
typedef void (cilk_sync_end)(__cilkrts_stack_frame *);
typedef void (cilk_leave_begin)(__cilkrts_stack_frame *);
typedef void (cilk_leave_end)();
typedef void (__cilkrts_cilk_for_32)(__cilk_abi_f32_t body, void *data,
                                     cilk32_t count, int grain);
typedef void (__cilkrts_cilk_for_64)(__cilk_abi_f64_t body, void *data,
                                     cilk64_t count, int grain);

#define CILKRTS_FUNC(name, CGF) Get__cilkrts_##name(CGF)

#define DEFAULT_GET_CILKRTS_FUNC(name)                                  \
  static Function *Get__cilkrts_##name(Module& M) {         \
    return cast<Function>(M.getOrInsertFunction(            \
                                          "__cilkrts_"#name,            \
                                          TypeBuilder<__cilkrts_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

//DEFAULT_GET_CILKRTS_FUNC(get_nworkers)
//#pragma GCC diagnostic ignored "-Wunused-function"
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
DEFAULT_GET_CILKRTS_FUNC(init)
DEFAULT_GET_CILKRTS_FUNC(sync)
DEFAULT_GET_CILKRTS_FUNC(rethrow)
DEFAULT_GET_CILKRTS_FUNC(leave_frame)
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker)
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker_fast)
DEFAULT_GET_CILKRTS_FUNC(bind_thread_1)

DEFAULT_GET_CILKRTS_FUNC(cilk_for_32)
DEFAULT_GET_CILKRTS_FUNC(cilk_for_64)

// #define CILK_CSI_FUNC(name, CGF) Get_cilk_##name(CGF)

// #define GET_CILK_CSI_FUNC(name)                                         \
//   static Function *Get_cilk_##name(Module& M) {             \
//     return cast<Function>(M.getOrInsertFunction(            \
//                                           "cilk_"#name,                 \
//                                           TypeBuilder<cilk_##name, false>::get(M.getContext()) \
//                                                                         )); \
//   }

// #define GET_CILK_CSI_FUNC2(name)                                        \
//   static Function *Get_cilk_##name(Module& M) {             \
//     return cast<Function>(M.getOrInsertFunction(            \
//                                           "cilk_"#name,                 \
//                                           TypeBuilder<cilk_##name, false>::get(M.getContext()) \
//                                                                         )); \
//   }

// GET_CILK_CSI_FUNC(enter_begin)
// GET_CILK_CSI_FUNC(enter_helper_begin)
// GET_CILK_CSI_FUNC(enter_end)
// GET_CILK_CSI_FUNC(detach_begin)
// GET_CILK_CSI_FUNC(detach_end)
// GET_CILK_CSI_FUNC2(spawn_prepare)
// GET_CILK_CSI_FUNC2(spawn_or_continue)
// GET_CILK_CSI_FUNC(sync_begin)
// GET_CILK_CSI_FUNC(sync_end)
// GET_CILK_CSI_FUNC(leave_begin)
// GET_CILK_CSI_FUNC(leave_end)

typedef std::map<LLVMContext*, StructType*> TypeBuilderCache;

/// Specializations of TypeBuilder for:
///   __cilkrts_pedigree,
///   __cilkrts_worker,
///   __cilkrts_stack_frame
template <bool X>
class TypeBuilder<__cilkrts_pedigree, X> {
public:
  static StructType *get(LLVMContext &C) {
    static TypeBuilderCache cache;
    TypeBuilderCache::iterator I = cache.find(&C);
    if (I != cache.end())
      return I->second;
    StructType *ExistingTy = StructType::lookupOrCreate(C, "struct.__cilkrts_pedigree");
    cache[&C] = ExistingTy;
    StructType *NewTy = StructType::create(C);
    NewTy->setBody(
        TypeBuilder<uint64_t,            X>::get(C), // rank
        TypeBuilder<__cilkrts_pedigree*, X>::get(C)  // next
                );
    if (ExistingTy->isOpaque())
      ExistingTy->setBody(NewTy->elements());
    else
      assert(ExistingTy->isLayoutIdentical(NewTy) &&
             "Conflicting definition of tye struct.__cilkrts_pedigree");
    return ExistingTy;
  }
  enum {
    rank,
    next
  };
};

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
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // protected_tail
          TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // ltq_limit
          TypeBuilder<int32_t,                 X>::get(C), // self
          TypeBuilder<void*,                   X>::get(C), // g
          TypeBuilder<void*,                   X>::get(C), // l
          TypeBuilder<void*,                   X>::get(C), // reducer_map
          TypeBuilder<__cilkrts_stack_frame*,  X>::get(C), // current_stack_frame
          TypeBuilder<void*,                   X>::get(C), // saved_protected_tail
          TypeBuilder<void*,                   X>::get(C), // sysdep
          TypeBuilder<__cilkrts_pedigree,      X>::get(C)  // pedigree
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
      assert((ExistingTy->getElementType(pedigree) ==
              TypeBuilder<__cilkrts_pedigree, X>::get(C)) &&
             "Invalid type for __cilkrts_worker.pedigree");
    }
    return ExistingTy;
  }
  enum {
    tail,
    head,
    exc,
    protected_tail,
    ltq_limit,
    self,
    g,
    l,
    reducer_map,
    current_stack_frame,
    saved_protected_tail,
    sysdep,
    pedigree
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
        TypeBuilder<int32_t,                X>::get(C), // size
        TypeBuilder<__cilkrts_stack_frame*, X>::get(C), // call_parent
        TypeBuilder<__cilkrts_worker*,      X>::get(C), // worker
        TypeBuilder<void*,                  X>::get(C), // except_data
        TypeBuilder<__CILK_JUMP_BUFFER,     X>::get(C), // ctx
        TypeBuilder<uint32_t,               X>::get(C), // mxcsr
        TypeBuilder<uint16_t,               X>::get(C), // fpcsr
        TypeBuilder<uint16_t,               X>::get(C), // reserved
        ExistingTy->isOpaque() ?
        StructType::get(
            TypeBuilder<__cilkrts_pedigree, X>::get(C)  // parent_pedigree
                        ) :
        ExistingTy->getStructElementType(parent_pedigree)
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
    size,
    call_parent,
    worker,
    except_data,
    ctx,
    mxcsr,
    fpcsr,
    reserved,
    parent_pedigree
  };
};

/// Helper typedefs for cilk struct TypeBuilders.
typedef TypeBuilder<__cilkrts_stack_frame, false> StackFrameBuilder;
typedef TypeBuilder<__cilkrts_worker, false> WorkerBuilder;
typedef TypeBuilder<__cilkrts_pedigree, false> PedigreeBuilder;

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
  FunctionType *FTy =
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
  B.CreateInsertValue(
      LoadField(B, SF, StackFrameBuilder::parent_pedigree),
      LoadField(B, W, WorkerBuilder::pedigree), { 0 });

  // sf->call_parent->parent_pedigree = w->pedigree;
  B.CreateInsertValue(
      LoadField(B,
                LoadField(B, SF, StackFrameBuilder::call_parent),
                StackFrameBuilder::parent_pedigree),
      LoadField(B, W, WorkerBuilder::pedigree), { 0 });

  // w->pedigree.rank = 0;
  {
    StructType *STy = PedigreeBuilder::get(Ctx);
    Type *Ty = STy->getElementType(PedigreeBuilder::rank);
    StoreField(B,
               ConstantInt::get(Ty, 0),
               GEP(B, W, WorkerBuilder::pedigree),
               PedigreeBuilder::rank);
  }

  // w->pedigree.next = &sf->spawn_helper_pedigree;
  StoreField(B,
             GEP(B, GEP(B, SF, StackFrameBuilder::parent_pedigree), 0),
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

    // if (instrument)
    //   // cilk_sync_begin
    //   B.CreateCall(CILK_CSI_FUNC(sync_begin, M), SF);

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
    B.CreateInsertValue(
        LoadField(B, SF, StackFrameBuilder::parent_pedigree),
        LoadField(B, LoadField(B, SF, StackFrameBuilder::worker,
                               /*isVolatile=*/true),
                  WorkerBuilder::pedigree), { 0 });

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
    // if (instrument)
    //   // cilk_sync_end
    //   B.CreateCall(CILK_CSI_FUNC(sync_end, M), SF);

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

  PointerType *WorkerPtrTy =
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
    Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
               ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
               SF, StackFrameBuilder::flags, /*isVolatile=*/true);
    B.CreateBr(Cont);
  }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
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
  Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);

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

    // if (instrument)
    //   // cilk_leave_begin
    //   B.CreateCall(CILK_CSI_FUNC(leave_begin, M), SF);

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
    // if (instrument)
    //   // cilk_leave_end
    //   B.CreateCall(CILK_CSI_FUNC(leave_end, M));
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const StringRef stack_frame_name = "__cilkrts_sf";
static const StringRef worker8_name = "__cilkrts_wc8";

// static Value *LookupStackFrame(Function &F) {
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
    curinst = dyn_cast<AllocaInst>(II);
    II++;
  } while (curinst != alloc);
  // Value *StackSave;
  IRBuilder<> IRB(&(F.getEntryBlock()), II);

  // if (instrument) {
  //   Type *Int8PtrTy = IRB.getInt8PtrTy();
  //   Value *ThisFn = ConstantExpr::getBitCast(&F, Int8PtrTy);
  //   Value *ReturnAddress =
  //     IRB.CreateCall(Intrinsic::getDeclaration(F.getParent(),
  //                                              Intrinsic::returnaddress),
  //                    IRB.getInt32(0));
  //   StackSave =
  //     IRB.CreateCall(Intrinsic::getDeclaration(F.getParent(),
  //                                              Intrinsic::stacksave));
  //   if (Helper) {
  //     Value *begin_args[3] = { alloc, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *F.getParent()),
  //                    begin_args);
  //   } else {
  //     Value *begin_args[4] = { IRB.getInt32(0), alloc, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_begin, *F.getParent()), begin_args);
  //   }
  // }
  Value *args[1] = { alloc };
  if (Helper || fastCilk)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *F.getParent()), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1, *F.getParent()), args);
  /* inst->insertAfter(alloc); */

  // if (instrument) {
  //   Value* end_args[2] = { alloc, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *F.getParent()), end_args);
  // }

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*F.getParent(), instrument),
                         args, "");
  }

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
    curinst = dyn_cast<AllocaInst>(II);
    II++;
  } while (curinst != SF);
  // Value *StackSave;
  IRBuilder<> IRB(&(extracted.getEntryBlock()), II);

  // if (instrument) {
  //   Type *Int8PtrTy = IRB.getInt8PtrTy();
  //   Value *ThisFn = ConstantExpr::getBitCast(&extracted, Int8PtrTy);
  //   Value *ReturnAddress =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::returnaddress),
  //                    IRB.getInt32(0));
  //   StackSave =
  //     IRB.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stacksave));
  //   if (SimpleHelper) {
  //     Value *begin_args[3] = { SF, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_helper_begin, *M), begin_args);
  //   } else {
  //     Value *begin_args[4] = { IRB.getInt32(0), SF, ThisFn, ReturnAddress };
  //     IRB.CreateCall(CILK_CSI_FUNC(enter_begin, *M), begin_args);
  //   }
  // }

  if (SimpleHelper)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *M), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1, *M), args);

  // if (instrument) {
  //   Value *end_args[2] = { SF, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  // }

  // Call __cilkrts_detach
  {
    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_begin, *M), args);

    IRB.CreateCall(CILKRTS_FUNC(detach, *M), args);

    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_end, *M));
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
    }
  }

  return true;
}

//##############################################################################

tapir::CilkABI::CilkABI() {}

/// \brief Get/Create the worker count for the spawning function.
Value* tapir::CilkABI::GetOrCreateWorker8(Function &F) {
  // Value* W8 = F.getValueSymbolTable()->lookup(worker8_name);
  // if (W8) return W8;
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  Value *P0 = B.CreateCall(CILKRTS_FUNC(get_nworkers, *F.getParent()));
  Value *P8 = B.CreateMul(P0, ConstantInt::get(P0->getType(), 8), worker8_name);
  return P8;
}

void tapir::CilkABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {
  Function &Fn = *(SI.getParent()->getParent());
  Module &M = *(Fn.getParent());

  Value *SF = GetOrInitCilkStackFrame(Fn, DetachCtxToStackFrame,
                                      /*isFast*/false, false);
  Value *args[] = { SF };
  assert( args[0] && "sync used in function without frame!" );
  CallInst *CI = CallInst::Create(GetCilkSyncFn(M, false), args, "",
                                  /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
}

Function *tapir::CilkABI::createDetach(DetachInst &detach,
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
                                      /*isFast=*/false, false);
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

  makeFunctionDetachable(*extracted, DetachCtxToStackFrame, false);

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

cl::opt<bool> fastCilk("fast-cilk", cl::init(false), cl::Hidden,
                       cl::desc("Attempt faster cilk call implementation"));
void tapir::CilkABI::preProcessFunction(Function &F) {
  if (fastCilk && F.getName()=="main") {
    IRBuilder<> start(F.getEntryBlock().getFirstNonPHIOrDbg());
    auto m = start.CreateCall(CILKRTS_FUNC(init, *F.getParent()));
    m->moveBefore(F.getEntryBlock().getTerminator());
  }
}

void tapir::CilkABI::postProcessFunction(Function &F) {
    inlineCilkFunctions(F);
}

void tapir::CilkABI::postProcessHelper(Function &F) {
    inlineCilkFunctions(F);
}


/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
///
/// This method assumes that the loop has a single loop latch.
Value* tapir::CilkABILoopSpawning::canonicalizeLoopLatch(PHINode *IV, Value *Limit) {
  Loop *L = OrigLoop;

  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition =
    Builder.CreateICmpULT(Builder.CreateAdd(IV,
                                            ConstantInt::get(IV->getType(), 1)),
                          Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  Builder.CreateCondBr(NewCondition, Header, ExitBlock);

  // Erase the old conditional branch.
  LatchBr->eraseFromParent();

  return NewCondition;
}

/// Top-level call to convert a Tapir loop to be processed using an appropriate
/// Cilk ABI call.
bool tapir::CilkABILoopSpawning::processLoop() {
  Loop *L = OrigLoop;

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();

  using namespace ore;

  // Check the exit blocks of the loop.
  if (!ExitBlock) {
    DEBUG(dbgs() << "LS loop does not contain valid exit block after latch.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "InvalidLatchExit",
                                        L->getStartLoc(),
                                        Header)
             << "invalid latch exit");
    return false;
  }

  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  for (const BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    if (!isa<UnreachableInst>(Exit->getTerminator())) {
      DEBUG(dbgs() << "LS loop contains a bad exit block " << *Exit);
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "BadExit",
                                          L->getStartLoc(),
                                          Header)
               << "bad exit block found");
      return false;
    }
  }

  Function *F = Header->getParent();
  Module* M = F->getParent();

  DEBUG(dbgs() << "LS loop header:" << *Header);
  DEBUG(dbgs() << "LS loop latch:" << *Latch);

  // DEBUG(dbgs() << "LS SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
  // DEBUG(dbgs() << "LS SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");

  /// Get loop limit.
  const SCEV *BETC = SE.getExitCount(L, Latch);
  const SCEV *Limit = SE.getAddExpr(BETC, SE.getOne(BETC->getType()));
  DEBUG(dbgs() << "LS Loop limit: " << *Limit << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getExitCount(L, Latch);
  // DEBUG(dbgs() << "LS predicated loop limit: " << *PLimit << "\n");
  // emitAnalysis(LoopSpawningReport()
  //              << "computed loop limit " << *Limit << "\n");
  if (SE.getCouldNotCompute() == Limit) {
    DEBUG(dbgs() << "SE could not compute loop limit.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "UnknownLoopLimit",
                                        L->getStartLoc(),
                                        Header)
             << "could not compute limit");
    return false;
  }
  // ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "LoopLimit", L->getStartLoc(),
  //                                     Header)
  //          << "loop limit: " << NV("Limit", Limit));
  /// Clean up the loop's induction variables.
  PHINode *CanonicalIV = canonicalizeIVs(Limit->getType());
  if (!CanonicalIV) {
    DEBUG(dbgs() << "Could not get canonical IV.\n");
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
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    if (CanonicalIV == PN) continue;
    // dbgs() << "IV " << *PN;
    const SCEV *S = SE.getSCEV(PN);
    // dbgs() << " SCEV " << *S << "\n";
    if (SE.getCouldNotCompute() == S) {
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Could not compute the scalar evolution.\n");
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoSCEV", PN)
               << "could not compute scalar evolution of "
               << NV("PHINode", PN));
      CanRemoveIVs = false;
    }
  }

  if (!CanRemoveIVs) {
    DEBUG(dbgs() << "Could not compute scalar evolutions for all IV's.\n");
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
    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV) continue;
      const SCEV *S = SE.getSCEV(PN);
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
      PN->replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(PN);
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
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    DEBUG({
        const SCEVAddRecExpr *PNSCEV =
          dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
        assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr");
        assert(PNSCEV->getStart()->isZero() &&
               "PHINode SCEV does not start at 0");
        dbgs() << "LS step recurrence for SCEV " << *PNSCEV << " is "
               << *(PNSCEV->getStepRecurrence(SE)) << "\n";
        assert(PNSCEV->getStepRecurrence(SE)->isOne() &&
               "PHINode SCEV step is not 1");
      });
    if (ConstantInt *C =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Preheader))) {
      if (C->isZero())
        IVs.push_back(PN);
    } else {
      AllCanonical = false;
      DEBUG(dbgs() << "Remaining non-canonical PHI Node found: " << *PN << "\n");
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Found a remaining non-canonical IV.\n");
      ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "NonCanonicalIV", PN)
               << "found a remaining noncanonical IV");
    }
  }
  if (!AllCanonical)
    return false;

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      Preheader->getTerminator());
  DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

  // Canonicalize the loop latch.
  Value *NewCond = canonicalizeLoopLatch(CanonicalIV, LimitVar);

  /// Clone the loop into a new function.

  // Get the inputs and outputs for the Loop blocks.
  SetVector<Value*> Inputs, Outputs;
  SetVector<Value*> BodyInputs, BodyOutputs;
  ValueToValueMapTy VMap, InputMap;
  std::vector<BasicBlock *> LoopBlocks;
  AllocaInst* closure;
  // Add start iteration, end iteration, and grainsize to inputs.
  {
    LoopBlocks = L->getBlocks();
    // // Add exit blocks terminated by unreachable.  There should not be any other
    // // exit blocks in the loop.
    // SmallSet<BasicBlock *, 4> UnreachableExits;
    // for (BasicBlock *Exit : ExitBlocks) {
    //   if (Exit == ExitBlock) continue;
    //   assert(isa<UnreachableInst>(Exit->getTerminator()) &&
    //          "Found problematic exit block.");
    //   UnreachableExits.insert(Exit);
    // }

    // // Add unreachable and exception-handling exits to the set of loop blocks to
    // // clone.
    // for (BasicBlock *BB : UnreachableExits)
    //   LoopBlocks.push_back(BB);
    // for (BasicBlock *BB : EHExits)
    //   LoopBlocks.push_back(BB);

    // DEBUG({
    //     dbgs() << "LoopBlocks: ";
    //     for (BasicBlock *LB : LoopBlocks)
    //       dbgs() << LB->getName() << "("
    //              << *(LB->getTerminator()) << "), ";
    //     dbgs() << "\n";
    //   });

    // Get the inputs and outputs for the loop body.
    {
      // CodeExtractor Ext(LoopBlocks, DT);
      // Ext.findInputsOutputs(BodyInputs, BodyOutputs);
      SmallPtrSet<BasicBlock *, 32> Blocks;
      for (BasicBlock *BB : LoopBlocks)
        Blocks.insert(BB);
      findInputsOutputs(Blocks, BodyInputs, BodyOutputs);
    }

    // Add argument for start of CanonicalIV.
    DEBUG({
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

    // Add argument for end.
    Value* ea;
    if (isa<Constant>(LimitVar)) {
      Argument *EndArg = new Argument(LimitVar->getType(), "end");
      Inputs.insert(EndArg);
      ea = InputMap[LimitVar] = EndArg;
    } else {
      Inputs.insert(LimitVar);
      ea = InputMap[LimitVar] = LimitVar;
    }

    // Put all of the inputs together, and clear redundant inputs from
    // the set for the loop body.
    SmallVector<Value*, 8> BodyInputsToRemove;
    SmallVector<Value*, 8> StructInputs;
    SmallVector<Type*, 8> StructIT;
    for (Value *V : BodyInputs) {
      if (!Inputs.count(V)) {
        StructInputs.push_back(V);
        StructIT.push_back(V->getType());
      }
      else
        BodyInputsToRemove.push_back(V);
    }
    StructType* ST = StructType::create(StructIT);
    IRBuilder<> B(L->getLoopPreheader()->getTerminator());
    IRBuilder<> B2(L->getHeader()->getFirstNonPHIOrDbgOrLifetime());
    closure = B.CreateAlloca(ST);
    for(unsigned i=0; i<StructInputs.size(); i++) {
      B.CreateStore(StructInputs[i], B.CreateConstGEP2_32(ST, closure, 0, i));
      auto l2 = B2.CreateLoad(B2.CreateConstGEP2_32(ST, closure, 0, i));
      auto UI = StructInputs[i]->use_begin(), E = StructInputs[i]->use_end();
      for (; UI != E;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (Usr && !L->contains(Usr->getParent()))
          continue;
        U.set(l2);
      }
    }
    Inputs.insert(closure);
    //errs() << "<B>\n";
    //for(auto& a : Inputs) a->dump();
    //errs() << "</B>\n";
    //StartArg->dump();
    //ea->dump();
    Inputs.remove(StartArg);
    Inputs.insert(StartArg);
    Inputs.remove(ea);
    Inputs.insert(ea);
    //errs() << "<A>\n";
    //for(auto& a : Inputs) a->dump();
    //errs() << "</A>\n";
    for (Value *V : BodyInputsToRemove)
      BodyInputs.remove(V);
    assert(0 == BodyOutputs.size() &&
           "All results from parallel loop should be passed by memory already.");
  }
  DEBUG({
      for (Value *V : Inputs)
        dbgs() << "EL input: " << *V << "\n";
      for (Value *V : Outputs)
        dbgs() << "EL output: " << *V << "\n";
    });


  Function *Helper;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.

    // LowerDbgDeclare(*(Header->getParent()));

    Helper = CreateHelper(Inputs, Outputs, L->getBlocks(),
                          Header, Preheader, ExitBlock/*L->getExitBlock()*/,
                          VMap, M,
                          F->getSubprogram() != nullptr, Returns, ".ls",
                          nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning loop.");

    // Use a fast calling convention for the helper.
    //Helper->setCallingConv(CallingConv::Fast);
    // Helper->setCallingConv(Header->getParent()->getCallingConv());
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
    DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
    for (PHINode *IV : IVs) {
      if (CanonicalIV == IV) continue;

      // Get the value of the IV at the start iteration.
      DEBUG(dbgs() << "IV " << *IV);
      const SCEV *IVSCEV = SE.getSCEV(IV);
      DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
      const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
      const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
      DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
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

  // If the loop limit is constant, then rewrite the loop latch
  // condition to use the end-iteration argument.
  if (isa<Constant>(LimitVar)) {
    CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
    assert(HelperCond->getOperand(1) == LimitVar);
    IRBuilder<> Builder(HelperCond);
    Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                 VMap[InputMap[LimitVar]]);
    HelperCond->replaceAllUsesWith(NewHelperCond);
    HelperCond->eraseFromParent();
  }

  // For debugging:
  BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
  {
    Value* v = &*Helper->arg_begin();
    auto UI = v->use_begin(), E = v->use_end();
    for (; UI != E;) {
      Use &U = *UI;
      ++UI;
      auto *Usr = dyn_cast<Instruction>(U.getUser());
      Usr->moveBefore(Helper->getEntryBlock().getTerminator());

      auto UI2 = Usr->use_begin(), E2 = Usr->use_end();
      for (; UI2 != E2;) {
        Use &U2 = *UI2;
        ++UI2;
        auto *Usr2 = dyn_cast<Instruction>(U2.getUser());
        Usr2->moveBefore(Helper->getEntryBlock().getTerminator());
      }
    }
  }

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Add call to new helper function in original function.
  {
    // Setup arguments for call.
    SetVector<Value*> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.insert(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add loop limit.
    TopCallArgs.insert(LimitVar);
    // Add grainsize.
    //TopCallArgs.insert(GrainVar);
    // Add the rest of the arguments.
    for (Value *V : BodyInputs)
      TopCallArgs.insert(V);

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());

    Function* F;
    if( ((IntegerType*)LimitVar->getType())->getBitWidth() == 32 )
      F = CILKRTS_FUNC(cilk_for_32, *M);
    else {
      assert( ((IntegerType*)LimitVar->getType())->getBitWidth() == 64 );
      F = CILKRTS_FUNC(cilk_for_64, *M);
    }
    Value* args[] = {
      Builder.CreatePointerCast(Helper, F->getFunctionType()->getParamType(0)),
      Builder.CreatePointerCast(closure, F->getFunctionType()->getParamType(1)),
      LimitVar,
      ConstantInt::get(IntegerType::get(F->getContext(), sizeof(int)*8),0)
    };

    /*CallInst *TopCall = */Builder.CreateCall(F, args);

    // Use a fast calling convention for the helper.
    //TopCall->setCallingConv(CallingConv::Fast);
    // TopCall->setCallingConv(Helper->getCallingConv());
    //TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // // Update CG graph with the call we just added.
    // CG[F]->addCalledFunction(TopCall, CG[Helper]);
  }

  ++LoopsConvertedToCilkABI;

  unlinkLoop();

  return Helper;
}
