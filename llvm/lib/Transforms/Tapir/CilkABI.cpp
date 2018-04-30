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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
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

using __CILK_JUMP_BUFFER = void *[5];

using __cilkrts_pedigree = CilkABI::__cilkrts_pedigree;
using __cilkrts_stack_frame = CilkABI::__cilkrts_stack_frame;
using __cilkrts_worker = CilkABI::__cilkrts_worker;

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


using cilk32_t = uint32_t;
using cilk64_t = uint64_t;
using __cilk_abi_f32_t = void (*)(void *data, cilk32_t low, cilk32_t high);
using __cilk_abi_f64_t = void (*)(void *data, cilk64_t low, cilk64_t high);

using __cilkrts_init = void ();

using __cilkrts_enter_frame_1 = void (__cilkrts_stack_frame *sf);
using __cilkrts_enter_frame_fast_1 = void (__cilkrts_stack_frame *sf);
using __cilkrts_leave_frame = void (__cilkrts_stack_frame *sf);
using __cilkrts_rethrow = void (__cilkrts_stack_frame *sf);
using __cilkrts_sync = void (__cilkrts_stack_frame *sf);
using __cilkrts_sync_nothrow = void (__cilkrts_stack_frame *sf);
using __cilkrts_detach = void (__cilkrts_stack_frame *sf);
using __cilkrts_pop_frame = void (__cilkrts_stack_frame *sf);
using __cilkrts_get_nworkers = int ();
using __cilkrts_get_tls_worker = __cilkrts_worker *();
using __cilkrts_get_tls_worker_fast = __cilkrts_worker *();
using __cilkrts_bind_thread_1 = __cilkrts_worker *();

using cilk_func = void (__cilkrts_stack_frame *);

using cilk_enter_begin = void (uint32_t, __cilkrts_stack_frame *, void *,
                               void *);
using cilk_enter_helper_begin = void (__cilkrts_stack_frame *, void *, void *);
using cilk_enter_end = void (__cilkrts_stack_frame *, void *);
using cilk_detach_begin = void (__cilkrts_stack_frame *);
using cilk_detach_end = void ();
using cilk_spawn_prepare = void (__cilkrts_stack_frame *);
using cilk_spawn_or_continue = void (int);
using cilk_sync_begin = void (__cilkrts_stack_frame *);
using cilk_sync_end = void (__cilkrts_stack_frame *);
using cilk_leave_begin = void (__cilkrts_stack_frame *);
using cilk_leave_end = void ();
using __cilkrts_cilk_for_32 = void (__cilk_abi_f32_t body, void *data,
                                    cilk32_t count, int grain);
using __cilkrts_cilk_for_64 = void (__cilk_abi_f64_t body, void *data,
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

using TypeBuilderCache = std::map<LLVMContext *, StructType *>;

namespace llvm {
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
} // end namespace llvm

/// Helper type definitions for cilk struct TypeBuilders.
using StackFrameBuilder = TypeBuilder<__cilkrts_stack_frame, false>;
using WorkerBuilder = TypeBuilder<__cilkrts_worker, false>;
using PedigreeBuilder = TypeBuilder<__cilkrts_pedigree, false>;

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

/// \brief Emit inline assembly code to save the floating point
/// state, for x86 Only.
static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
  using AsmPrototype = void (uint32_t *, uint16_t *);
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
  Triple T(M.getTargetTriple()); 
  if(T.getArch() == Triple::x86 || T.getArch() == Triple::x86_64) 
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
///   sf->call_parent = nullptr;
/// }
static Function *Get__cilkrts_pop_frame(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_pop_frame", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // sf->worker->current_stack_frame = sf->call_parent;
  StoreSTyField(B, DL, WorkerBuilder::get(Ctx),
                LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                             StackFrameBuilder::call_parent,
                             /*isVolatile=*/false,
                             AtomicOrdering::NotAtomic),
                LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                             StackFrameBuilder::worker,
                             /*isVolatile=*/false,
                             AtomicOrdering::Acquire),
                WorkerBuilder::current_stack_frame,
                /*isVolatile=*/false,
                AtomicOrdering::Release);

  // sf->call_parent = nullptr;
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                Constant::getNullValue(
                    TypeBuilder<__cilkrts_stack_frame*, false>::get(Ctx)),
                SF, StackFrameBuilder::call_parent, /*isVolatile=*/false,
                AtomicOrdering::Release);

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
static Function *Get__cilkrts_detach(Module &M) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_detach", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // struct __cilkrts_worker *w = sf->worker;
  Value *W = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                          StackFrameBuilder::worker, /*isVolatile=*/false,
                          AtomicOrdering::NotAtomic);

  // __cilkrts_stack_frame *parent = sf->call_parent;
  Value *Parent = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                               StackFrameBuilder::call_parent,
                               /*isVolatile=*/false,
                               AtomicOrdering::NotAtomic);

  // __cilkrts_stack_frame *volatile *tail = w->tail;
  Value *Tail = LoadSTyField(B, DL, WorkerBuilder::get(Ctx), W,
                             WorkerBuilder::tail, /*isVolatile=*/false,
                             AtomicOrdering::Acquire);

  // sf->spawn_helper_pedigree = w->pedigree;
  Value *WorkerPedigree = LoadSTyField(B, DL, WorkerBuilder::get(Ctx), W,
                                       WorkerBuilder::pedigree);
  Value *NewHelperPedigree = B.CreateInsertValue(
      LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                   StackFrameBuilder::parent_pedigree), WorkerPedigree, { 0 });
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), NewHelperPedigree, SF,
                StackFrameBuilder::parent_pedigree);

  // parent->parent_pedigree = w->pedigree;
  Value *NewParentPedigree = B.CreateInsertValue(
      LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), Parent,
                   StackFrameBuilder::parent_pedigree), WorkerPedigree, { 0 });
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), NewParentPedigree, Parent,
                StackFrameBuilder::parent_pedigree);

  // w->pedigree.rank = 0;
  {
    StructType *STy = PedigreeBuilder::get(Ctx);
    Type *Ty = STy->getElementType(PedigreeBuilder::rank);
    StoreSTyField(B, DL, STy, ConstantInt::get(Ty, 0),
                  GEP(B, W, WorkerBuilder::pedigree), PedigreeBuilder::rank,
                  /*isVolatile=*/false, AtomicOrdering::Release);
  }

  // w->pedigree.next = &sf->spawn_helper_pedigree;
  StoreSTyField(B, DL, PedigreeBuilder::get(Ctx),
                GEP(B, GEP(B, SF, StackFrameBuilder::parent_pedigree), 0),
                GEP(B, W, WorkerBuilder::pedigree), PedigreeBuilder::next,
                /*isVolatile=*/false, AtomicOrdering::Release);

  // StoreStore_fence();
  B.CreateFence(AtomicOrdering::Release);

  // *tail++ = parent;
  B.CreateStore(Parent, Tail, /*isVolatile=*/true);
  Tail = B.CreateConstGEP1_32(Tail, 1);

  // w->tail = tail;
  StoreSTyField(B, DL, WorkerBuilder::get(Ctx), Tail, W, WorkerBuilder::tail,
                /*isVolatile=*/false, AtomicOrdering::Release);

  // sf->flags |= CILK_FRAME_DETACHED;
  {
    Value *F = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                            StackFrameBuilder::flags, /*isVolatile=*/false,
                            AtomicOrdering::Acquire);
    F = B.CreateOr(F, ConstantInt::get(F->getType(), CILK_FRAME_DETACHED));
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), F, SF,
                  StackFrameBuilder::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);
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
    Value *Flags = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                StackFrameBuilder::flags, /*isVolatile=*/false,
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
        LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                     StackFrameBuilder::parent_pedigree),
        LoadSTyField(B, DL, WorkerBuilder::get(Ctx),
                     LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                  StackFrameBuilder::worker,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire),
                     WorkerBuilder::pedigree), { 0 });
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), NewParentPedigree, SF,
                  StackFrameBuilder::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF, M);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Excepting);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(sf);
    B.CreateCall(CILKRTS_FUNC(sync, M), SF);
    B.CreateBr(Exit);
  }

  // Excepting
  {
    IRBuilder<> B(Excepting);
    if (Rethrow) {
      // if (sf->flags & CILK_FRAME_EXCEPTING)
      Value *Flags = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                  StackFrameBuilder::flags,
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
    B.CreateCall(CILKRTS_FUNC(rethrow, M), SF)->setDoesNotReturn();
    B.CreateUnreachable();
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Worker = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                 StackFrameBuilder::worker,
                                 /*isVolatile=*/false,
                                 AtomicOrdering::Acquire);
    Value *Pedigree = GEP(B, Worker, WorkerBuilder::pedigree);
    Value *Rank = GEP(B, Pedigree, PedigreeBuilder::rank);
    unsigned RankAlignment = GetAlignment(DL, PedigreeBuilder::get(Ctx),
                                          PedigreeBuilder::rank);
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

  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);
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
static Function *GetCilkSyncNoThrowFn(Module &M, bool instrument = false) {
  Function *Fn = nullptr;

  if (GetOrCreateFunction<cilk_func>("__cilk_sync_nothrow", M, Fn,
                                     Function::InternalLinkage,
                                     /*doesNotThrow*/true))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();
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
    Value *Flags = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                StackFrameBuilder::flags, /*isVolatile=*/false,
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
        LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                     StackFrameBuilder::parent_pedigree),
        LoadSTyField(B, DL, WorkerBuilder::get(Ctx),
                     LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                  StackFrameBuilder::worker,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire),
                     WorkerBuilder::pedigree), { 0 });
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), NewParentPedigree, SF,
                  StackFrameBuilder::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF, M);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Exit);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(sf);
    B.CreateCall(CILKRTS_FUNC(sync, M), SF);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Worker = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                 StackFrameBuilder::worker,
                                 /*isVolatile=*/false,
                                 AtomicOrdering::Acquire);
    Value *Pedigree = GEP(B, Worker, WorkerBuilder::pedigree);
    Value *Rank = GEP(B, Pedigree, PedigreeBuilder::rank);
    unsigned RankAlignment = GetAlignment(DL, PedigreeBuilder::get(Ctx),
                                          PedigreeBuilder::rank);
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
  const DataLayout &DL = M.getDataLayout();

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
    // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
    if (fastCilk)
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast, M));
    else
      W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

    // if (w == 0)
    Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    B.CreateCondBr(Cond, SlowPath, FastPath);
  }
  // Block  (SlowPath)
  CallInst *Wslow = nullptr;
  {
    IRBuilder<> B(SlowPath);
    // w = __cilkrts_bind_thread_1();
    Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1, M));
    // sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
    Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                  ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
                  SF, StackFrameBuilder::flags, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    B.CreateBr(Cont);
  }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    // sf->flags = CILK_FRAME_VERSION;
    Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                  ConstantInt::get(Ty, CILK_FRAME_VERSION),
                  SF, StackFrameBuilder::flags, /*isVolatile=*/false,
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
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                  LoadSTyField(B, DL, WorkerBuilder::get(Ctx), W,
                               WorkerBuilder::current_stack_frame,
                               /*isVolatile=*/false,
                               AtomicOrdering::Acquire),
                  SF, StackFrameBuilder::call_parent, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // sf->worker = w;
    StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), W, SF,
                  StackFrameBuilder::worker, /*isVolatile=*/false,
                  AtomicOrdering::Release);
    // w->current_stack_frame = sf;
    StoreSTyField(B, DL, WorkerBuilder::get(Ctx), SF, W,
                  WorkerBuilder::current_stack_frame, /*isVolatile=*/false,
                  AtomicOrdering::Release);

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
  const DataLayout &DL = M.getDataLayout();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);

  IRBuilder<> B(Entry);
  Value *W;

  // struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
  // if (fastCilk)
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker_fast, M));
  // else
  //   W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));

  StructType *SFTy = StackFrameBuilder::get(Ctx);
  Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);

  // sf->flags = CILK_FRAME_VERSION;
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                ConstantInt::get(Ty, CILK_FRAME_VERSION),
                SF, StackFrameBuilder::flags, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // sf->call_parent = w->current_stack_frame;
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx),
                LoadSTyField(B, DL, WorkerBuilder::get(Ctx), W,
                             WorkerBuilder::current_stack_frame,
                             /*isVolatile=*/false,
                             AtomicOrdering::Acquire),
                SF, StackFrameBuilder::call_parent, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // sf->worker = w;
  StoreSTyField(B, DL, StackFrameBuilder::get(Ctx), W, SF,
                StackFrameBuilder::worker, /*isVolatile=*/false,
                AtomicOrdering::Release);
  // w->current_stack_frame = sf;
  StoreSTyField(B, DL, WorkerBuilder::get(Ctx), SF, W,
                WorkerBuilder::current_stack_frame, /*isVolatile=*/false,
                AtomicOrdering::Release);

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
    B.CreateCall(CILKRTS_FUNC(pop_frame, M), SF);

    // if (sf->flags != CILK_FRAME_VERSION)
    Value *Flags = LoadSTyField(B, DL, StackFrameBuilder::get(Ctx), SF,
                                StackFrameBuilder::flags, /*isVolatile=*/false,
                                AtomicOrdering::Acquire);
    Value *Cond = B.CreateICmpNE(
        Flags, ConstantInt::get(Flags->getType(), CILK_FRAME_VERSION));
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

/// \brief Create the __cilkrts_stack_frame for the spawning function.
static AllocaInst *CreateStackFrame(Function &F) {
  // assert(!LookupStackFrame(F) && "already created the stack frame");

  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *SFTy = StackFrameBuilder::get(Ctx);

  // Instruction *I = F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime();

  // AllocaInst *SF = new AllocaInst(SFTy, DL.getAllocaAddrSpace(),
  //                                 /*ArraySize*/nullptr, /*Align*/8,
  //                                 /*Name*/stack_frame_name, /*InsertBefore*/I);
  // if (!I)
  //   F.getEntryBlock().getInstList().push_back(SF);
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *SF = B.CreateAlloca(SFTy, DL.getAllocaAddrSpace(),
                                  /*ArraySize*/nullptr,
                                  /*Name*/stack_frame_name);
  SF->setAlignment(8);

  return SF;
}

static Value *GetOrInitCilkStackFrame(
    Function &F, ValueToValueMapTy &DetachCtxToStackFrame,
    bool Helper, bool instrument = false) {
  if (DetachCtxToStackFrame.count(&F))
    return DetachCtxToStackFrame[&F];
  // if (Value *V = DetachCtxToStackFrame[&F])
  //   return V;

  Module *M = F.getParent();

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
  Value *args[1] = { SF };
  if (Helper || fastCilk)
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *M), args);
  else
    IRB.CreateCall(CILKRTS_FUNC(enter_frame_1, *M), args);

  // if (instrument) {
  //   Value* end_args[2] = { SF, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  // }

  EscapeEnumerator EE(F, "cilkabi_epilogue", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (isa<ReturnInst>(AtExit->GetInsertPoint()))
      AtExit->CreateCall(GetCilkParentEpilogue(*M, instrument), args, "");
    else if (ResumeInst *RI = dyn_cast<ResumeInst>(AtExit->GetInsertPoint())) {
      // /*
      //   sf.flags = sf.flags | CILK_FRAME_EXCEPTING;
      //   sf.except_data = Exn;
      // */
      // IRBuilder<> B(RI);
      // Value *Exn = AtExit->CreateExtractValue(RI->getValue(),
      //                                         ArrayRef<unsigned>(0));
      // Value *Flags = LoadSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), SF,
      //                             StackFrameBuilder::flags,
      //                             /*isVolatile=*/false,
      //                             AtomicOrdering::Acquire);
      // Flags = AtExit->CreateOr(Flags,
      //                          ConstantInt::get(Flags->getType(),
      //                                           CILK_FRAME_EXCEPTING));
      // StoreSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), Flags, SF,
      //               StackFrameBuilder::flags, /*isVolatile=*/false,
      //               AtomicOrdering::Release);
      // StoreSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), Exn, SF,
      //               StackFrameBuilder::except_data, /*isVolatile=*/false,
      //               AtomicOrdering::Release);
      /*
        __cilkrts_pop_frame(&sf);
        if (sf->flags)
          __cilkrts_leave_frame(&sf);
      */
      AtExit->CreateCall(GetCilkParentEpilogue(*M, instrument), args, "");
    }
  }

  return SF;
}

static bool makeFunctionDetachable(
    Function &Extracted, ValueToValueMapTy &DetachCtxToStackFrame,
    bool instrument = false) {
  Module *M = Extracted.getParent();
  LLVMContext& Ctx = M->getContext();
  const DataLayout& DL = M->getDataLayout();
  /*
    __cilkrts_stack_frame sf;
    __cilkrts_enter_frame_fast_1(&sf);
    __cilkrts_detach();
    *x = f(y);
  */

  AllocaInst *SF = CreateStackFrame(Extracted);
  DetachCtxToStackFrame[&Extracted] = SF;
  assert(SF && "Error creating Cilk stack frame in helper.");
  Value *args[1] = { SF };

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

  IRB.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, *M), args);

  // if (instrument) {
  //   Value *end_args[2] = { SF, StackSave };
  //   IRB.CreateCall(CILK_CSI_FUNC(enter_end, *M), end_args);
  // }

  // __cilkrts_detach()
  {
    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_begin, *M), args);

    IRB.CreateCall(CILKRTS_FUNC(detach, *M), args);

    // if (instrument)
    //   IRB.CreateCall(CILK_CSI_FUNC(detach_end, *M));
  }

  EscapeEnumerator EE(Extracted, "cilkabi_epilogue", false);
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
      Value *Flags = LoadSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), SF,
                                  StackFrameBuilder::flags,
                                  /*isVolatile=*/false,
                                  AtomicOrdering::Acquire);
      Flags = AtExit->CreateOr(Flags,
                               ConstantInt::get(Flags->getType(),
                                                CILK_FRAME_EXCEPTING));
      StoreSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), Flags, SF,
                    StackFrameBuilder::flags, /*isVolatile=*/false,
                    AtomicOrdering::Release);
      StoreSTyField(*AtExit, DL, StackFrameBuilder::get(Ctx), Exn, SF,
                    StackFrameBuilder::except_data, /*isVolatile=*/false,
                    AtomicOrdering::Release);
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

// CilkABI::CilkABI() {}

/// \brief Lower a call to get the grainsize of this Tapir loop.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
Value *CilkABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  Module *M = GrainsizeCall->getModule();
  IRBuilder<> Builder(GrainsizeCall);

  // Get 8 * workers
  Value *Workers = Builder.CreateCall(CILKRTS_FUNC(get_nworkers, *M));
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

void CilkABI::createSync(SyncInst &SI) {
  Function &Fn = *(SI.getParent()->getParent());
  Module &M = *(Fn.getParent());

  Value *SF = GetOrInitCilkStackFrame(Fn, DetachCtxToStackFrame,
                                      /*isFast*/false, false);
  Value *args[] = { SF };
  assert(args[0] && "sync used in function without frame!");
  CallInst *CI;

  if (Fn.doesNotThrow())
    CI = CallInst::Create(GetCilkSyncNoThrowFn(M, false), args, "",
                          /*insert before*/&SI);
  else
    CI = CallInst::Create(GetCilkSyncFn(M, false), args, "",
                          /*insert before*/&SI);
  CI->setDebugLoc(SI.getDebugLoc());
  BasicBlock *Succ = SI.getSuccessor(0);
  SI.eraseFromParent();
  BranchInst::Create(Succ, CI->getParent());
  // Mark this function as stealable.
  Fn.addFnAttr(Attribute::Stealable);
}

void CilkABI::processOutlinedTask(Function &F) {
  makeFunctionDetachable(F, DetachCtxToStackFrame, false);
}

void CilkABI::processSpawner(Function &F) {
  GetOrInitCilkStackFrame(F, DetachCtxToStackFrame,
                          /*isFast=*/false, false);

  // Mark this function as stealable.
  F.addFnAttr(Attribute::Stealable);
}

void CilkABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;

  Function &F = *ReplCall->getParent()->getParent();
  Module &M = *F.getParent();
  assert(DetachCtxToStackFrame.count(&F) &&
         "No frame found for spawning task.");
  Value *SF = DetachCtxToStackFrame[&F];
  // assert(SF && "No frame found for spawning task");

  // Split the basic block containing the detach replacement just before the
  // start of the detach-replacement instructions.
  BasicBlock *DetBlock = ReplStart->getParent();
  BasicBlock *CallBlock = SplitBlock(DetBlock, ReplStart, &DT);

  // Emit a Cilk setjmp at the end of the block preceding the split-off detach
  // replacement.
  Instruction *SetJmpPt = DetBlock->getTerminator();
  IRBuilder<> B(SetJmpPt);
  Value *SetJmpRes = EmitCilkSetJmp(B, SF, M);

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

Function *CilkABI::createDetach(DetachInst &Detach,
                                DominatorTree &DT, AssumptionCache &AC) {
  BasicBlock *Detacher = Detach.getParent();
  Function &F = *(Detacher->getParent());

  BasicBlock *Continue = Detach.getContinue();

  Module *M = F.getParent();
  //replace with branch to succesor
  //entry / cilk.spawn.savestate
  Value *SF = GetOrInitCilkStackFrame(F, DetachCtxToStackFrame,
                                      /*isFast=*/false, false);
  assert(SF && "null stack frame unexpected");

  BasicBlock *CallBlock = SplitBlock(Detacher, &Detach, &DT);
  Instruction *SetJmpPt = Detacher->getTerminator();

  Instruction *CallSite = nullptr;
  Function *Extracted = extractDetachBodyToFunction(Detach, DT, AC, &CallSite);
  assert(Extracted && "could not extract detach body to function");

  // Unlink the detached CFG in the original function.  The heavy lifting of
  // removing the outlined detached-CFG is left to subsequent DCE.

  // Replace the detach with a branch to the continuation.
  BranchInst *ContinueBr = BranchInst::Create(Continue);
  ReplaceInstWithInst(&Detach, ContinueBr);

  Value *SetJmpRes;
  {
    IRBuilder<> B(SetJmpPt);
    SetJmpRes = EmitCilkSetJmp(B, SF, *M);
  }

  // Conditionally call the new helper function based on the result of the
  // setjmp.
  {
    // BasicBlock *CallBlock = SplitBlock(CallSite->getParent(), CallSite, &DT);
    BasicBlock *CallCont;
    if (InvokeInst *II = dyn_cast<InvokeInst>(CallSite))
      CallCont = SplitEdge(CallBlock, II->getNormalDest(), &DT);
    else // isa<CallInst>(CallSite)
      CallCont = SplitBlock(CallBlock, CallBlock->getTerminator(), &DT);

    IRBuilder<> B(SetJmpPt);
    SetJmpRes = B.CreateICmpEQ(SetJmpRes,
                               ConstantInt::get(SetJmpRes->getType(), 0));
    B.CreateCondBr(SetJmpRes, CallBlock, CallCont);
    SetJmpPt->eraseFromParent();
  }

  // Mark this function as stealable.
  F.addFnAttr(Attribute::Stealable);

  makeFunctionDetachable(*Extracted, DetachCtxToStackFrame, false);

  return Extracted;
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
    B.CreateCall(CILKRTS_FUNC(init, *F.getParent()));
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


/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
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
    CilkForABI = CILKRTS_FUNC(cilk_for_32, *M);
  else if (LimitVar->getType()->isIntegerTy(64))
    CilkForABI = CILKRTS_FUNC(cilk_for_64, *M);
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
