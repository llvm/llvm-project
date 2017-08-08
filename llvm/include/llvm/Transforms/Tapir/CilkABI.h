//===- CilkABI.h - Interface to the Intel Cilk Plus runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//
#ifndef CILK_ABI_H_
#define CILK_ABI_H_

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <deque>

extern llvm::cl::opt<bool> fastCilk;

namespace {

typedef void *__CILK_JUMP_BUFFER[5];

struct __cilkrts_pedigree {};
struct __cilkrts_stack_frame {};
struct __cilkrts_worker {};
struct global_state_t {};

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
  static llvm::Function *Get__cilkrts_##name(llvm::Module& M) {         \
    return llvm::cast<llvm::Function>(M.getOrInsertFunction(            \
                                          "__cilkrts_"#name,            \
                                          llvm::TypeBuilder<__cilkrts_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

//DEFAULT_GET_CILKRTS_FUNC(get_nworkers)
#pragma GCC diagnostic ignored "-Wunused-function"
static llvm::Function *Get__cilkrts_get_nworkers(llvm::Module& M) {
  llvm::LLVMContext &C = M.getContext();
  llvm::AttributeList AL;
  AL = AL.addAttribute(C, llvm::AttributeList::FunctionIndex,
                       llvm::Attribute::ReadNone);
  // AL = AL.addAttribute(C, llvm::AttributeSet::FunctionIndex,
  //                      llvm::Attribute::InaccessibleMemOnly);
  AL = AL.addAttribute(C, llvm::AttributeList::FunctionIndex,
                       llvm::Attribute::NoUnwind);
  llvm::Function *F = llvm::cast<llvm::Function>(
      M.getOrInsertFunction(
          "__cilkrts_get_nworkers",
          llvm::TypeBuilder<__cilkrts_get_nworkers, false>::get(C),
          AL));
  return F;
}

// TODO: set up these CILKRTS and CILK_CSI functions in a cleaner
// way so we don't need these pragmas.
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(init)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(sync)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(rethrow)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(leave_frame)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker_fast)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(bind_thread_1)

#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(cilk_for_32)
#pragma GCC diagnostic ignored "-Wunused-function"
DEFAULT_GET_CILKRTS_FUNC(cilk_for_64)

#define CILK_CSI_FUNC(name, CGF) Get_cilk_##name(CGF)

#define GET_CILK_CSI_FUNC(name)                                         \
  static llvm::Function *Get_cilk_##name(llvm::Module& M) {             \
    return llvm::cast<llvm::Function>(M.getOrInsertFunction(            \
                                          "cilk_"#name,                 \
                                          llvm::TypeBuilder<cilk_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

#define GET_CILK_CSI_FUNC2(name)                                        \
  static llvm::Function *Get_cilk_##name(llvm::Module& M) {             \
    return llvm::cast<llvm::Function>(M.getOrInsertFunction(            \
                                          "cilk_"#name,                 \
                                          llvm::TypeBuilder<cilk_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(enter_begin)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(enter_helper_begin)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(enter_end)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(detach_begin)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(detach_end)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC2(spawn_prepare)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC2(spawn_or_continue)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(sync_begin)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(sync_end)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(leave_begin)
#pragma GCC diagnostic ignored "-Wunused-function"
GET_CILK_CSI_FUNC(leave_end)

  typedef std::map<llvm::LLVMContext*, llvm::StructType*> TypeBuilderCache;

}  // namespace

namespace llvm {

/// Specializations of llvm::TypeBuilder for:
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
    StructType *ExistingTy = StructType::getOrCreate(C, "struct.__cilkrts_pedigree");
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
    StructType *Ty = StructType::getOrCreate(C, "struct.__cilkrts_worker");
    assert(Ty->isOpaque() &&
           "Conflicting definition of type struct.__cilkrts_worker.");
    cache[&C] = Ty;
    Ty->setBody(
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
        TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // saved_protected_tail
        TypeBuilder<void*,                   X>::get(C), // sysdep
        TypeBuilder<__cilkrts_pedigree,      X>::get(C)  // pedigree
                );
    return Ty;
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
    StructType *Ty = StructType::create(C, "struct.__cilkrts_stack_frame");
    cache[&C] = Ty;
    Ty->setBody(
        TypeBuilder<uint32_t,               X>::get(C), // flags
        TypeBuilder<int32_t,                X>::get(C), // size
        TypeBuilder<__cilkrts_stack_frame*, X>::get(C), // call_parent
        TypeBuilder<__cilkrts_worker*,      X>::get(C), // worker
        TypeBuilder<void*,                  X>::get(C), // except_data
        TypeBuilder<__CILK_JUMP_BUFFER,     X>::get(C), // ctx
        TypeBuilder<uint32_t,               X>::get(C), // mxcsr
        TypeBuilder<uint16_t,               X>::get(C), // fpcsr
        TypeBuilder<uint16_t,               X>::get(C), // reserved
        TypeBuilder<__cilkrts_pedigree,     X>::get(C)  // parent_pedigree
                );
    return Ty;
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

} // namespace llvm


////////////////////////////////////////////////////////////////////////////////

namespace llvm {
namespace cilk {

Value *GetOrCreateWorker8(Function &F);
void createSync(SyncInst &inst, ValueToValueMapTy &DetachCtxToStackFrame,
                bool instrument = false);

bool verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                       bool error = true);

bool populateDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                         SmallPtrSetImpl<BasicBlock *> &functionPieces,
                         SmallVectorImpl<BasicBlock *> &reattachB,
                         SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
                         bool replace, bool error = true);

Function *extractDetachBodyToFunction(DetachInst &Detach,
                                      DominatorTree &DT, AssumptionCache &AC,
                                      CallInst **call = nullptr);

Function *createDetach(DetachInst &Detach,
                       ValueToValueMapTy &DetachCtxToStackFrame,
                       DominatorTree &DT, AssumptionCache &AC,
                       bool instrument = false);

}  // end of cilk namespace
}  // end of llvm namespace

#endif
