//===- CoroShape.h - Coroutine info for lowering --------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file declares the shape info struct that is required by many coroutine
// utility methods.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COROUTINES_COROSHAPE_H
#define LLVM_TRANSFORMS_COROUTINES_COROSHAPE_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"

namespace llvm {

class CallGraph;

namespace coro {

enum class ABI {
  /// The "resume-switch" lowering, where there are separate resume and
  /// destroy functions that are shared between all suspend points.  The
  /// coroutine frame implicitly stores the resume and destroy functions,
  /// the current index, and any promise value.
  Switch,

  /// The "returned-continuation" lowering, where each suspend point creates a
  /// single continuation function that is used for both resuming and
  /// destroying.  Does not support promises.
  Retcon,

  /// The "unique returned-continuation" lowering, where each suspend point
  /// creates a single continuation function that is used for both resuming
  /// and destroying.  Does not support promises.  The function is known to
  /// suspend at most once during its execution, and the return value of
  /// the continuation is void.
  RetconOnce,

  /// The "async continuation" lowering, where each suspend point creates a
  /// single continuation function. The continuation function is available as an
  /// intrinsic.
  Async,
};

// Holds structural Coroutine Intrinsics for a particular function and other
// values used during CoroSplit pass.
struct Shape {
  CoroBeginInst *CoroBegin = nullptr;
  SmallVector<AnyCoroEndInst *, 4> CoroEnds;
  SmallVector<CoroSizeInst *, 2> CoroSizes;
  SmallVector<CoroAlignInst *, 2> CoroAligns;
  SmallVector<AnyCoroSuspendInst *, 4> CoroSuspends;
  SmallVector<CoroAwaitSuspendInst *, 4> CoroAwaitSuspends;
  SmallVector<CallInst *, 2> SymmetricTransfers;

  // Values invalidated by replaceSwiftErrorOps()
  SmallVector<CallInst *, 2> SwiftErrorOps;

  void clear() {
    CoroBegin = nullptr;
    CoroEnds.clear();
    CoroSizes.clear();
    CoroAligns.clear();
    CoroSuspends.clear();
    CoroAwaitSuspends.clear();
    SymmetricTransfers.clear();

    SwiftErrorOps.clear();

    FrameTy = nullptr;
    FramePtr = nullptr;
    AllocaSpillBlock = nullptr;
  }

  // Scan the function and collect the above intrinsics for later processing
  void analyze(Function &F, SmallVectorImpl<CoroFrameInst *> &CoroFrames,
               SmallVectorImpl<CoroSaveInst *> &UnusedCoroSaves);
  // If for some reason, we were not able to find coro.begin, bailout.
  void invalidateCoroutine(Function &F,
                           SmallVectorImpl<CoroFrameInst *> &CoroFrames);
  // Perform ABI related initial transformation
  void initABI();
  // Remove orphaned and unnecessary intrinsics
  void cleanCoroutine(SmallVectorImpl<CoroFrameInst *> &CoroFrames,
                      SmallVectorImpl<CoroSaveInst *> &UnusedCoroSaves);

  // Field indexes for special fields in the switch lowering.
  struct SwitchFieldIndex {
    enum {
      Resume,
      Destroy

      // The promise field is always at a fixed offset from the start of
      // frame given its type, but the index isn't a constant for all
      // possible frames.

      // The switch-index field isn't at a fixed offset or index, either;
      // we just work it in where it fits best.
    };
  };

  coro::ABI ABI;

  StructType *FrameTy = nullptr;
  Align FrameAlign;
  uint64_t FrameSize = 0;
  Value *FramePtr = nullptr;
  BasicBlock *AllocaSpillBlock = nullptr;

  struct SwitchLoweringStorage {
    SwitchInst *ResumeSwitch;
    AllocaInst *PromiseAlloca;
    BasicBlock *ResumeEntryBlock;
    unsigned IndexField;
    unsigned IndexAlign;
    unsigned IndexOffset;
    bool HasFinalSuspend;
    bool HasUnwindCoroEnd;
  };

  struct RetconLoweringStorage {
    Function *ResumePrototype;
    Function *Alloc;
    Function *Dealloc;
    BasicBlock *ReturnBlock;
    bool IsFrameInlineInStorage;
  };

  struct AsyncLoweringStorage {
    Value *Context;
    CallingConv::ID AsyncCC;
    unsigned ContextArgNo;
    uint64_t ContextHeaderSize;
    uint64_t ContextAlignment;
    uint64_t FrameOffset; // Start of the frame.
    uint64_t ContextSize; // Includes frame size.
    GlobalVariable *AsyncFuncPointer;

    Align getContextAlignment() const { return Align(ContextAlignment); }
  };

  union {
    SwitchLoweringStorage SwitchLowering;
    RetconLoweringStorage RetconLowering;
    AsyncLoweringStorage AsyncLowering;
  };

  CoroIdInst *getSwitchCoroId() const {
    assert(ABI == coro::ABI::Switch);
    return cast<CoroIdInst>(CoroBegin->getId());
  }

  AnyCoroIdRetconInst *getRetconCoroId() const {
    assert(ABI == coro::ABI::Retcon || ABI == coro::ABI::RetconOnce);
    return cast<AnyCoroIdRetconInst>(CoroBegin->getId());
  }

  CoroIdAsyncInst *getAsyncCoroId() const {
    assert(ABI == coro::ABI::Async);
    return cast<CoroIdAsyncInst>(CoroBegin->getId());
  }

  unsigned getSwitchIndexField() const {
    assert(ABI == coro::ABI::Switch);
    assert(FrameTy && "frame type not assigned");
    return SwitchLowering.IndexField;
  }
  IntegerType *getIndexType() const {
    assert(ABI == coro::ABI::Switch);
    assert(FrameTy && "frame type not assigned");
    return cast<IntegerType>(FrameTy->getElementType(getSwitchIndexField()));
  }
  ConstantInt *getIndex(uint64_t Value) const {
    return ConstantInt::get(getIndexType(), Value);
  }

  PointerType *getSwitchResumePointerType() const {
    assert(ABI == coro::ABI::Switch);
    assert(FrameTy && "frame type not assigned");
    return cast<PointerType>(FrameTy->getElementType(SwitchFieldIndex::Resume));
  }

  FunctionType *getResumeFunctionType() const {
    switch (ABI) {
    case coro::ABI::Switch:
      return FunctionType::get(Type::getVoidTy(FrameTy->getContext()),
                               PointerType::getUnqual(FrameTy->getContext()),
                               /*IsVarArg=*/false);
    case coro::ABI::Retcon:
    case coro::ABI::RetconOnce:
      return RetconLowering.ResumePrototype->getFunctionType();
    case coro::ABI::Async:
      // Not used. The function type depends on the active suspend.
      return nullptr;
    }

    llvm_unreachable("Unknown coro::ABI enum");
  }

  ArrayRef<Type *> getRetconResultTypes() const {
    assert(ABI == coro::ABI::Retcon || ABI == coro::ABI::RetconOnce);
    auto FTy = CoroBegin->getFunction()->getFunctionType();

    // The safety of all this is checked by checkWFRetconPrototype.
    if (auto STy = dyn_cast<StructType>(FTy->getReturnType())) {
      return STy->elements().slice(1);
    } else {
      return ArrayRef<Type *>();
    }
  }

  ArrayRef<Type *> getRetconResumeTypes() const {
    assert(ABI == coro::ABI::Retcon || ABI == coro::ABI::RetconOnce);

    // The safety of all this is checked by checkWFRetconPrototype.
    auto FTy = RetconLowering.ResumePrototype->getFunctionType();
    return FTy->params().slice(1);
  }

  CallingConv::ID getResumeFunctionCC() const {
    switch (ABI) {
    case coro::ABI::Switch:
      return CallingConv::Fast;

    case coro::ABI::Retcon:
    case coro::ABI::RetconOnce:
      return RetconLowering.ResumePrototype->getCallingConv();
    case coro::ABI::Async:
      return AsyncLowering.AsyncCC;
    }
    llvm_unreachable("Unknown coro::ABI enum");
  }

  AllocaInst *getPromiseAlloca() const {
    if (ABI == coro::ABI::Switch)
      return SwitchLowering.PromiseAlloca;
    return nullptr;
  }

  BasicBlock::iterator getInsertPtAfterFramePtr() const {
    if (auto *I = dyn_cast<Instruction>(FramePtr)) {
      BasicBlock::iterator It = std::next(I->getIterator());
      It.setHeadBit(true); // Copy pre-RemoveDIs behaviour.
      return It;
    }
    return cast<Argument>(FramePtr)->getParent()->getEntryBlock().begin();
  }

  /// Allocate memory according to the rules of the active lowering.
  ///
  /// \param CG - if non-null, will be updated for the new call
  Value *emitAlloc(IRBuilder<> &Builder, Value *Size, CallGraph *CG) const;

  /// Deallocate memory according to the rules of the active lowering.
  ///
  /// \param CG - if non-null, will be updated for the new call
  void emitDealloc(IRBuilder<> &Builder, Value *Ptr, CallGraph *CG) const;

  Shape() = default;
  explicit Shape(Function &F) {
    SmallVector<CoroFrameInst *, 8> CoroFrames;
    SmallVector<CoroSaveInst *, 2> UnusedCoroSaves;

    analyze(F, CoroFrames, UnusedCoroSaves);
    if (!CoroBegin) {
      invalidateCoroutine(F, CoroFrames);
      return;
    }
    cleanCoroutine(CoroFrames, UnusedCoroSaves);
  }
};

} // end namespace coro

} // end namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_COROSHAPE_H
