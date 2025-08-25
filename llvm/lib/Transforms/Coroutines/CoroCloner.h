//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper class for splitting a coroutine into separate functions. For example
// the returned-continuation coroutine is split into separate continuation
// functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_COROCLONER_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_COROCLONER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

namespace coro {

enum class CloneKind {
  /// The shared resume function for a switch lowering.
  SwitchResume,

  /// The shared unwind function for a switch lowering.
  SwitchUnwind,

  /// The shared cleanup function for a switch lowering.
  SwitchCleanup,

  /// An individual continuation function.
  Continuation,

  /// An async resume function.
  Async,
};

class BaseCloner {
protected:
  Function &OrigF;
  const Twine &Suffix;
  coro::Shape &Shape;
  CloneKind FKind;
  IRBuilder<> Builder;
  TargetTransformInfo &TTI;

  ValueToValueMapTy VMap;
  Function *NewF = nullptr;
  Value *NewFramePtr = nullptr;

  /// The active suspend instruction; meaningful only for continuation and async
  /// ABIs.
  AnyCoroSuspendInst *ActiveSuspend = nullptr;

  /// Create a cloner for a continuation lowering.
  BaseCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             Function *NewF, AnyCoroSuspendInst *ActiveSuspend,
             TargetTransformInfo &TTI)
      : OrigF(OrigF), Suffix(Suffix), Shape(Shape),
        FKind(Shape.ABI == ABI::Async ? CloneKind::Async
                                      : CloneKind::Continuation),
        Builder(OrigF.getContext()), TTI(TTI), NewF(NewF),
        ActiveSuspend(ActiveSuspend) {
    assert(Shape.ABI == ABI::Retcon || Shape.ABI == ABI::RetconOnce ||
           Shape.ABI == ABI::Async);
    assert(NewF && "need existing function for continuation");
    assert(ActiveSuspend && "need active suspend point for continuation");
  }

public:
  BaseCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             CloneKind FKind, TargetTransformInfo &TTI)
      : OrigF(OrigF), Suffix(Suffix), Shape(Shape), FKind(FKind),
        Builder(OrigF.getContext()), TTI(TTI) {}

  virtual ~BaseCloner() {}

  /// Create a clone for a continuation lowering.
  static Function *createClone(Function &OrigF, const Twine &Suffix,
                               coro::Shape &Shape, Function *NewF,
                               AnyCoroSuspendInst *ActiveSuspend,
                               TargetTransformInfo &TTI) {
    assert(Shape.ABI == ABI::Retcon || Shape.ABI == ABI::RetconOnce ||
           Shape.ABI == ABI::Async);
    TimeTraceScope FunctionScope("BaseCloner");

    BaseCloner Cloner(OrigF, Suffix, Shape, NewF, ActiveSuspend, TTI);
    Cloner.create();
    return Cloner.getFunction();
  }

  Function *getFunction() const {
    assert(NewF != nullptr && "declaration not yet set");
    return NewF;
  }

  virtual void create();

protected:
  bool isSwitchDestroyFunction() {
    switch (FKind) {
    case CloneKind::Async:
    case CloneKind::Continuation:
    case CloneKind::SwitchResume:
      return false;
    case CloneKind::SwitchUnwind:
    case CloneKind::SwitchCleanup:
      return true;
    }
    llvm_unreachable("Unknown ClonerKind enum");
  }

  void replaceEntryBlock();
  Value *deriveNewFramePointer();
  void replaceRetconOrAsyncSuspendUses();
  void replaceCoroSuspends();
  void replaceCoroEnds();
  void replaceCoroIsInRamp();
  void replaceSwiftErrorOps();
  void salvageDebugInfo();
  void handleFinalSuspend();
};

class SwitchCloner : public BaseCloner {
protected:
  /// Create a cloner for a switch lowering.
  SwitchCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
               CloneKind FKind, TargetTransformInfo &TTI)
      : BaseCloner(OrigF, Suffix, Shape, FKind, TTI) {}

  void create() override;

public:
  /// Create a clone for a switch lowering.
  static Function *createClone(Function &OrigF, const Twine &Suffix,
                               coro::Shape &Shape, CloneKind FKind,
                               TargetTransformInfo &TTI) {
    assert(Shape.ABI == ABI::Switch);
    TimeTraceScope FunctionScope("SwitchCloner");

    SwitchCloner Cloner(OrigF, Suffix, Shape, FKind, TTI);
    Cloner.create();
    return Cloner.getFunction();
  }
};

} // end namespace coro

} // end namespace llvm

#endif // LLVM_LIB_TRANSFORMS_COROUTINES_COROCLONER_H
