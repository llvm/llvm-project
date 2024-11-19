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

#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

class CoroCloner {
public:
  enum class Kind {
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

protected:
  Function &OrigF;
  const Twine &Suffix;
  coro::Shape &Shape;
  Kind FKind;
  IRBuilder<> Builder;
  TargetTransformInfo &TTI;

  ValueToValueMapTy VMap;
  Function *NewF = nullptr;
  Value *NewFramePtr = nullptr;

  /// The active suspend instruction; meaningful only for continuation and async
  /// ABIs.
  AnyCoroSuspendInst *ActiveSuspend = nullptr;

  /// Create a cloner for a continuation lowering.
  CoroCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             Function *NewF, AnyCoroSuspendInst *ActiveSuspend,
             TargetTransformInfo &TTI)
      : OrigF(OrigF), Suffix(Suffix), Shape(Shape),
        FKind(Shape.ABI == coro::ABI::Async ? Kind::Async : Kind::Continuation),
        Builder(OrigF.getContext()), TTI(TTI), NewF(NewF),
        ActiveSuspend(ActiveSuspend) {
    assert(Shape.ABI == coro::ABI::Retcon ||
           Shape.ABI == coro::ABI::RetconOnce || Shape.ABI == coro::ABI::Async);
    assert(NewF && "need existing function for continuation");
    assert(ActiveSuspend && "need active suspend point for continuation");
  }

public:
  CoroCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             Kind FKind, TargetTransformInfo &TTI)
      : OrigF(OrigF), Suffix(Suffix), Shape(Shape), FKind(FKind),
        Builder(OrigF.getContext()), TTI(TTI) {}

  virtual ~CoroCloner() {}

  /// Create a clone for a continuation lowering.
  static Function *createClone(Function &OrigF, const Twine &Suffix,
                               coro::Shape &Shape, Function *NewF,
                               AnyCoroSuspendInst *ActiveSuspend,
                               TargetTransformInfo &TTI) {
    assert(Shape.ABI == coro::ABI::Retcon ||
           Shape.ABI == coro::ABI::RetconOnce || Shape.ABI == coro::ABI::Async);
    TimeTraceScope FunctionScope("CoroCloner");

    CoroCloner Cloner(OrigF, Suffix, Shape, NewF, ActiveSuspend, TTI);
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
    case Kind::Async:
    case Kind::Continuation:
    case Kind::SwitchResume:
      return false;
    case Kind::SwitchUnwind:
    case Kind::SwitchCleanup:
      return true;
    }
    llvm_unreachable("Unknown CoroCloner::Kind enum");
  }

  void replaceEntryBlock();
  Value *deriveNewFramePointer();
  void replaceRetconOrAsyncSuspendUses();
  void replaceCoroSuspends();
  void replaceCoroEnds();
  void replaceSwiftErrorOps();
  void salvageDebugInfo();
  void handleFinalSuspend();
};

class CoroSwitchCloner : public CoroCloner {
protected:
  /// Create a cloner for a switch lowering.
  CoroSwitchCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
                   Kind FKind, TargetTransformInfo &TTI)
      : CoroCloner(OrigF, Suffix, Shape, FKind, TTI) {}

  void create() override;

public:
  /// Create a clone for a switch lowering.
  static Function *createClone(Function &OrigF, const Twine &Suffix,
                               coro::Shape &Shape, Kind FKind,
                               TargetTransformInfo &TTI) {
    assert(Shape.ABI == coro::ABI::Switch);
    TimeTraceScope FunctionScope("CoroCloner");

    CoroSwitchCloner Cloner(OrigF, Suffix, Shape, FKind, TTI);
    Cloner.create();
    return Cloner.getFunction();
  }
};
