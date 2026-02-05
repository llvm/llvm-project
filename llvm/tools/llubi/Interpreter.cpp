//===- Interpreter.cpp - Interpreter Loop for llubi -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the evaluation loop for each kind of instruction.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Value.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Allocator.h"

namespace llvm::ubi {

enum class FrameState {
  Entry,
  Running,
  Pending,
  Exit,
};

struct Frame {
  Function &Func;
  Frame *LastFrame;
  CallBase *CallSite;
  ArrayRef<AnyValue> Args;
  AnyValue &RetVal;

  TargetLibraryInfo TLI;
  BasicBlock *BB;
  BasicBlock::iterator PC;
  FrameState State = FrameState::Entry;
  SmallVector<MemoryObject *> Allocas;
  DenseMap<Instruction *, AnyValue> ValueMap;

  // Reserved for in-flight subroutines.
  SmallVector<AnyValue> CalleeArgs;
  AnyValue CalleeRetVal;

  Frame(Function &F, CallBase *CallSite, Frame *LastFrame,
        ArrayRef<AnyValue> Args, AnyValue &RetVal,
        const TargetLibraryInfoImpl &TLIImpl)
      : Func(F), LastFrame(LastFrame), CallSite(CallSite), Args(Args),
        RetVal(RetVal), TLI(TLIImpl, &F) {
    BB = &Func.getEntryBlock();
    PC = BB->begin();
  }
};

/// Instruction executor using the visitor pattern.
/// visit* methods return true on success, false on error.
/// Unlike the Context class that manages the global state,
/// InstExecutor only maintains the state for call frames.
class InstExecutor : public InstVisitor<InstExecutor, bool> {
  Context &Ctx;
  EventHandler &Handler;
  std::list<Frame> CallStack;
  // Used to indicate whether the interpreter should continue execution.
  bool Status;
  Frame *CurrentFrame = nullptr;
  AnyValue None;

  void reportImmediateUB(StringRef Msg) {
    // Check if we have already reported an immediate UB.
    if (!Status) {
      return;
    }
    Status = false;
    // TODO: Provide stack trace information.
    Handler.onImmediateUB(Msg);
  }

  const AnyValue &getValue(Value *V) {
    if (auto *C = dyn_cast<Constant>(V))
      return Ctx.getConstantValue(C);
    return CurrentFrame->ValueMap.at(cast<Instruction>(V));
  }

public:
  InstExecutor(Context &C, EventHandler &H, Function &F,
               ArrayRef<AnyValue> Args, AnyValue &RetVal)
      : Ctx(C), Handler(H), Status(true) {
    CallStack.emplace_back(F, /*CallSite=*/nullptr, /*LastFrame=*/nullptr, Args,
                           RetVal, Ctx.getTLIImpl());
  }
  bool visitReturnInst(ReturnInst &RI) {
    if (auto *RV = RI.getReturnValue())
      CurrentFrame->RetVal = getValue(RV);
    CurrentFrame->State = FrameState::Exit;
    return Handler.onInstructionExecuted(RI, None);
  }
  bool visitInstruction(Instruction &I) {
    Handler.onUnrecognizedInstruction(I);
    return false;
  }

  bool runMainLoop() {
    uint32_t MaxSteps = Ctx.getMaxSteps();
    uint32_t Steps = 0;
    while (Status && !CallStack.empty()) {
      Frame &Top = CallStack.back();
      CurrentFrame = &Top;
      if (Top.State == FrameState::Entry) {
        Handler.onFunctionEntry(Top.Func, Top.Args, Top.CallSite);
        // TODO:Handle arg attributes
      } else {
        assert(Top.State == FrameState::Pending &&
               "Expected to return from a callee.");
      }

      Top.State = FrameState::Running;
      // Interpreter loop inside a function
      while (Status) {
        assert(Top.State == FrameState::Running &&
               "Expected to be in running state.");
        if (MaxSteps != 0 && Steps >= MaxSteps) {
          reportImmediateUB("Exceeded maximum number of execution steps.");
          break;
        }
        ++Steps;

        Instruction &I = *Top.PC;
        if (!visit(&I)) {
          Status = false;
          break;
        }
        if (!Status)
          break;

        if (Top.State != FrameState::Pending && !I.isTerminator()) {
          if (I.getType()->isVoidTy())
            Handler.onInstructionExecuted(I, None);
          else
            Handler.onInstructionExecuted(I, Top.ValueMap.at(&I));
        }

        if (Top.State != FrameState::Running) {
          // A function call or return has occurred.
          break;
        } else if (!I.isTerminator()) {
          ++Top.PC;
        }
      }

      if (!Status)
        break;

      if (Top.State == FrameState::Exit) {
        assert((Top.Func.getReturnType()->isVoidTy() || !Top.RetVal.isNone()) &&
               "Expected return value to be set on function exit.");
        // TODO:Handle retval attributes
        Handler.onFunctionExit(Top.Func, Top.RetVal);
        CallStack.pop_back();
      } else {
        assert(Top.State == FrameState::Pending &&
               "Expected to enter a callee.");
      }
    }
    return Status;
  }
};

/// This function implements the main interpreter loop.
/// It handles function calls in a non-recursive manner to avoid stack
/// overflows.
bool Context::runFunction(Function &F, ArrayRef<AnyValue> Args,
                          AnyValue &RetVal, EventHandler &Handler) {
  InstExecutor Executor(*this, Handler, F, Args, RetVal);
  return Executor.runMainLoop();
}

} // namespace llvm::ubi
