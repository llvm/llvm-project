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
#include "llvm/IR/Operator.h"
#include "llvm/Support/Allocator.h"

namespace llvm::ubi {

enum class FrameState {
  // It is about to enter the function.
  // Valid transition:
  //   -> Running
  Entry,
  // It is executing instructions inside the function.
  // Valid transitions:
  //   -> Pending (on call)
  //   -> Exit (on return)
  Running,
  // It is about to enter a callee or handle return value from the callee.
  // Valid transitions:
  //   -> Running (after returning from callee)
  Pending,
  // It is about to return the control to the caller.
  Exit,
};

/// Context for a function call.
/// This struct maintains the state during the execution of a function,
/// including the control flow, values of executed instructions, and stack
/// objects.
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
  // Stack objects allocated in this frame. They will be automatically freed
  // when the function returns.
  SmallVector<IntrusiveRefCntPtr<MemoryObject>> Allocas;
  // Values of arguments and executed instructions in this function.
  DenseMap<Value *, AnyValue> ValueMap;

  // Reserved for in-flight subroutines.
  SmallVector<AnyValue> CalleeArgs;
  AnyValue CalleeRetVal;

  Frame(Function &F, CallBase *CallSite, Frame *LastFrame,
        ArrayRef<AnyValue> Args, AnyValue &RetVal,
        const TargetLibraryInfoImpl &TLIImpl)
      : Func(F), LastFrame(LastFrame), CallSite(CallSite), Args(Args),
        RetVal(RetVal), TLI(TLIImpl, &F) {
    assert((Args.size() == F.arg_size() ||
            (F.isVarArg() && Args.size() >= F.arg_size())) &&
           "Expected enough arguments to call the function.");
    BB = &Func.getEntryBlock();
    PC = BB->begin();
    for (Argument &Arg : F.args())
      ValueMap[&Arg] = Args[Arg.getArgNo()];
  }
};

static AnyValue addNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  APInt Res = LHS + RHS;
  if (HasNUW && Res.ult(RHS))
    return AnyValue::poison();
  if (HasNSW && LHS.isNonNegative() == RHS.isNonNegative() &&
      LHS.isNonNegative() != Res.isNonNegative())
    return AnyValue::poison();
  return Res;
}

static AnyValue subNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  APInt Res = LHS - RHS;
  if (HasNUW && Res.ugt(LHS))
    return AnyValue::poison();
  if (HasNSW && LHS.isNonNegative() != RHS.isNonNegative() &&
      LHS.isNonNegative() != Res.isNonNegative())
    return AnyValue::poison();
  return Res;
}

static AnyValue mulNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  bool Overflow = false;
  APInt Res = LHS.smul_ov(RHS, Overflow);
  if (HasNSW && Overflow)
    return AnyValue::poison();
  if (HasNUW) {
    (void)LHS.umul_ov(RHS, Overflow);
    if (Overflow)
      return AnyValue::poison();
  }
  return Res;
}

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
    if (!Status)
      return;
    Status = false;
    // TODO: Provide stack trace information.
    Handler.onImmediateUB(Msg);
  }

  const AnyValue &getValue(Value *V) {
    if (auto *C = dyn_cast<Constant>(V))
      return Ctx.getConstantValue(C);
    return CurrentFrame->ValueMap.at(V);
  }

  bool setResult(Instruction &I, AnyValue V) {
    if (Status)
      Handler.onInstructionExecuted(I, V);
    CurrentFrame->ValueMap.insert_or_assign(&I, std::move(V));
    return true;
  }

  AnyValue computeUnOp(Type *Ty, const AnyValue &Operand,
                       function_ref<AnyValue(const AnyValue &)> ScalarFn) {
    if (Ty->isVectorTy()) {
      auto &OperandVec = Operand.asAggregate();
      std::vector<AnyValue> ResVec;
      ResVec.reserve(OperandVec.size());
      for (const auto &Scalar : OperandVec)
        ResVec.push_back(ScalarFn(Scalar));
      return std::move(ResVec);
    }
    return ScalarFn(Operand);
  }

  bool visitUnOp(Instruction &I,
                 function_ref<AnyValue(const AnyValue &)> ScalarFn) {
    return setResult(
        I, computeUnOp(I.getType(), getValue(I.getOperand(0)), ScalarFn));
  }

  bool visitIntUnOp(Instruction &I,
                    function_ref<AnyValue(const APInt &)> ScalarFn) {
    return visitUnOp(I, [&](const AnyValue &Operand) -> AnyValue {
      if (Operand.isPoison())
        return AnyValue::poison();
      return ScalarFn(Operand.asInteger());
    });
  }

  AnyValue computeBinOp(
      Type *Ty, const AnyValue &LHS, const AnyValue &RHS,
      function_ref<AnyValue(const AnyValue &, const AnyValue &)> ScalarFn) {
    if (Ty->isVectorTy()) {
      auto &LHSVec = LHS.asAggregate();
      auto &RHSVec = RHS.asAggregate();
      std::vector<AnyValue> ResVec;
      ResVec.reserve(LHSVec.size());
      for (const auto &[ScalarLHS, ScalarRHS] : zip(LHSVec, RHSVec))
        ResVec.push_back(ScalarFn(ScalarLHS, ScalarRHS));
      return std::move(ResVec);
    }
    return ScalarFn(LHS, RHS);
  }

  bool visitBinOp(
      Instruction &I,
      function_ref<AnyValue(const AnyValue &, const AnyValue &)> ScalarFn) {
    return setResult(I, computeBinOp(I.getType(), getValue(I.getOperand(0)),
                                     getValue(I.getOperand(1)), ScalarFn));
  }

  bool
  visitIntBinOp(Instruction &I,
                function_ref<AnyValue(const APInt &, const APInt &)> ScalarFn) {
    return visitBinOp(
        I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
          if (LHS.isPoison() || RHS.isPoison())
            return AnyValue::poison();
          return ScalarFn(LHS.asInteger(), RHS.asInteger());
        });
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

  bool visitAdd(BinaryOperator &I) {
    return visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return addNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  bool visitSub(BinaryOperator &I) {
    return visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return subNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  bool visitMul(BinaryOperator &I) {
    return visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return mulNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  bool visitSDiv(BinaryOperator &I) {
    return visitBinOp(
        I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
          // Priority: Immediate UB > poison > normal value
          if (RHS.isPoison()) {
            reportImmediateUB("Division by zero (refine RHS to 0).");
            return AnyValue::poison();
          }
          const APInt &RHSVal = RHS.asInteger();
          if (RHSVal.isZero()) {
            reportImmediateUB("Division by zero.");
            return AnyValue::poison();
          }
          if (LHS.isPoison()) {
            if (RHSVal.isAllOnes())
              reportImmediateUB(
                  "Signed division overflow (refine LHS to INT_MIN).");
            return AnyValue::poison();
          }
          const APInt &LHSVal = LHS.asInteger();
          if (LHSVal.isMinSignedValue() && RHSVal.isAllOnes()) {
            reportImmediateUB("Signed division overflow.");
            return AnyValue::poison();
          }

          if (I.isExact()) {
            APInt Q, R;
            APInt::sdivrem(LHSVal, RHSVal, Q, R);
            if (!R.isZero())
              return AnyValue::poison();
            return Q;
          } else {
            return LHSVal.sdiv(RHSVal);
          }
        });
  }

  bool visitSRem(BinaryOperator &I) {
    return visitBinOp(
        I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
          // Priority: Immediate UB > poison > normal value
          if (RHS.isPoison()) {
            reportImmediateUB("Division by zero (refine RHS to 0).");
            return AnyValue::poison();
          }
          const APInt &RHSVal = RHS.asInteger();
          if (RHSVal.isZero()) {
            reportImmediateUB("Division by zero.");
            return AnyValue::poison();
          }
          if (LHS.isPoison()) {
            if (RHSVal.isAllOnes())
              reportImmediateUB(
                  "Signed division overflow (refine LHS to INT_MIN).");
            return AnyValue::poison();
          }
          const APInt &LHSVal = LHS.asInteger();
          if (LHSVal.isMinSignedValue() && RHSVal.isAllOnes()) {
            reportImmediateUB("Signed division overflow.");
            return AnyValue::poison();
          }

          return LHSVal.srem(RHSVal);
        });
  }

  bool visitUDiv(BinaryOperator &I) {
    return visitBinOp(
        I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
          // Priority: Immediate UB > poison > normal value
          if (RHS.isPoison()) {
            reportImmediateUB("Division by zero (refine RHS to 0).");
            return AnyValue::poison();
          }
          const APInt &RHSVal = RHS.asInteger();
          if (RHSVal.isZero()) {
            reportImmediateUB("Division by zero.");
            return AnyValue::poison();
          }
          if (LHS.isPoison())
            return AnyValue::poison();
          const APInt &LHSVal = LHS.asInteger();

          if (I.isExact()) {
            APInt Q, R;
            APInt::udivrem(LHSVal, RHSVal, Q, R);
            if (!R.isZero())
              return AnyValue::poison();
            return Q;
          } else {
            return LHSVal.udiv(RHSVal);
          }
        });
  }

  bool visitURem(BinaryOperator &I) {
    return visitBinOp(
        I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
          // Priority: Immediate UB > poison > normal value
          if (RHS.isPoison()) {
            reportImmediateUB("Division by zero (refine RHS to 0).");
            return AnyValue::poison();
          }
          const APInt &RHSVal = RHS.asInteger();
          if (RHSVal.isZero()) {
            reportImmediateUB("Division by zero.");
            return AnyValue::poison();
          }
          if (LHS.isPoison())
            return AnyValue::poison();
          const APInt &LHSVal = LHS.asInteger();
          return LHSVal.urem(RHSVal);
        });
  }

  bool visitTruncInst(TruncInst &Trunc) {
    return visitIntUnOp(Trunc, [&](const APInt &Operand) -> AnyValue {
      unsigned DestBW = Trunc.getType()->getScalarSizeInBits();
      if (Trunc.hasNoSignedWrap() && Operand.getSignificantBits() > DestBW)
        return AnyValue::poison();
      if (Trunc.hasNoUnsignedWrap() && Operand.getActiveBits() > DestBW)
        return AnyValue::poison();
      return Operand.trunc(DestBW);
    });
  }

  bool visitZExtInst(ZExtInst &ZExt) {
    return visitIntUnOp(ZExt, [&](const APInt &Operand) -> AnyValue {
      uint32_t DestBW = ZExt.getDestTy()->getScalarSizeInBits();
      if (ZExt.hasNonNeg() && Operand.isNegative())
        return AnyValue::poison();
      return Operand.zext(DestBW);
    });
  }

  bool visitSExtInst(SExtInst &SExt) {
    return visitIntUnOp(SExt, [&](const APInt &Operand) -> AnyValue {
      uint32_t DestBW = SExt.getDestTy()->getScalarSizeInBits();
      return Operand.sext(DestBW);
    });
  }

  bool visitAnd(BinaryOperator &I) {
    return visitIntBinOp(I, [](const APInt &LHS, const APInt &RHS) -> AnyValue {
      return LHS & RHS;
    });
  }

  bool visitXor(BinaryOperator &I) {
    return visitIntBinOp(I, [](const APInt &LHS, const APInt &RHS) -> AnyValue {
      return LHS ^ RHS;
    });
  }

  bool visitOr(BinaryOperator &I) {
    return visitIntBinOp(
        I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
          if (cast<PossiblyDisjointInst>(I).isDisjoint() && LHS.intersects(RHS))
            return AnyValue::poison();
          return LHS | RHS;
        });
  }

  bool visitShl(BinaryOperator &I) {
    return visitIntBinOp(
        I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
          if (RHS.uge(LHS.getBitWidth()))
            return AnyValue::poison();
          if (I.hasNoSignedWrap() && RHS.uge(LHS.getNumSignBits()))
            return AnyValue::poison();
          if (I.hasNoUnsignedWrap() && RHS.ugt(LHS.countl_zero()))
            return AnyValue::poison();
          return LHS.shl(RHS);
        });
  }

  bool visitLShr(BinaryOperator &I) {
    return visitIntBinOp(I,
                         [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
                           if (RHS.uge(cast<PossiblyExactOperator>(I).isExact()
                                           ? LHS.countr_zero() + 1
                                           : LHS.getBitWidth()))
                             return AnyValue::poison();
                           return LHS.lshr(RHS);
                         });
  }

  bool visitAShr(BinaryOperator &I) {
    return visitIntBinOp(I,
                         [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
                           if (RHS.uge(cast<PossiblyExactOperator>(I).isExact()
                                           ? LHS.countr_zero() + 1
                                           : LHS.getBitWidth()))
                             return AnyValue::poison();
                           return LHS.ashr(RHS);
                         });
  }

  bool visitSelect(SelectInst &SI) {
    // TODO: handle fast-math flags.
    if (SI.getCondition()->getType()->isIntegerTy(1)) {
      switch (getValue(SI.getCondition()).asBoolean()) {
      case BooleanKind::True:
        return setResult(SI, getValue(SI.getTrueValue()));
      case BooleanKind::False:
        return setResult(SI, getValue(SI.getFalseValue()));
      case BooleanKind::Poison:
        return setResult(SI, AnyValue::getPoisonValue(Ctx, SI.getType()));
      }
    }

    auto &Cond = getValue(SI.getCondition()).asAggregate();
    auto &TV = getValue(SI.getTrueValue()).asAggregate();
    auto &FV = getValue(SI.getFalseValue()).asAggregate();
    std::vector<AnyValue> Res;
    size_t Len = Cond.size();
    Res.reserve(Len);
    for (uint32_t I = 0; I != Len; ++I) {
      switch (Cond[I].asBoolean()) {
      case BooleanKind::True:
        Res.push_back(TV[I]);
        break;
      case BooleanKind::False:
        Res.push_back(FV[I]);
        break;
      case BooleanKind::Poison:
        Res.push_back(AnyValue::poison());
        break;
      }
    }
    return setResult(SI, std::move(Res));
  }

  bool visitInstruction(Instruction &I) {
    Handler.onUnrecognizedInstruction(I);
    return false;
  }

  /// This function implements the main interpreter loop.
  /// It handles function calls in a non-recursive manner to avoid stack
  /// overflows.
  bool runMainLoop() {
    uint32_t MaxSteps = Ctx.getMaxSteps();
    uint32_t Steps = 0;
    while (Status && !CallStack.empty()) {
      Frame &Top = CallStack.back();
      CurrentFrame = &Top;
      if (Top.State == FrameState::Entry) {
        Handler.onFunctionEntry(Top.Func, Top.Args, Top.CallSite);
        // TODO: Handle arg attributes
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

        // A function call or return has occurred.
        // We need to exit the inner loop and switch to a different frame.
        if (Top.State != FrameState::Running)
          break;

        // Otherwise, move to the next instruction if it is not a terminator.
        // For terminators, the PC is updated in the visit* method.
        if (!I.isTerminator())
          ++Top.PC;
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

bool Context::runFunction(Function &F, ArrayRef<AnyValue> Args,
                          AnyValue &RetVal, EventHandler &Handler) {
  InstExecutor Executor(*this, Handler, F, Args, RetVal);
  return Executor.runMainLoop();
}

} // namespace llvm::ubi
