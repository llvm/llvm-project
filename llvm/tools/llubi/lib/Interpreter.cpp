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
#include "llvm/IR/InlineAsm.h"
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
  Function *ResolvedCallee = nullptr;
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
/// Unlike the Context class that manages the global state,
/// InstExecutor only maintains the state for call frames.
class InstExecutor : public InstVisitor<InstExecutor, void> {
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

  void reportError(StringRef Msg) {
    // Check if we have already reported an error message.
    if (!Status)
      return;
    Status = false;
    Handler.onError(Msg);
  }

  const AnyValue &getValue(Value *V) {
    if (auto *C = dyn_cast<Constant>(V))
      return Ctx.getConstantValue(C);
    return CurrentFrame->ValueMap.at(V);
  }

  void setResult(Instruction &I, AnyValue V) {
    if (Status)
      Status &= Handler.onInstructionExecuted(I, V);
    CurrentFrame->ValueMap.insert_or_assign(&I, std::move(V));
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

  void visitUnOp(Instruction &I,
                 function_ref<AnyValue(const AnyValue &)> ScalarFn) {
    setResult(I, computeUnOp(I.getType(), getValue(I.getOperand(0)), ScalarFn));
  }

  void visitIntUnOp(Instruction &I,
                    function_ref<AnyValue(const APInt &)> ScalarFn) {
    visitUnOp(I, [&](const AnyValue &Operand) -> AnyValue {
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

  void visitBinOp(
      Instruction &I,
      function_ref<AnyValue(const AnyValue &, const AnyValue &)> ScalarFn) {
    setResult(I, computeBinOp(I.getType(), getValue(I.getOperand(0)),
                              getValue(I.getOperand(1)), ScalarFn));
  }

  void
  visitIntBinOp(Instruction &I,
                function_ref<AnyValue(const APInt &, const APInt &)> ScalarFn) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
      if (LHS.isPoison() || RHS.isPoison())
        return AnyValue::poison();
      return ScalarFn(LHS.asInteger(), RHS.asInteger());
    });
  }

  void jumpTo(Instruction &Terminator, BasicBlock *DestBB) {
    if (!Handler.onBBJump(Terminator, *DestBB)) {
      Status = false;
      return;
    }
    BasicBlock *From = CurrentFrame->BB;
    CurrentFrame->BB = DestBB;
    CurrentFrame->PC = DestBB->begin();
    // Update PHI nodes in batch to avoid the interference between PHI nodes.
    // We need to store the incoming values into a temporary buffer.
    // Otherwise, the incoming value may be overwritten before it is
    // used by other PHI nodes.
    SmallVector<std::pair<PHINode *, AnyValue>> IncomingValues;
    PHINode *PHI = nullptr;
    while ((PHI = dyn_cast<PHINode>(CurrentFrame->PC))) {
      Value *Incoming = PHI->getIncomingValueForBlock(From);
      // TODO: handle fast-math flags.
      IncomingValues.emplace_back(PHI, getValue(Incoming));
      ++CurrentFrame->PC;
    }
    for (auto &[K, V] : IncomingValues)
      setResult(*K, std::move(V));
  }

  /// Helper function to determine whether an inline asm is a no-op, which is
  /// used to implement black_box style optimization blockers.
  bool isNoopInlineAsm(Value *V, Type *RetTy) {
    if (auto *Asm = dyn_cast<InlineAsm>(V))
      return Asm->getAsmString().empty() && RetTy->isVoidTy();
    return false;
  }

public:
  InstExecutor(Context &C, EventHandler &H, Function &F,
               ArrayRef<AnyValue> Args, AnyValue &RetVal)
      : Ctx(C), Handler(H), Status(true) {
    CallStack.emplace_back(F, /*CallSite=*/nullptr, /*LastFrame=*/nullptr, Args,
                           RetVal, Ctx.getTLIImpl());
  }

  void visitReturnInst(ReturnInst &RI) {
    if (auto *RV = RI.getReturnValue())
      CurrentFrame->RetVal = getValue(RV);
    CurrentFrame->State = FrameState::Exit;
    Status &= Handler.onInstructionExecuted(RI, None);
  }

  void visitBranchInst(BranchInst &BI) {
    if (BI.isConditional()) {
      switch (getValue(BI.getCondition()).asBoolean()) {
      case BooleanKind::True:
        jumpTo(BI, BI.getSuccessor(0));
        return;
      case BooleanKind::False:
        jumpTo(BI, BI.getSuccessor(1));
        return;
      case BooleanKind::Poison:
        reportImmediateUB("Branch on poison condition.");
        return;
      }
    }
    jumpTo(BI, BI.getSuccessor(0));
  }

  void visitSwitchInst(SwitchInst &SI) {
    auto &Cond = getValue(SI.getCondition());
    if (Cond.isPoison()) {
      reportImmediateUB("Switch on poison condition.");
      return;
    }
    for (auto &Case : SI.cases()) {
      if (Case.getCaseValue()->getValue() == Cond.asInteger()) {
        jumpTo(SI, Case.getCaseSuccessor());
        return;
      }
    }
    jumpTo(SI, SI.getDefaultDest());
  }

  void visitUnreachableInst(UnreachableInst &) {
    reportImmediateUB("Unreachable code.");
  }

  void visitCallBrInst(CallBrInst &CI) {
    if (isNoopInlineAsm(CI.getCalledOperand(), CI.getType())) {
      jumpTo(CI, CI.getDefaultDest());
      return;
    }

    Handler.onUnrecognizedInstruction(CI);
    Status = false;
  }

  void visitIndirectBrInst(IndirectBrInst &IBI) {
    auto &Target = getValue(IBI.getAddress());
    if (Target.isPoison()) {
      reportImmediateUB("Indirect branch on poison.");
      return;
    }
    if (BasicBlock *DestBB = Ctx.getTargetBlock(Target.asPointer())) {
      if (any_of(IBI.successors(),
                 [DestBB](BasicBlock *Succ) { return Succ == DestBB; }))
        jumpTo(IBI, DestBB);
      else
        reportImmediateUB("Indirect branch on unlisted target BB.");

      return;
    }
    reportImmediateUB("Indirect branch on invalid target BB.");
  }

  void returnFromCallee() {
    // TODO: handle retval attributes (Attributes from known callee should be
    // applied if available).
    // TODO: handle metadata
    auto &CB = cast<CallBase>(*CurrentFrame->PC);
    CurrentFrame->CalleeArgs.clear();
    AnyValue &RetVal = CurrentFrame->CalleeRetVal;
    setResult(CB, std::move(RetVal));

    if (auto *II = dyn_cast<InvokeInst>(&CB))
      jumpTo(*II, II->getNormalDest());
    else if (CurrentFrame->State == FrameState::Pending)
      ++CurrentFrame->PC;
  }

  AnyValue callIntrinsic(CallBase &CB) {
    Intrinsic::ID IID = CB.getIntrinsicID();
    switch (IID) {
    case Intrinsic::assume:
      switch (getValue(CB.getArgOperand(0)).asBoolean()) {
      case BooleanKind::True:
        break;
      case BooleanKind::False:
      case BooleanKind::Poison:
        reportImmediateUB("Assume on false or poison condition.");
        break;
      }
      // TODO: handle llvm.assume with operand bundles
      return AnyValue();
    default:
      Handler.onUnrecognizedInstruction(CB);
      Status = false;
      return AnyValue();
    }
  }

  AnyValue callLibFunc(CallBase &CB, Function *ResolvedCallee) {
    LibFunc LF;
    // Respect nobuiltin attributes on call site.
    if (CB.isNoBuiltin() ||
        !CurrentFrame->TLI.getLibFunc(*ResolvedCallee, LF)) {
      Handler.onUnrecognizedInstruction(CB);
      Status = false;
      return AnyValue();
    }

    Handler.onUnrecognizedInstruction(CB);
    Status = false;
    return AnyValue();
  }

  void enterCall(CallBase &CB) {
    Function *Callee = CB.getCalledFunction();
    // TODO: handle parameter attributes (Attributes from known callee should be
    // applied if available).
    // TODO: handle byval/initializes
    auto &CalleeArgs = CurrentFrame->CalleeArgs;
    assert(CalleeArgs.empty() &&
           "Forgot to call returnFromCallee before entering a new call.");
    for (Value *Arg : CB.args())
      CalleeArgs.push_back(getValue(Arg));

    if (!Callee) {
      Value *CalledOperand = CB.getCalledOperand();
      if (isNoopInlineAsm(CalledOperand, CB.getType())) {
        CurrentFrame->ResolvedCallee = nullptr;
        returnFromCallee();
        return;
      }

      if (isa<InlineAsm>(CalledOperand)) {
        Handler.onUnrecognizedInstruction(CB);
        Status = false;
        return;
      }

      auto &CalleeVal = getValue(CalledOperand);
      if (CalleeVal.isPoison()) {
        reportImmediateUB("Indirect call through poison function pointer.");
        return;
      }
      Callee = Ctx.getTargetFunction(CalleeVal.asPointer());
      if (!Callee) {
        reportImmediateUB("Indirect call through invalid function pointer.");
        return;
      }
      if (Callee->getFunctionType() != CB.getFunctionType()) {
        reportImmediateUB("Indirect call through a function pointer with "
                          "mismatched signature.");
        return;
      }
    }

    assert(Callee && "Expected a resolved callee function.");
    assert(
        Callee->getFunctionType() == CB.getFunctionType() &&
        "Expected the callee function type to match the call site signature.");
    CurrentFrame->ResolvedCallee = Callee;
    if (Callee->isIntrinsic()) {
      CurrentFrame->CalleeRetVal = callIntrinsic(CB);
      returnFromCallee();
      return;
    } else if (Callee->isDeclaration()) {
      CurrentFrame->CalleeRetVal = callLibFunc(CB, Callee);
      returnFromCallee();
      return;
    } else {
      uint32_t MaxStackDepth = Ctx.getMaxStackDepth();
      if (MaxStackDepth && CallStack.size() >= MaxStackDepth) {
        reportError("Maximum stack depth exceeded.");
        return;
      }
      assert(!Callee->empty() && "Expected a defined function.");
      // Suspend the current frame and push the callee frame onto the stack.
      ArrayRef<AnyValue> Args = CurrentFrame->CalleeArgs;
      AnyValue &RetVal = CurrentFrame->CalleeRetVal;
      CurrentFrame->State = FrameState::Pending;
      CallStack.emplace_back(*Callee, &CB, CurrentFrame, Args, RetVal,
                             Ctx.getTLIImpl());
    }
  }

  void visitCallInst(CallInst &CI) { enterCall(CI); }

  void visitInvokeInst(InvokeInst &II) {
    // TODO: handle exceptions
    enterCall(II);
  }

  void visitAdd(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return addNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  void visitSub(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return subNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  void visitMul(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) {
      return mulNoWrap(LHS, RHS, I.hasNoSignedWrap(), I.hasNoUnsignedWrap());
    });
  }

  void visitSDiv(BinaryOperator &I) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
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

  void visitSRem(BinaryOperator &I) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
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

  void visitUDiv(BinaryOperator &I) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
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

  void visitURem(BinaryOperator &I) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
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

  void visitTruncInst(TruncInst &Trunc) {
    visitIntUnOp(Trunc, [&](const APInt &Operand) -> AnyValue {
      unsigned DestBW = Trunc.getType()->getScalarSizeInBits();
      if (Trunc.hasNoSignedWrap() && Operand.getSignificantBits() > DestBW)
        return AnyValue::poison();
      if (Trunc.hasNoUnsignedWrap() && Operand.getActiveBits() > DestBW)
        return AnyValue::poison();
      return Operand.trunc(DestBW);
    });
  }

  void visitZExtInst(ZExtInst &ZExt) {
    visitIntUnOp(ZExt, [&](const APInt &Operand) -> AnyValue {
      uint32_t DestBW = ZExt.getDestTy()->getScalarSizeInBits();
      if (ZExt.hasNonNeg() && Operand.isNegative())
        return AnyValue::poison();
      return Operand.zext(DestBW);
    });
  }

  void visitSExtInst(SExtInst &SExt) {
    visitIntUnOp(SExt, [&](const APInt &Operand) -> AnyValue {
      uint32_t DestBW = SExt.getDestTy()->getScalarSizeInBits();
      return Operand.sext(DestBW);
    });
  }

  void visitAnd(BinaryOperator &I) {
    visitIntBinOp(I, [](const APInt &LHS, const APInt &RHS) -> AnyValue {
      return LHS & RHS;
    });
  }

  void visitXor(BinaryOperator &I) {
    visitIntBinOp(I, [](const APInt &LHS, const APInt &RHS) -> AnyValue {
      return LHS ^ RHS;
    });
  }

  void visitOr(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
      if (cast<PossiblyDisjointInst>(I).isDisjoint() && LHS.intersects(RHS))
        return AnyValue::poison();
      return LHS | RHS;
    });
  }

  void visitShl(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
      if (RHS.uge(LHS.getBitWidth()))
        return AnyValue::poison();
      if (I.hasNoSignedWrap() && RHS.uge(LHS.getNumSignBits()))
        return AnyValue::poison();
      if (I.hasNoUnsignedWrap() && RHS.ugt(LHS.countl_zero()))
        return AnyValue::poison();
      return LHS.shl(RHS);
    });
  }

  void visitLShr(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
      if (RHS.uge(cast<PossiblyExactOperator>(I).isExact()
                      ? LHS.countr_zero() + 1
                      : LHS.getBitWidth()))
        return AnyValue::poison();
      return LHS.lshr(RHS);
    });
  }

  void visitAShr(BinaryOperator &I) {
    visitIntBinOp(I, [&](const APInt &LHS, const APInt &RHS) -> AnyValue {
      if (RHS.uge(cast<PossiblyExactOperator>(I).isExact()
                      ? LHS.countr_zero() + 1
                      : LHS.getBitWidth()))
        return AnyValue::poison();
      return LHS.ashr(RHS);
    });
  }

  void visitICmpInst(ICmpInst &I) {
    visitBinOp(I, [&](const AnyValue &LHS, const AnyValue &RHS) -> AnyValue {
      if (LHS.isPoison() || RHS.isPoison())
        return AnyValue::poison();
      // TODO: handle pointer comparison.
      const APInt &LHSVal = LHS.asInteger();
      const APInt &RHSVal = RHS.asInteger();
      if (I.hasSameSign() && LHSVal.isNonNegative() != RHSVal.isNonNegative())
        return AnyValue::poison();
      return AnyValue::boolean(
          ICmpInst::compare(LHSVal, RHSVal, I.getPredicate()));
    });
  }

  void visitSelect(SelectInst &SI) {
    // TODO: handle fast-math flags.
    if (SI.getCondition()->getType()->isIntegerTy(1)) {
      switch (getValue(SI.getCondition()).asBoolean()) {
      case BooleanKind::True:
        setResult(SI, getValue(SI.getTrueValue()));
        return;
      case BooleanKind::False:
        setResult(SI, getValue(SI.getFalseValue()));
        return;
      case BooleanKind::Poison:
        setResult(SI, AnyValue::getPoisonValue(Ctx, SI.getType()));
        return;
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
    setResult(SI, std::move(Res));
  }

  void visitInstruction(Instruction &I) {
    Handler.onUnrecognizedInstruction(I);
    Status = false;
  }

  void visitExtractValueInst(ExtractValueInst &EVI) {
    auto &Res = getValue(EVI.getAggregateOperand());
    const AnyValue *Pos = &Res;
    for (unsigned Idx : EVI.indices())
      Pos = &Pos->asAggregate()[Idx];
    setResult(EVI, *Pos);
  }

  void visitInsertValueInst(InsertValueInst &IVI) {
    AnyValue Res = getValue(IVI.getAggregateOperand());
    AnyValue *Pos = &Res;
    for (unsigned Idx : IVI.indices())
      Pos = &Pos->asAggregate()[Idx];
    *Pos = getValue(IVI.getInsertedValueOperand());
    setResult(IVI, std::move(Res));
  }

  void visitInsertElementInst(InsertElementInst &IEI) {
    auto Res = getValue(IEI.getOperand(0));
    auto &ResVec = Res.asAggregate();
    auto &Idx = getValue(IEI.getOperand(2));
    if (Idx.isPoison() || Idx.asInteger().uge(ResVec.size())) {
      setResult(IEI, AnyValue::getPoisonValue(Ctx, IEI.getType()));
      return;
    }
    ResVec[Idx.asInteger().getZExtValue()] = getValue(IEI.getOperand(1));
    setResult(IEI, std::move(Res));
  }

  void visitExtractElementInst(ExtractElementInst &EEI) {
    auto &SrcVec = getValue(EEI.getOperand(0)).asAggregate();
    auto &Idx = getValue(EEI.getOperand(1));
    if (Idx.isPoison() || Idx.asInteger().uge(SrcVec.size())) {
      setResult(EEI, AnyValue::getPoisonValue(Ctx, EEI.getType()));
      return;
    }
    setResult(EEI, SrcVec[Idx.asInteger().getZExtValue()]);
  }

  void visitShuffleVectorInst(ShuffleVectorInst &SVI) {
    auto &LHSVec = getValue(SVI.getOperand(0)).asAggregate();
    auto &RHSVec = getValue(SVI.getOperand(1)).asAggregate();
    uint32_t Size = cast<VectorType>(SVI.getOperand(0)->getType())
                        ->getElementCount()
                        .getKnownMinValue();
    std::vector<AnyValue> Res;
    uint32_t DstLen = Ctx.getEVL(SVI.getType()->getElementCount());
    Res.reserve(DstLen);
    uint32_t Stride = SVI.getShuffleMask().size();
    // For scalable vectors, we need to repeat the shuffle mask until we fill
    // the destination vector.
    for (uint32_t Off = 0; Off != DstLen; Off += Stride) {
      for (int Idx : SVI.getShuffleMask()) {
        if (Idx == PoisonMaskElem)
          Res.push_back(AnyValue::poison());
        else if (Idx < static_cast<int>(Size))
          Res.push_back(LHSVec[Idx]);
        else
          Res.push_back(RHSVec[Idx - Size]);
      }
    }
    setResult(SVI, std::move(Res));
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
      } else {
        assert(Top.State == FrameState::Pending &&
               "Expected to return from a callee.");
        returnFromCallee();
      }

      Top.State = FrameState::Running;
      // Interpreter loop inside a function
      while (Status) {
        assert(Top.State == FrameState::Running &&
               "Expected to be in running state.");
        if (MaxSteps != 0 && Steps >= MaxSteps) {
          reportError("Exceeded maximum number of execution steps.");
          break;
        }
        ++Steps;

        Instruction &I = *Top.PC;
        visit(&I);
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
