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
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"

namespace llvm::ubi {

using namespace PatternMatch;

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
  const DataLayout &DL;
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

  /// Check if the upcoming memory access is valid. Returns the offset relative
  /// to the underlying object if it is valid.
  std::optional<uint64_t> verifyMemAccess(const MemoryObject &MO,
                                          const APInt &Address,
                                          uint64_t AccessSize,
                                          uint64_t Alignment, bool IsStore) {
    // Loading from a stack object outside its lifetime is not undefined
    // behavior and returns a poison value instead. Storing to it is still
    // undefined behavior.
    if (IsStore ? MO.getState() != MemoryObjectState::Alive
                : MO.getState() == MemoryObjectState::Freed) {
      reportImmediateUB("Try to access a dead memory object.");
      return std::nullopt;
    }

    assert(isPowerOf2_64(Alignment) && "Alignment should be a power of 2.");
    if (Address.countr_zero() < Log2_64(Alignment)) {
      reportImmediateUB("Misaligned memory access.");
      return std::nullopt;
    }

    if (AccessSize > MO.getSize() || Address.ult(MO.getAddress())) {
      reportImmediateUB("Memory access is out of bounds.");
      return std::nullopt;
    }

    APInt Offset = Address - MO.getAddress();

    if (Offset.ugt(MO.getSize() - AccessSize)) {
      reportImmediateUB("Memory access is out of bounds.");
      return std::nullopt;
    }

    return Offset.getZExtValue();
  }

  AnyValue load(const AnyValue &Ptr, uint64_t Align, Type *ValTy) {
    if (Ptr.isPoison()) {
      reportImmediateUB("Invalid memory access with a poison pointer.");
      return AnyValue::getPoisonValue(Ctx, ValTy);
    }
    auto &PtrVal = Ptr.asPointer();
    auto *MO = PtrVal.getMemoryObject();
    if (!MO) {
      reportImmediateUB(
          "Invalid memory access via a pointer with nullary provenance.");
      return AnyValue::getPoisonValue(Ctx, ValTy);
    }
    // TODO: pointer capability check
    if (auto Offset = verifyMemAccess(
            *MO, PtrVal.address(), Ctx.getEffectiveTypeStoreSize(ValTy), Align,
            /*IsStore=*/false)) {
      // Load from a dead stack object yields poison value.
      if (MO->getState() == MemoryObjectState::Dead)
        return AnyValue::getPoisonValue(Ctx, ValTy);

      return Ctx.load(*MO, *Offset, ValTy);
    }
    return AnyValue::getPoisonValue(Ctx, ValTy);
  }

  void store(const AnyValue &Ptr, uint64_t Align, const AnyValue &Val,
             Type *ValTy) {
    if (Ptr.isPoison()) {
      reportImmediateUB("Invalid memory access with a poison pointer.");
      return;
    }
    auto &PtrVal = Ptr.asPointer();
    auto *MO = PtrVal.getMemoryObject();
    if (!MO) {
      reportImmediateUB(
          "Invalid memory access via a pointer with nullary provenance.");
      return;
    }
    // TODO: pointer capability check
    if (auto Offset = verifyMemAccess(
            *MO, PtrVal.address(), Ctx.getEffectiveTypeStoreSize(ValTy), Align,
            /*IsStore=*/true))
      Ctx.store(*MO, *Offset, Val, ValTy);
  }

  AnyValue computePtrAdd(const Pointer &Ptr, const APInt &Offset,
                         GEPNoWrapFlags Flags, AnyValue &AccumulatedOffset) {
    if (Offset.isZero())
      return Ptr;
    APInt IndexBits = Ptr.address().trunc(Offset.getBitWidth());
    auto NewIndex = addNoWrap(IndexBits, Offset, /*HasNSW=*/false,
                              Flags.hasNoUnsignedWrap());
    if (NewIndex.isPoison())
      return AnyValue::poison();
    if (Flags.hasNoUnsignedSignedWrap()) {
      // The successive addition of the current address, truncated to the
      // pointer index type and interpreted as an unsigned number, and each
      // offset, interpreted as a signed number, does not wrap the pointer index
      // type.
      if (Offset.isNonNegative() ? NewIndex.asInteger().ult(IndexBits)
                                 : NewIndex.asInteger().ugt(IndexBits))
        return AnyValue::poison();
    }
    APInt NewAddr = Ptr.address();
    NewAddr.insertBits(NewIndex.asInteger(), 0);

    auto *MO = Ptr.getMemoryObject();
    if (Flags.isInBounds() && (!MO || !MO->inBounds(NewAddr)))
      return AnyValue::poison();

    if (!AccumulatedOffset.isPoison()) {
      AccumulatedOffset =
          addNoWrap(AccumulatedOffset.asInteger(), Offset,
                    Flags.hasNoUnsignedSignedWrap(), Flags.hasNoUnsignedWrap());
      if (AccumulatedOffset.isPoison())
        return AnyValue::poison();
    }

    // Should not expose provenance here even if the new address doesn't point
    // to the original object.
    return Ptr.getWithNewAddr(NewAddr);
  }

  AnyValue computePtrAdd(const AnyValue &Ptr, const APInt &Offset,
                         GEPNoWrapFlags Flags, AnyValue &AccumulatedOffset) {
    if (Ptr.isPoison())
      return AnyValue::poison();
    return computePtrAdd(Ptr.asPointer(), Offset, Flags, AccumulatedOffset);
  }

  AnyValue computeScaledPtrAdd(const AnyValue &Ptr, const AnyValue &Index,
                               const APInt &Scale, GEPNoWrapFlags Flags,
                               AnyValue &AccumulatedOffset) {
    if (Ptr.isPoison() || Index.isPoison())
      return AnyValue::poison();
    assert(Ptr.isPointer() && Index.isInteger() && "Unexpected type.");
    if (Scale.isOne())
      return computePtrAdd(Ptr, Index.asInteger(), Flags, AccumulatedOffset);
    auto ScaledOffset =
        mulNoWrap(Index.asInteger(), Scale, Flags.hasNoUnsignedSignedWrap(),
                  Flags.hasNoUnsignedWrap());
    if (ScaledOffset.isPoison())
      return AnyValue::poison();
    return computePtrAdd(Ptr, ScaledOffset.asInteger(), Flags,
                         AccumulatedOffset);
  }

  AnyValue canonicalizeIndex(const AnyValue &Idx, unsigned IndexBitWidth,
                             GEPNoWrapFlags Flags) {
    if (Idx.isPoison())
      return AnyValue::poison();
    auto &IdxInt = Idx.asInteger();
    if (IdxInt.getBitWidth() == IndexBitWidth)
      return Idx;
    if (IdxInt.getBitWidth() > IndexBitWidth) {
      if (Flags.hasNoUnsignedSignedWrap() &&
          !IdxInt.isSignedIntN(IndexBitWidth))
        return AnyValue::poison();

      if (Flags.hasNoUnsignedWrap() && !IdxInt.isIntN(IndexBitWidth))
        return AnyValue::poison();

      return IdxInt.trunc(IndexBitWidth);
    }
    return IdxInt.sext(IndexBitWidth);
  }

public:
  InstExecutor(Context &C, EventHandler &H, Function &F,
               ArrayRef<AnyValue> Args, AnyValue &RetVal)
      : Ctx(C), DL(Ctx.getDataLayout()), Handler(H), Status(true) {
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
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end: {
      auto *Ptr = CB.getArgOperand(0);
      if (isa<PoisonValue>(Ptr))
        return AnyValue();
      auto *MO = getValue(Ptr).asPointer().getMemoryObject();
      assert(MO && "Memory object accessed by lifetime intrinsic should be "
                   "always valid.");
      if (IID == Intrinsic::lifetime_start) {
        MO->setState(MemoryObjectState::Alive);
        fill(MO->getBytes(), Byte::undef());
      } else {
        MO->setState(MemoryObjectState::Dead);
        fill(MO->getBytes(), Byte::poison());
      }
      return AnyValue();
    }
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

  void visitAllocaInst(AllocaInst &AI) {
    uint64_t AllocSize = Ctx.getEffectiveTypeAllocSize(AI.getAllocatedType());
    if (AI.isArrayAllocation()) {
      auto &Size = getValue(AI.getArraySize());
      if (Size.isPoison()) {
        reportImmediateUB("Alloca with poison array size.");
        return;
      }
      if (Size.asInteger().getActiveBits() > 64) {
        reportImmediateUB(
            "Alloca with large array size that overflows uint64_t.");
        return;
      }
      bool Overflowed = false;
      AllocSize = SaturatingMultiply(AllocSize, Size.asInteger().getZExtValue(),
                                     &Overflowed);
      if (Overflowed) {
        reportImmediateUB(
            "Alloca with allocation size that overflows uint64_t.");
        return;
      }
    }
    // If it is used by llvm.lifetime.start, it should be initially dead.
    bool IsInitiallyDead = any_of(AI.users(), [](User *U) {
      return match(U, m_Intrinsic<Intrinsic::lifetime_start>());
    });
    auto Obj = Ctx.allocate(AllocSize, AI.getPointerAlignment(DL).value(),
                            AI.getName(), AI.getAddressSpace(),
                            IsInitiallyDead ? MemInitKind::Poisoned
                                            : MemInitKind::Uninitialized);
    if (!Obj) {
      reportError("Insufficient stack space.");
      return;
    }
    CurrentFrame->Allocas.push_back(Obj);
    setResult(AI, Ctx.deriveFromMemoryObject(Obj));
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    uint32_t IndexBitWidth =
        DL.getIndexSizeInBits(GEP.getType()->getPointerAddressSpace());
    GEPNoWrapFlags Flags = GEP.getNoWrapFlags();
    AnyValue Res = getValue(GEP.getPointerOperand());
    AnyValue AccumulatedOffset = APInt(IndexBitWidth, 0);
    if (Res.isAggregate())
      AccumulatedOffset =
          AnyValue::getVectorSplat(AccumulatedOffset, Res.asAggregate().size());
    auto ApplyScaledOffset = [&](const AnyValue &Index, const APInt &Scale) {
      if (Index.isAggregate() && !Res.isAggregate()) {
        Res = AnyValue::getVectorSplat(Res, Index.asAggregate().size());
        AccumulatedOffset = AnyValue::getVectorSplat(
            AccumulatedOffset, Index.asAggregate().size());
      }
      if (Index.isAggregate() && Res.isAggregate()) {
        for (auto &&[ResElem, IndexElem, OffsetElem] :
             zip(Res.asAggregate(), Index.asAggregate(),
                 AccumulatedOffset.asAggregate()))
          ResElem = computeScaledPtrAdd(
              ResElem, canonicalizeIndex(IndexElem, IndexBitWidth, Flags),
              Scale, Flags, OffsetElem);
      } else {
        AnyValue CanonicalIndex =
            canonicalizeIndex(Index, IndexBitWidth, Flags);
        if (Res.isAggregate()) {
          for (auto &&[ResElem, OffsetElem] :
               zip(Res.asAggregate(), AccumulatedOffset.asAggregate()))
            ResElem = computeScaledPtrAdd(ResElem, CanonicalIndex, Scale, Flags,
                                          OffsetElem);
        } else {
          Res = computeScaledPtrAdd(Res, CanonicalIndex, Scale, Flags,
                                    AccumulatedOffset);
        }
      }
    };

    for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
         GTI != GTE; ++GTI) {
      Value *V = GTI.getOperand();

      // Fast path for zero offsets.
      if (auto *CI = dyn_cast<ConstantInt>(V)) {
        if (CI->isZero())
          continue;
      }
      if (isa<ConstantAggregateZero>(V))
        continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        unsigned ElementIdx = cast<ConstantInt>(V)->getZExtValue();
        const StructLayout *SL = DL.getStructLayout(STy);
        // Element offset is in bytes.
        ApplyScaledOffset(
            APInt(IndexBitWidth, SL->getElementOffset(ElementIdx)),
            APInt(IndexBitWidth, 1));
        continue;
      }

      // Truncate if type size exceeds index space.
      // TODO: Should be documented in LangRef: GEPs with nowrap flags should
      // return poison when the type size exceeds index space.
      TypeSize Offset = GTI.getSequentialElementStride(DL);
      APInt Scale(IndexBitWidth,
                  Offset.isScalable()
                      ? Offset.getKnownMinValue() * Ctx.getVScale()
                      : Offset.getFixedValue(),
                  /*isSigned=*/false, /*implicitTrunc=*/true);
      if (!Scale.isZero())
        ApplyScaledOffset(getValue(V), Scale);
    }

    setResult(GEP, std::move(Res));
  }

  void visitIntToPtr(IntToPtrInst &I) {
    return visitUnOp(I, [&](const AnyValue &V) -> AnyValue {
      if (V.isPoison())
        return AnyValue::poison();
      // TODO: expose provenance
      // TODO: check metadata
      return Pointer(V.asInteger().zextOrTrunc(
          DL.getPointerSizeInBits(I.getType()->getPointerAddressSpace())));
    });
  }

  void visitLoadInst(LoadInst &LI) {
    auto RetVal = load(getValue(LI.getPointerOperand()), LI.getAlign().value(),
                       LI.getType());
    // TODO: track volatile loads
    // TODO: handle metadata
    setResult(LI, std::move(RetVal));
  }

  void visitStoreInst(StoreInst &SI) {
    auto &Ptr = getValue(SI.getPointerOperand());
    auto &Val = getValue(SI.getValueOperand());
    // TODO: track volatile stores
    // TODO: handle metadata
    store(Ptr, SI.getAlign().value(), Val, SI.getValueOperand()->getType());
    if (Status)
      Status &= Handler.onInstructionExecuted(SI, AnyValue());
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
        // Free stack objects allocated in this frame.
        for (auto &Obj : Top.Allocas)
          Ctx.free(Obj->getAddress());
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
