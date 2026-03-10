//===--- Interpreter.h - Interpreter Loop for llubi -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_INSTEXECUTOR_H
#define LLVM_TOOLS_LLUBI_INSTEXECUTOR_H

#include "Context.h"
#include "Value.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstVisitor.h"

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
  std::optional<ProgramExitInfo> ExitInfo;
  Frame *CurrentFrame = nullptr;
  AnyValue None;

  void reportImmediateUB(StringRef Msg);
  void reportError(StringRef Msg);

  /// Check if the upcoming memory access is valid. Returns the offset relative
  /// to the underlying object if it is valid.
  std::optional<uint64_t> verifyMemAccess(const MemoryObject &MO,
                                          const APInt &Address,
                                          uint64_t AccessSize, Align Alignment,
                                          bool IsStore);

  const AnyValue &getValue(Value *V);
  void setResult(Instruction &I, AnyValue V);

  AnyValue computeUnOp(Type *Ty, const AnyValue &Operand,
                       function_ref<AnyValue(const AnyValue &)> ScalarFn);
  void visitUnOp(Instruction &I,
                 function_ref<AnyValue(const AnyValue &)> ScalarFn);

  void visitIntUnOp(Instruction &I,
                    function_ref<AnyValue(const APInt &)> ScalarFn);

  AnyValue computeBinOp(
      Type *Ty, const AnyValue &LHS, const AnyValue &RHS,
      function_ref<AnyValue(const AnyValue &, const AnyValue &)> ScalarFn);
  void visitBinOp(
      Instruction &I,
      function_ref<AnyValue(const AnyValue &, const AnyValue &)> ScalarFn);

  void
  visitIntBinOp(Instruction &I,
                function_ref<AnyValue(const APInt &, const APInt &)> ScalarFn);

  void jumpTo(Instruction &Terminator, BasicBlock *DestBB);

  /// Helper function to determine whether an inline asm is a no-op, which is
  /// used to implement black_box style optimization blockers.
  bool isNoopInlineAsm(Value *V, Type *RetTy);

  AnyValue load(const AnyValue &Ptr, Align Alignment, Type *ValTy);
  void store(const AnyValue &Ptr, Align Alignment, const AnyValue &Val,
             Type *ValTy);

  AnyValue computePtrAdd(const Pointer &Ptr, const APInt &Offset,
                         GEPNoWrapFlags Flags, AnyValue &AccumulatedOffset);
  AnyValue computePtrAdd(const AnyValue &Ptr, const APInt &Offset,
                         GEPNoWrapFlags Flags, AnyValue &AccumulatedOffset);
  AnyValue computeScaledPtrAdd(const AnyValue &Ptr, const AnyValue &Index,
                               const APInt &Scale, GEPNoWrapFlags Flags,
                               AnyValue &AccumulatedOffset);

  AnyValue canonicalizeIndex(const AnyValue &Idx, unsigned IndexBitWidth,
                             GEPNoWrapFlags Flags);

  friend class LibraryEnvironment;

public:
  InstExecutor(Context &C, EventHandler &H, Function &F,
               ArrayRef<AnyValue> Args, AnyValue &RetVal)
      : Ctx(C), DL(Ctx.getDataLayout()), Handler(H), Status(true) {
    CallStack.emplace_back(F, /*CallSite=*/nullptr, /*LastFrame=*/nullptr, Args,
                           RetVal, Ctx.getTLIImpl());
  }

  void visitReturnInst(ReturnInst &RI);
  void visitBranchInst(BranchInst &BI);
  void visitSwitchInst(SwitchInst &SI);
  void visitUnreachableInst(UnreachableInst &);
  void visitCallBrInst(CallBrInst &CI);
  void visitIndirectBrInst(IndirectBrInst &IBI);

  void returnFromCallee();

  AnyValue callIntrinsic(CallBase &CB);
  AnyValue callLibFunc(CallBase &CB, Function *ResolvedCallee);

  void requestProgramExit(ProgramExitKind Kind, uint64_t ExitCode = 0);
  bool hasProgramExit() const { return ExitInfo.has_value(); }
  std::optional<ProgramExitInfo> getProgramExitInfo() const { return ExitInfo; }

  void enterCall(CallBase &CB);
  void visitCallInst(CallInst &CI);
  void visitInvokeInst(InvokeInst &II);
  void visitAdd(BinaryOperator &I);
  void visitSub(BinaryOperator &I);
  void visitMul(BinaryOperator &I);
  void visitSDiv(BinaryOperator &I);
  void visitSRem(BinaryOperator &I);
  void visitUDiv(BinaryOperator &I);
  void visitURem(BinaryOperator &I);
  void visitTruncInst(TruncInst &Trunc);
  void visitZExtInst(ZExtInst &ZExt);
  void visitSExtInst(SExtInst &SExt);
  void visitAnd(BinaryOperator &I);
  void visitXor(BinaryOperator &I);
  void visitOr(BinaryOperator &I);
  void visitShl(BinaryOperator &I);
  void visitLShr(BinaryOperator &I);
  void visitAShr(BinaryOperator &I);
  void visitICmpInst(ICmpInst &I);
  void visitSelect(SelectInst &SI);
  void visitAllocaInst(AllocaInst &AI);
  void visitGetElementPtrInst(GetElementPtrInst &GEP);
  void visitIntToPtr(IntToPtrInst &I);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void visitInstruction(Instruction &I);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);
  void visitInsertElementInst(InsertElementInst &IEI);
  void visitExtractElementInst(ExtractElementInst &EEI);
  void visitShuffleVectorInst(ShuffleVectorInst &SVI);

  /// This function implements the main interpreter loop.
  /// It handles function calls in a non-recursive manner to avoid stack
  /// overflows.
  ExecutionStatus runMainLoop();
};
} // namespace llvm::ubi

#endif
