//===- ControlPoint.cpp - Synthesise the yk control point -----------------===//
//
// This pass finds the user's call to the dummy control point and replaces it
// with a call to a new control point that implements the necessary logic to
// drive the yk JIT.
//
// The pass converts an interpreter loop that looks like this:
//
// ```
// YkMT *mt = ...;
// YkLocation loc = ...;
// pc = 0;
// while (...) {
//     yk_mt_control_point(mt, loc); // <- dummy control point
//     bc = program[pc];
//     switch (bc) {
//         // bytecode handlers here.
//     }
// }
// ```
//
// Into one that looks like this (note that this transformation happens at the
// IR level):
//
// ```
// // The YkCtrlPointStruct contains one member for each live LLVM variable
// // just before the call to the control point.
// struct YkCtrlPointStruct {
//     size_t pc;
// }
//
// struct YkCtrlPointStruct cp_vars;
// pc = 0;
// while (...) {
//     // Now we call the patched control point.
//     cp_vars.pc = pc;
//     __ykrt__control_point(mt, loc, &cp_vars);
//     pc = cp_vars.pc;
//     bc = program[pc];
//     switch (bc) {
//         // bytecode handlers here.
//     }
// }
// ```
//
// Note that this transformation occurs at the LLVM IR level. The above example
// is shown as C code for easy comprehension.

#include "llvm/Transforms/Yk/ControlPoint.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/LivenessAnalysis.h"
#include <map>

#define DEBUG_TYPE "yk-control-point"
#define JIT_STATE_PREFIX "jit-state: "

#define YK_OLD_CONTROL_POINT_NUM_ARGS 2

#define YK_CONTROL_POINT_ARG_MT_IDX 0
#define YK_CONTROL_POINT_ARG_LOC_IDX 1
#define YK_CONTROL_POINT_ARG_VARS_IDX 2
#define YK_CONTROL_POINT_NUM_ARGS 3

using namespace llvm;

/// Find the call to the dummy control point that we want to patch.
/// Returns either a pointer the call instruction, or `nullptr` if the call
/// could not be found.
/// YKFIXME: For now assumes there's only one control point.
CallInst *findControlPointCall(Module &M) {
  // Find the declaration of `yk_mt_control_point()`.
  Function *CtrlPoint = M.getFunction(YK_DUMMY_CONTROL_POINT);
  if (CtrlPoint == nullptr)
    return nullptr;

  // Find the call site of `yk_mt_control_point()`.
  const Value::user_iterator U = CtrlPoint->user_begin();
  if (U == CtrlPoint->user_end())
    return nullptr;

  return cast<CallInst>(*U);
}

namespace llvm {
void initializeYkControlPointPass(PassRegistry &);
} // namespace llvm

namespace {
class YkControlPoint : public ModulePass {
public:
  static char ID;
  YkControlPoint() : ModulePass(ID) {
    initializeYkControlPointPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    // Locate the "dummy" control point provided by the user.
    CallInst *OldCtrlPointCall = findControlPointCall(M);
    if (OldCtrlPointCall == nullptr) {
      Context.emitError(
          "ykllvm couldn't find the call to `yk_mt_control_point()`");
      return false;
    }

    // Get function containing the control point.
    Function *Caller = OldCtrlPointCall->getFunction();

    // Check that the control point is inside a loop.
    DominatorTree DT(*Caller);
    const LoopInfo Loops(DT);
    if (!std::any_of(Loops.begin(), Loops.end(), [OldCtrlPointCall](Loop *L) {
          return L->contains(OldCtrlPointCall);
        })) {
      ;
      Context.emitError("yk_mt_control_point() must be called inside a loop.");
      return false;
    }

    // Find all live variables just before the call to the control point.
    LivenessAnalysis LA(OldCtrlPointCall->getFunction());
    const std::set<Value *> LiveVals = LA.getLiveVarsBefore(OldCtrlPointCall);
    if (LiveVals.size() == 0) {
      Context.emitError(
          "The interpreter loop has no live variables!\n"
          "ykllvm doesn't support this scenario, as such an interpreter would "
          "make little sense.");
      return false;
    }

    // Generate the YkCtrlPointVars struct. This struct is used to package up a
    // copy of all LLVM variables that are live just before the call to the
    // control point. These are passed in to the patched control point so that
    // they can be used as inputs and outputs to JITted trace code. The control
    // point returns a new YkCtrlPointVars whose members may have been mutated
    // by JITted trace code (if a trace was executed).
    std::vector<Type *> TypeParams;
    for (Value *V : LiveVals) {
      TypeParams.push_back(V->getType());
    }
    StructType *CtrlPointVarsTy =
        StructType::create(TypeParams, "YkCtrlPointVars");

    // The old control point should be of the form:
    //    control_point(YkMT*, YkLocation*)
    assert(OldCtrlPointCall->arg_size() == YK_OLD_CONTROL_POINT_NUM_ARGS);
    Type *YkMTTy =
        OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_MT_IDX)->getType();
    Type *YkLocTy =
        OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_LOC_IDX)
            ->getType();

    // Create the new control point, which is of the form:
    //   bool new_control_point(YkMT*, YkLocation*, CtrlPointVars*,
    //   ReturnValue*)
    // If the return type of the control point's caller is void (i.e. if a
    // function f calls yk_control_point and f's return type is void), create
    // an Int1 pointer as a dummy. We have to pass something as the yk_stopgap
    // signature expects a pointer, even if its never used.
    Type *ReturnTy = Caller->getReturnType();
    Type *ReturnPtrTy;
    if (ReturnTy->isVoidTy()) {
      // Create dummy pointer which we pass in but which is never written to.
      ReturnPtrTy = Type::getInt1Ty(Context);
    } else {
      ReturnPtrTy = ReturnTy;
    }
    FunctionType *FType =
        FunctionType::get(Type::getInt1Ty(Context),
                          {YkMTTy, YkLocTy, CtrlPointVarsTy->getPointerTo(),
                           ReturnPtrTy->getPointerTo()},
                          false);
    Function *NF = Function::Create(FType, GlobalVariable::ExternalLinkage,
                                    YK_NEW_CONTROL_POINT, M);

    // At the top of the function, instantiate a `YkCtrlPointStruct` to pass in
    // to the control point. We do so on the stack, so that we can pass the
    // struct by pointer.
    IRBuilder<> Builder(Caller->getEntryBlock().getFirstNonPHI());
    Value *InputStruct = Builder.CreateAlloca(CtrlPointVarsTy, 0, "");

    // Also at the top, generate storage for the interpreted return of the
    // control points caller.
    Value *ReturnPtr = Builder.CreateAlloca(ReturnPtrTy, 0, "");

    Builder.SetInsertPoint(OldCtrlPointCall);
    unsigned LvIdx = 0;
    for (Value *LV : LiveVals) {
      Value *FieldPtr =
          Builder.CreateGEP(CtrlPointVarsTy, InputStruct,
                            {Builder.getInt32(0), Builder.getInt32(LvIdx)});
      Builder.CreateStore(LV, FieldPtr);
      assert(LvIdx != UINT_MAX);
      LvIdx++;
    }

    // Insert call to the new control point.
    Instruction *NewCtrlPointCallInst = Builder.CreateCall(
        NF, {OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_MT_IDX),
             OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_LOC_IDX),
             InputStruct, ReturnPtr});

    // Once the control point returns we need to extract the (potentially
    // mutated) values from the returned YkCtrlPointStruct and reassign them to
    // their corresponding live variables. In LLVM IR we can do this by simply
    // replacing all future references with the new values.
    LvIdx = 0;
    Instruction *New;
    for (Value *LV : LiveVals) {
      Value *FieldPtr =
          Builder.CreateGEP(CtrlPointVarsTy, InputStruct,
                            {Builder.getInt32(0), Builder.getInt32(LvIdx)});
      New = Builder.CreateLoad(TypeParams[LvIdx], FieldPtr);
      LV->replaceUsesWithIf(
          New, [&](Use &U) { return DT.dominates(NewCtrlPointCallInst, U); });
      assert(LvIdx != UINT_MAX);
      LvIdx++;
    }

    // Replace the call to the dummy control point.
    OldCtrlPointCall->eraseFromParent();

    // Get the result of the control point call. If it returns true, that means
    // the stopgap interpreter has interpreted a return so we need to return as
    // well.

    // Create the new exit block.
    BasicBlock *ExitBB = BasicBlock::Create(Context, "", Caller);
    Builder.SetInsertPoint(ExitBB);
    // YKFIXME: We need to return the value of interpreted return and the return
    // type must be that of the control point's caller.
    if (ReturnTy->isVoidTy()) {
      Builder.CreateRetVoid();
    } else {
      Value *ReturnValue = Builder.CreateLoad(ReturnTy, ReturnPtr);
      Builder.CreateRet(ReturnValue);
    }

    // To do so we need to first split up the current block and then
    // insert a conditional branch that either continues or returns.

    BasicBlock *BB = NewCtrlPointCallInst->getParent();
    BasicBlock *ContBB = BB->splitBasicBlock(New);

    Instruction &OldBr = BB->back();
    OldBr.eraseFromParent();
    Builder.SetInsertPoint(BB);
    Builder.CreateCondBr(NewCtrlPointCallInst, ExitBB, ContBB);

#ifndef NDEBUG
    // Our pass runs after LLVM normally does its verify pass. In debug builds
    // we run it again to check that our pass is generating valid IR.
    if (verifyModule(M, &errs())) {
      Context.emitError("Control point pass generated invalid IR!");
      return false;
    }
#endif
    return true;
  }
};
} // namespace

char YkControlPoint::ID = 0;
INITIALIZE_PASS(YkControlPoint, DEBUG_TYPE, "yk control point", false, false)

ModulePass *llvm::createYkControlPointPass() { return new YkControlPoint(); }
