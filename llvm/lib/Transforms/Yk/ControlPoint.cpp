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
#include "llvm/IR/DiagnosticInfo.h"
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
      // This program doesn't have a control point. We can't do any
      // transformations on it, but we do still want to compile it.
      Context.diagnose(DiagnosticInfoInlineAsm(
          "ykllvm couldn't find the call to `yk_mt_control_point()`",
          DS_Warning));
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
    const std::vector<Value *> LiveVals =
        LA.getLiveVarsBefore(OldCtrlPointCall);
    if (LiveVals.size() == 0) {
      Context.emitError(
          "The interpreter loop has no live variables!\n"
          "ykllvm doesn't support this scenario, as such an interpreter would "
          "make little sense.");
      return false;
    }

    // The old control point should be of the form:
    //    control_point(YkMT*, YkLocation*)
    assert(OldCtrlPointCall->arg_size() == YK_OLD_CONTROL_POINT_NUM_ARGS);
    Type *YkMTTy =
        OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_MT_IDX)->getType();
    Type *YkLocTy =
        OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_LOC_IDX)
            ->getType();

    // Create the new control point, which is of the form:
    //   void new_control_point(YkMT*, YkLocation*, i64)
    Type *Int64Ty = Type::getInt64Ty(Context);
    FunctionType *FType = FunctionType::get(Type::getVoidTy(Context),
                                            {YkMTTy, YkLocTy, Int64Ty}, false);
    Function *NF = Function::Create(FType, GlobalVariable::ExternalLinkage,
                                    YK_NEW_CONTROL_POINT, M);

    // At the top of the function, instantiate a `YkCtrlPointStruct` to pass in
    // to the control point. We do so on the stack, so that we can pass the
    // struct by pointer.
    IRBuilder<> Builder(OldCtrlPointCall);

    // Insert call to the new control point. The last argument is the stackmap
    // id belonging to the control point. This is temporarily set to INT_MAX
    // and overwritten by the stackmap pass.
    Builder.CreateCall(
        NF, {OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_MT_IDX),
             OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_LOC_IDX),
             Builder.getInt64(UINT64_MAX)});

    // Replace the call to the dummy control point.
    OldCtrlPointCall->eraseFromParent();

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
