//===- ControlPoint.cpp - Synthesise the yk control point -----------------===//
//
// This pass finds the user's call to the dummy control point and replaces it
// with a call to a new control point that implements the necessary logic to
// drive the yk JIT.
//
// The pass converts an interpreter loop that looks like this:
//
// ```
// pc = 0;
// while (...) {
//     yk_control_point(); // <- dummy control point
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
//     yk_new_control_point(&cp_vars);
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>

#define DEBUG_TYPE "yk-control-point"
#define JIT_STATE_PREFIX "jit-state: "

using namespace llvm;

/// Find the call to the dummy control point that we want to patch.
/// Returns either a pointer the call instruction, or `nullptr` if the call
/// could not be found.
/// YKFIXME: For now assumes there's only one control point.
CallInst *findControlPointCall(Module &M) {
  // Find the declaration of `yk_control_point()`.
  Function *CtrlPoint = M.getFunction(YK_DUMMY_CONTROL_POINT);
  if (CtrlPoint == nullptr)
    return nullptr;

  // Find the call site of `yk_control_point()`.
  Value::user_iterator U = CtrlPoint->user_begin();
  if (U == CtrlPoint->user_end())
    return nullptr;

  return cast<CallInst>(*U);
}

/// Extract all live variables that need to be passed into the control point.
std::vector<Value *> getLiveVars(DominatorTree &DT, CallInst *OldCtrlPoint) {
  std::vector<Value *> Vec;
  Function *Func = OldCtrlPoint->getFunction();
  for (auto &BB : *Func) {
    if (!DT.dominates(cast<Instruction>(OldCtrlPoint), &BB)) {
      for (auto &I : BB) {
        if ((!I.getType()->isVoidTy()) &&
            (DT.dominates(&I, cast<Instruction>(OldCtrlPoint)))) {
          Vec.push_back(&I);
        }
      }
    }
  }
  return Vec;
}

namespace llvm {
void initializeYkControlPointPass(PassRegistry &);
}

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
          "ykllvm couldn't find the call to `yk_control_point()`");
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
      Context.emitError("yk_control_point() must be called inside a loop.");
      return false;
    }

    // Find all live variables just before the call to the control point.
    std::vector<Value *> LiveVals = getLiveVars(DT, OldCtrlPointCall);
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

    // Create the new control point.
    Type *YkLocTy = OldCtrlPointCall->getArgOperand(0)->getType();
    FunctionType *FType =
        FunctionType::get(Type::getVoidTy(Context),
                          {YkLocTy, CtrlPointVarsTy->getPointerTo()}, false);
    Function *NF = Function::Create(FType, GlobalVariable::ExternalLinkage,
                                    YK_NEW_CONTROL_POINT, M);

    // At the top of the function, instantiate a `YkCtrlPointStruct` to pass in
    // to the control point. We do so on the stack, so that we can pass the
    // struct by pointer.
    IRBuilder<> Builder(Caller->getEntryBlock().getFirstNonPHI());
    Value *InputStruct = Builder.CreateAlloca(CtrlPointVarsTy, 0, "");

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
        NF, {OldCtrlPointCall->getArgOperand(0), InputStruct});

    // Once the control point returns we need to extract the (potentially
    // mutated) values from the returned YkCtrlPointStruct and reassign them to
    // their corresponding live variables. In LLVM IR we can do this by simply
    // replacing all future references with the new values.
    LvIdx = 0;
    for (Value *LV : LiveVals) {
      Value *FieldPtr =
          Builder.CreateGEP(CtrlPointVarsTy, InputStruct,
                            {Builder.getInt32(0), Builder.getInt32(LvIdx)});
      Value *New = Builder.CreateLoad(TypeParams[LvIdx], FieldPtr);
      LV->replaceUsesWithIf(
          New, [&](Use &U) { return DT.dominates(NewCtrlPointCallInst, U); });
      assert(LvIdx != UINT_MAX);
      LvIdx++;
    }

    // Replace the call to the dummy control point.
    OldCtrlPointCall->eraseFromParent();

    // Generate new control point logic.
    return true;
  }
};
} // namespace

char YkControlPoint::ID = 0;
INITIALIZE_PASS(YkControlPoint, DEBUG_TYPE, "yk control point", false, false)

ModulePass *llvm::createYkControlPointPass() { return new YkControlPoint(); }
