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
  Value::user_iterator U = CtrlPoint->user_begin();
  if (U == CtrlPoint->user_end())
    return nullptr;

  return cast<CallInst>(*U);
}

/// Wrapper to make `std::set_difference` more concise.
///
/// Store the difference between `S1` and `S2` into `Into`.
void vset_difference(const std::set<Value *> &S1, const std::set<Value *> &S2,
                     std::set<Value *> &Into) {
  std::set_difference(S1.begin(), S1.end(), S2.begin(), S2.end(),
                      std::inserter(Into, Into.begin()));
}

/// Wrapper to make `std::set_union` more concise.
///
/// Store the union of `S1` and `S2` into `Into`.
void vset_union(const std::set<Value *> &S1, const std::set<Value *> &S2,
                std::set<Value *> &Into) {
  std::set_union(S1.begin(), S1.end(), S2.begin(), S2.end(),
                 std::inserter(Into, Into.begin()));
}

namespace llvm {
// A liveness analysis for LLVM IR.
//
// This is based on the algorithm shown in Chapter 10 of the book:
//
//   Modern Compiler Implementation in Java (2nd edition)
//   by Andrew W. Appel
class LivenessAnalysis {
  std::map<Instruction *, std::set<Value *>> In;

  /// Find the successor instructions of the specified instruction.
  std::set<Instruction *> getSuccessorInstructions(Instruction *I) {
    Instruction *Term = I->getParent()->getTerminator();
    std::set<Instruction *> SuccInsts;
    if (I != Term) {
      // Non-terminating instruction: the sole successor instruction is the
      // next instruction in the block.
      SuccInsts.insert(I->getNextNode());
    } else {
      // Terminating instruction: successor instructions are the first
      // instructions of all successor blocks.
      for (unsigned SuccIdx = 0; SuccIdx < Term->getNumSuccessors(); SuccIdx++)
        SuccInsts.insert(&*Term->getSuccessor(SuccIdx)->begin());
    }
    return SuccInsts;
  }

  /// Replaces the value set behind the pointer `S` with the value set `R` and
  /// returns whether the set behind `S` changed.
  bool updateValueSet(std::set<Value *> *S, const std::set<Value *> R) {
    bool Changed = (*S != R);
    *S = R;
    return Changed;
  }

public:
  LivenessAnalysis(Function *Func) {
    // Compute defs and uses for each instruction.
    std::map<Instruction *, std::set<Value *>> Defs;
    std::map<Instruction *, std::set<Value *>> Uses;
    for (BasicBlock &BB : *Func) {
      for (Instruction &I : BB) {
        // Record what this instruction defines.
        if (!I.getType()->isVoidTy())
          Defs[&I].insert(cast<Value>(&I));

        // Record what this instruction uses.
        //
        // Note that Phi nodes are special and must be skipped. If we consider
        // their operands as uses, then Phi nodes in loops may use variables
        // before they are defined, and this messes with the algorithm.
        //
        // The book doesn't cover this quirk, as it explains liveness for
        // non-SSA form, and thus doesn't need to worry about Phi nodes.
        if (isa<PHINode>(I))
          continue;

        for (auto *U = I.op_begin(); U < I.op_end(); U++) {
          if ((!isa<Constant>(U)) && (!isa<BasicBlock>(U)) &&
              (!isa<MetadataAsValue>(U)) && (!isa<InlineAsm>(U))) {
            Uses[&I].insert(*U);
          }
        }
      }
    }

    // A function implicitly defines its arguments.
    //
    // To propagate the arguments properly we pretend that the first instruction
    // in the entry block defines the arguments.
    Instruction *FirstInst = &*Func->getEntryBlock().begin();
    for (auto &Arg : Func->args())
      Defs[FirstInst].insert(&Arg);

    // Compute the live sets for each instruction.
    //
    // This is the fixed-point of the following data-flow equations (page 206
    // in the book referenced above):
    //
    //    in[I] = use[I] ∪ (out[I] - def[I])
    //
    //    out[I] =       ∪
    //             (S in succ[I])    in[S]
    //
    // Note that only the `In` map is kept after this constructor ends, so
    // only `In` is a field.
    std::map<Instruction *, std::set<Value *>> Out;
    bool Changed;
    do {
      Changed = false;
      // As the book explains, fixed-points are reached quicker if we process
      // control flow in "approximately reverse direction" and if we compute
      // `out[I]` before `in[I]`.
      //
      // Because the alrogithm works by propagating liveness from use sites
      // backwards to def sites (where liveness is killed), by working
      // backwards we are able to propagate long runs of liveness in one
      // iteration of the algorithm.
      for (BasicBlock *BB : post_order(&*Func)) {
        for (BasicBlock::reverse_iterator II = BB->rbegin(); II != BB->rend();
             II++) {
          Instruction *I = &*II;
          // Update out[I].
          std::set<Instruction *> SuccInsts = getSuccessorInstructions(I);
          std::set<Value *> NewOut;
          for (Instruction *SI : SuccInsts) {
            NewOut.insert(In[SI].begin(), In[SI].end());
          }
          Changed |= updateValueSet(&Out[I], std::move(NewOut));

          // Update in[I].
          std::set<Value *> OutMinusDef;
          vset_difference(Out[I], Defs[I], OutMinusDef);

          std::set<Value *> NewIn;
          vset_union(Uses[I], OutMinusDef, NewIn);
          Changed |= updateValueSet(&In[I], std::move(NewIn));
        }
      }
    } while (Changed); // Until a fixed-point.
  }

  /// Returns the set of live variables immediately before the specified
  /// instruction.
  std::set<Value *> getLiveVarsBefore(Instruction *I) { return In[I]; }
};

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
    std::set<Value *> LiveVals = LA.getLiveVarsBefore(OldCtrlPointCall);
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
    // bool new_control_point(YkMT*, YkLocation*, CtrlPointVars*)
    FunctionType *FType = FunctionType::get(
        Type::getInt1Ty(Context),
        {YkMTTy, YkLocTy, CtrlPointVarsTy->getPointerTo()}, false);
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
        NF, {OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_MT_IDX),
             OldCtrlPointCall->getArgOperand(YK_CONTROL_POINT_ARG_LOC_IDX),
             InputStruct});

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
    Type *RetTy = Caller->getReturnType();
    if (RetTy->isVoidTy()) {
      Builder.CreateRetVoid();
    } else {
      Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 0));
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
