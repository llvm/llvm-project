//===- StackMaps.cpp - Pass to add Stackmaps for JIT deoptimisation --===//
//
// Add stackmap calls to the AOT-compiled module at each point that there could
// be a guard failure once traced and JITted. The information collected by
// those calls allows us to recreate the stack after a guard failure and return
// straight back to the AOT-compiled interpreter without the need for a
// stop-gap interpreter.

#include "llvm/Transforms/Yk/Stackmaps.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/ControlPoint.h"
#include "llvm/Transforms/Yk/LivenessAnalysis.h"
#include "llvm/Transforms/Yk/ModuleClone.h"
#include <map>

#define DEBUG_TYPE "yk-stackmaps"

using namespace llvm;

namespace llvm {
void initializeYkStackmapsPass(PassRegistry &);
} // namespace llvm

namespace {

class YkStackmaps : public ModulePass {
private:
  uint64_t StackmapIDStart;

public:
  static char ID;
  YkStackmaps(uint64_t stackmapIDStart = 1)
      : ModulePass(ID), StackmapIDStart(stackmapIDStart) {
    initializeYkStackmapsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    const Intrinsic::ID SMFuncID =
        Function::lookupIntrinsicID("llvm.experimental.stackmap");
    if (SMFuncID == Intrinsic::not_intrinsic) {
      Context.emitError("can't find stackmap()");
      return false;
    }

    std::map<Instruction *, std::vector<Value *>> SMCalls;
    for (Function &F : M) {
      if (F.empty()) // skip declarations.
        continue;
      if (F.getName().startswith(YK_UNOPT_PREFIX)) // skip cloned functions
        continue;

      LivenessAnalysis LA(&F);
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (isa<CallInst>(I)) {
            CallInst &CI = cast<CallInst>(I);
            if (CI.isInlineAsm())
              continue;
            // We don't need to insert stackmaps after intrinsics. But since we
            // can't tell if an indirect call is an intrinsic at compile time,
            // emit a stackmap in those cases too.

            if (!CI.isIndirectCall()) {
              if (CI.getCalledFunction()->isIntrinsic()) {
                // This also skips the control point which is called via a
                // patchpoint intrinsic, which already emits a stackmap.
                continue;
              }

              if (CI.getCalledFunction()->isDeclaration() &&
                  !CI.getCalledFunction()->getName().startswith(
                      "__yk_promote")) {
                continue;
              }
            }
            // OPT: Geting the live vars from before `I` might include
            // variables that die immediately after the call. We should try
            // using the live variables *after* the call, but making sure not to
            // include the value computed by `I` itself (since that doesn't
            // exist at the time of the call).
            SMCalls.insert({&I, LA.getLiveVarsBefore(&I)});

            if (!CI.isIndirectCall() &&
                CI.getCalledFunction()->getName().startswith("__yk_promote")) {
              // If it's a call to yk_promote* then the return value of the
              // promotion needs to be tracked too. This is because the trace
              // builder will recognise calls to yk_promote specially and
              // replace them with a guard that deopts to immediately after the
              // call (at which point the return value is live and needs to be
              // materialised for correctness).
              SMCalls[&I].push_back(&CI);
            }
          } else if ((isa<BranchInst>(I) &&
                      cast<BranchInst>(I).isConditional()) ||
                     isa<SwitchInst>(I)) {
            SMCalls.insert({&I, LA.getLiveVarsBefore(&I)});
          }
        }
      }
    }

    Function *SMFunc = Intrinsic::getDeclaration(&M, SMFuncID);
    assert(SMFunc != nullptr);

    uint64_t Count = StackmapIDStart;
    Value *Shadow = ConstantInt::get(Type::getInt32Ty(Context), 0);
    for (auto It : SMCalls) {
      Instruction *I = cast<Instruction>(It.first);
      const std::vector<Value *> L = It.second;

      IRBuilder<> Bldr(I);
      Value *SMID = ConstantInt::get(Type::getInt64Ty(Context), Count);
      std::vector<Value *> Args = {SMID, Shadow};
      for (Value *A : L)
        Args.push_back(A);

      if (isa<CallInst>(I)) {
        // Insert the stackmap call after (not before) the call instruction, so
        // the offset of the stackmap entry will record the instruction after
        // the call, which is where we want to continue after deoptimisation.
        Bldr.SetInsertPoint(I->getNextNode());
      }
      Bldr.CreateCall(SMFunc->getFunctionType(), SMFunc,
                      ArrayRef<Value *>(Args));
      Count++;
    }

#ifndef NDEBUG
    // Our pass runs after LLVM normally does its verify pass. In debug builds
    // we run it again to check that our pass is generating valid IR.
    if (verifyModule(M, &errs())) {
      Context.emitError("Stackmap insertion pass generated invalid IR!");
      return false;
    }
#endif
    return true;
  }
};
} // namespace

char YkStackmaps::ID = 0;
INITIALIZE_PASS(YkStackmaps, DEBUG_TYPE, "yk stackmaps", false, false)

/**
 * @brief Creates a new YkStackmaps pass.
 *
 * @param stackmapIDStart The first stackmap ID available for use by this pass.
 *                        Defaults to 1 if not specified.
 * @return ModulePass* A pointer to the newly created YkStackmaps pass.
 */
ModulePass *llvm::createYkStackmapsPass(uint64_t stackmapIDStart = 1) {
  return new YkStackmaps(stackmapIDStart);
}
