#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
struct WebAssemblyReduceToAnyAllTrueLegacy final : FunctionPass {
  static char ID;

  WebAssemblyTargetMachine &TM;
  Module *CachedModule = nullptr;
  bool ModuleHasInterestingIntrinsics = false;

  WebAssemblyReduceToAnyAllTrueLegacy(WebAssemblyTargetMachine &TM)
      : FunctionPass(ID), TM(TM) {}

  StringRef getPassName() const override {
    return "WebAssembly convert reduce to any_true/all_true";
  }

  bool runOnFunction(Function &F) override;
};
} // end anonymous namespace

char WebAssemblyReduceToAnyAllTrueLegacy::ID = 0;

static bool hasInterestingIntrinsics(Module &M, Module *&CachedModule,
                                     bool &ModuleHasInterestingIntrinsics) {
  if (CachedModule == &M)
    return ModuleHasInterestingIntrinsics;

  CachedModule = &M;
  ModuleHasInterestingIntrinsics = false;

  for (const Function &Fn : M.functions()) {
    switch (Fn.getIntrinsicID()) {
    case Intrinsic::vector_reduce_or:
    case Intrinsic::vector_reduce_and:
      ModuleHasInterestingIntrinsics = true;
      return true;
    default:
      break;
    }
  }

  return false;
}

static bool reduceToAnyAllTrue(Function &F, WebAssemblyTargetMachine &TM,
                               Module *&CachedModule,
                               bool &ModuleHasInterestingIntrinsics) {
  if (!TM.getSubtarget<WebAssemblySubtarget>(F).hasSIMD128())
    return false;

  if (!hasInterestingIntrinsics(*F.getParent(), CachedModule,
                                ModuleHasInterestingIntrinsics))
    return false;

  bool Changed = false;

  for (auto &BB : F) {
    for (auto It = BB.begin(), E = BB.end(); It != E;) {
      Instruction *I = &*It++;
      auto *Cmp = dyn_cast<ICmpInst>(I);
      if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_NE)
        continue;

      Value *Reduce = nullptr;
      if (!match(Cmp, m_ICmp(m_Value(Reduce), m_ZeroInt())))
        continue;

      auto *II = dyn_cast<IntrinsicInst>(Reduce);
      if (!II || !II->hasOneUse())
        continue;

      IRBuilder<> B(Cmp);
      Value *Vec = II->getArgOperand(0);
      Module *M = F.getParent();

      auto makeIntrinsic = [&](Intrinsic::ID ID, Value *Arg) {
        Function *Fn =
            Intrinsic::getOrInsertDeclaration(M, ID, {Arg->getType()});
        return B.CreateCall(Fn, {Arg});
      };

      Value *New = nullptr;

      switch (II->getIntrinsicID()) {
      case Intrinsic::vector_reduce_or: {
        // reduce.or(X) != 0  -> anytrue(X)
        Value *Any = makeIntrinsic(Intrinsic::wasm_anytrue, Vec);
        New = B.CreateICmpNE(Any, ConstantInt::get(Any->getType(), 0));
        break;
      }

      case Intrinsic::vector_reduce_and: {
        // reduce.and(zext (icmp ne X, zeroinitializer)) != 0  -> alltrue(X)

        // Match: zext (icmp ne X, 0) from <N x i1> to <N x iX>
        CmpPredicate Pred;
        Value *LHS = nullptr;
        if (!match(Vec, m_ZExt(m_c_ICmp(Pred, m_Value(LHS), m_Zero()))))
          continue;
        if (Pred != ICmpInst::ICMP_NE)
          continue;

        Value *All = makeIntrinsic(Intrinsic::wasm_alltrue, LHS);
        New = B.CreateICmpNE(All, ConstantInt::get(All->getType(), 0));
        break;
      }

      default:
        continue;
      }

      Cmp->replaceAllUsesWith(New);
      Cmp->eraseFromParent();

      if (II->use_empty())
        II->eraseFromParent();

      Changed = true;
    }
  }

  return Changed;
}

bool WebAssemblyReduceToAnyAllTrueLegacy::runOnFunction(Function &F) {
  return reduceToAnyAllTrue(F, TM, CachedModule,
                            ModuleHasInterestingIntrinsics);
}

PreservedAnalyses
WebAssemblyReduceToAnyAllTruePass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  return reduceToAnyAllTrue(F, TM, CachedModule, ModuleHasInterestingIntrinsics)
             ? PreservedAnalyses::none().preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}

FunctionPass *llvm::createWebAssemblyReduceToAnyAllTrueLegacyPass(
    WebAssemblyTargetMachine &TM) {
  return new WebAssemblyReduceToAnyAllTrueLegacy(TM);
}
