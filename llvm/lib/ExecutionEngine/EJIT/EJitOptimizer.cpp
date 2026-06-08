//===-- EJitOptimizer.cpp - JIT Optimization Pipeline ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/ExecutionEngine/EJIT/EJitPassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-optimizer"

EJitOptimizer::EJitOptimizer(PeriodArrayRegistry &reg)
    : registry_(reg) {
  EJitPassBuilder::registerFunctionAnalyses(FAM_);
  EJitPassBuilder::registerLoopAnalyses(LAM_);
  EJitPassBuilder::registerCGSCCAnalyses(CGAM_);
  EJitPassBuilder::registerModuleAnalyses(MAM_);
  EJitPassBuilder::crossRegisterProxies(LAM_, FAM_, CGAM_, MAM_);
}

void EJitOptimizer::clearAnalyses() {
  FAM_.clear();
  LAM_.clear();
  CGAM_.clear();
  MAM_.clear();
}

void EJitOptimizer::runPipeline(Module &M,
                                const SpecializationContext &ctx) {
  // 1. Parameter substitution: replace ejit_period_arr_ind args with constants
  preReplacePeriodIndices(M, ctx);

  // 2. InstCombine: fold constant GEP chains from substituted params
  //    so StructFieldPass can compute correct byte offsets.
  runInstCombine(M);

  // 3. StructFieldPass: replace may_const loads with runtime constants
  runStructFieldPass(M);

  // 4. Core optimization at the configured level
  runOptimizationPipeline(M, ctx.optLevel);
}

void EJitOptimizer::preReplacePeriodIndices(
    Module &M, const SpecializationContext &ctx) {
  LLVM_DEBUG(dbgs() << "ejit-optimizer: preReplacePeriodIndices, "
                    << ctx.dimensions.size() << " dim(s)\n");
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;

    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 3)
        continue;

      auto *Tag = dyn_cast<MDString>(Sub->getOperand(0));
      if (!Tag || Tag->getString() != TAG_EJIT_PERIOD_ARR_IND)
        continue;

      auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
      auto *IdxC = mdconst::dyn_extract<ConstantInt>(Sub->getOperand(2));
      if (!PN || !IdxC)
        continue;

      unsigned argIdx = static_cast<unsigned>(IdxC->getZExtValue());
      if (argIdx >= F.arg_size())
        continue;

      for (auto &dim : ctx.dimensions) {
        if (dim.periodName == PN->getString()) {
          Argument *arg = F.getArg(argIdx);
          arg->replaceAllUsesWith(
              ConstantInt::get(arg->getType(), dim.cellIdx));
          break;
        }
      }
    }
  }
}

void EJitOptimizer::runInstCombine(Module &M) {
  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());

  for (Function &F : M.functions())
    if (!F.isDeclaration())
      FPM.run(F, FAM_);
}

void EJitOptimizer::runStructFieldPass(Module &M) {
  EJitStructFieldPass structField(registry_);
  for (Function &F : M.functions())
    if (!F.isDeclaration())
      structField.run(F, FAM_);
}

void EJitOptimizer::runOptimizationPipeline(Module &M,
                                            OptimizationLevel level) {
  // L1: SCCP + ADCE + SimplifyCFG — constant propagation, dead code
  // elimination, and CFG cleanup. Captures the vast majority of EJIT
  // performance gains (may_const load → constant → branch folding).
  {
    FunctionPassManager FPM;
    FPM.addPass(SCCPPass());
    FPM.addPass(ADCEPass());
    FPM.addPass(SimplifyCFGPass());

    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM.run(F, FAM_);
  }

  // L2: Inline always_inline helpers, then clean up + re-run
  // StructFieldPass for loads exposed by inlining.
  if (static_cast<int>(level) >= 2) {
    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass());
    MPM.run(M, MAM_);

    {
      FunctionPassManager FPM;
      FPM.addPass(SimplifyCFGPass());
      for (Function &F : M.functions())
        if (!F.isDeclaration())
          FPM.run(F, FAM_);
    }

    runStructFieldPass(M);
    runInstCombine(M);
  }

  // L3: Unroll small loops with may_const-dependent bodies, then clean up
  // + re-run StructFieldPass for loads exposed by unrolling.
  if (static_cast<int>(level) >= 3) {
    FunctionPassManager FPM3;
    FPM3.addPass(LoopSimplifyPass());
    {
      LoopPassManager LPM;
      LPM.addPass(LoopFullUnrollPass());
      FPM3.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    }
    FPM3.addPass(PromotePass());
    FPM3.addPass(SimplifyCFGPass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM3.run(F, FAM_);

    runStructFieldPass(M);
    runInstCombine(M);
  }
}
