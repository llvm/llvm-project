//===-- EJitOptimizer.cpp - JIT Optimization Pipeline ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

using namespace llvm;
using namespace llvm::ejit;

EJitOptimizer::EJitOptimizer(PeriodArrayRegistry &reg)
    : registry_(reg) {
  (void)registry_; // Used by PASS6 integration
}

void EJitOptimizer::preReplacePeriodIndices(
    Module &M, const SpecializationContext &ctx) {
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;

    // Find ejit_period_arr_ind entries
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

      // Find matching dimension in the specialization context
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
  FunctionAnalysisManager FAM;

  // Register required analyses
  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);

  for (Function &F : M.functions()) {
    if (!F.isDeclaration())
      FPM.run(F, FAM);
  }
}

void EJitOptimizer::runInline(Module &M) {
  ModulePassManager MPM;
  MPM.addPass(AlwaysInlinerPass());

  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);

  MPM.run(M, MAM);
}

void EJitOptimizer::runOptimizationPipeline(Module &M,
                                            OptimizationLevel level) {
  FunctionPassManager FPM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);
  PB.registerModuleAnalyses(MAM);

  // L1: SCCP + ADCE + SimplifyCFG
  FPM.addPass(SCCPPass());
  FPM.addPass(ADCEPass());
  FPM.addPass(SimplifyCFGPass());

  for (Function &F : M.functions()) {
    if (!F.isDeclaration())
      FPM.run(F, FAM);
  }

  if (static_cast<int>(level) >= 2) {
    // L2: Second inline + SimplifyCFG
    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass());
    MPM.run(M, MAM);

    for (Function &F : M.functions()) {
      if (!F.isDeclaration()) {
        FunctionPassManager FPM2;
        FPM2.addPass(SimplifyCFGPass());
        FPM2.run(F, FAM);
      }
    }
  }

  if (static_cast<int>(level) >= 3) {
    // L3: Additional cleanup via SimplifyCFG + Mem2Reg
    for (Function &F : M.functions()) {
      if (!F.isDeclaration()) {
        FunctionPassManager FPM3;
        FPM3.addPass(PromotePass());
        FPM3.addPass(SimplifyCFGPass());
        FPM3.run(F, FAM);
      }
    }
  }
}
