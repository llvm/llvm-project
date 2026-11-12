//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CostModel.h"
#include "AnalysisOptions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

static InstructionCost getCost(const AnalysisOptions &Opts, Instruction &Inst,
                               TTI::TargetCostKind CostKind,
                               TargetTransformInfo &TTI) {
  auto *II = dyn_cast<IntrinsicInst>(&Inst);
  if (II &&
      Opts.intrinsic_cost_strategy != IntrinsicCostStrategy::InstructionCost) {
    IntrinsicCostAttributes ICA(
        II->getIntrinsicID(), *II, InstructionCost::getInvalid(),
        /*TypeBasedOnly=*/Opts.intrinsic_cost_strategy ==
            IntrinsicCostStrategy::TypeBasedIntrinsicCost);
    return TTI.getIntrinsicInstrCost(ICA, CostKind);
  }

  return TTI.getInstructionCost(&Inst, CostKind);
}

static TTI::TargetCostKind
OutputCostKindToTargetCostKind(OutputCostKind CostKind) {
  switch (CostKind) {
  case OutputCostKind::RecipThroughput:
    return TTI::TCK_RecipThroughput;
  case OutputCostKind::Latency:
    return TTI::TCK_Latency;
  case OutputCostKind::CodeSize:
    return TTI::TCK_CodeSize;
  case OutputCostKind::SizeAndLatency:
    return TTI::TCK_SizeAndLatency;
  default:
    llvm_unreachable("Unexpected OutputCostKind!");
  };
}

PreservedAnalyses CostModelPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  const AnalysisOptions &Opts = getAnalysisOptions(F);
  OS << "Printing analysis 'Cost Model Analysis' for function '" << F.getName() << "':\n";
  for (BasicBlock &B : F) {
    for (Instruction &Inst : B) {
      OS << "Cost Model: ";
      if (Opts.cost_kind == OutputCostKind::All) {
        OS << "Found costs of ";
        InstructionCost RThru =
            getCost(Opts, Inst, TTI::TCK_RecipThroughput, TTI);
        InstructionCost CodeSize = getCost(Opts, Inst, TTI::TCK_CodeSize, TTI);
        InstructionCost Lat = getCost(Opts, Inst, TTI::TCK_Latency, TTI);
        InstructionCost SizeLat =
            getCost(Opts, Inst, TTI::TCK_SizeAndLatency, TTI);
        if (RThru == CodeSize && RThru == Lat && RThru == SizeLat)
          OS << RThru;
        else
          OS << "RThru:" << RThru << " CodeSize:" << CodeSize << " Lat:" << Lat
             << " SizeLat:" << SizeLat;
        OS << " for: " << Inst << "\n";
      } else {
        InstructionCost Cost = getCost(
            Opts, Inst, OutputCostKindToTargetCostKind(Opts.cost_kind), TTI);
        if (Cost.isValid())
          OS << "Found an estimated cost of " << Cost;
        else
          OS << "Invalid cost";
        OS << " for instruction: " << Inst << "\n";
      }
    }
  }
  return PreservedAnalyses::all();
}
