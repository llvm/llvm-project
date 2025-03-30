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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

enum class OutputCostKind {
  RecipThroughput,
  Latency,
  CodeSize,
  SizeAndLatency,
  All,
};

static cl::opt<OutputCostKind> CostKind(
    "cost-kind", cl::desc("Target cost kind"),
    cl::init(OutputCostKind::RecipThroughput),
    cl::values(clEnumValN(OutputCostKind::RecipThroughput, "throughput",
                          "Reciprocal throughput"),
               clEnumValN(OutputCostKind::Latency, "latency",
                          "Instruction latency"),
               clEnumValN(OutputCostKind::CodeSize, "code-size", "Code size"),
               clEnumValN(OutputCostKind::SizeAndLatency, "size-latency",
                          "Code size and latency"),
               clEnumValN(OutputCostKind::All, "all", "Print all cost kinds")));

enum class IntrinsicCostStrategy {
  InstructionCost,
  IntrinsicCost,
  TypeBasedIntrinsicCost,
};

static cl::opt<IntrinsicCostStrategy> IntrinsicCost(
    "intrinsic-cost-strategy",
    cl::desc("Costing strategy for intrinsic instructions"),
    cl::init(IntrinsicCostStrategy::InstructionCost),
    cl::values(
        clEnumValN(IntrinsicCostStrategy::InstructionCost, "instruction-cost",
                   "Use TargetTransformInfo::getInstructionCost"),
        clEnumValN(IntrinsicCostStrategy::IntrinsicCost, "intrinsic-cost",
                   "Use TargetTransformInfo::getIntrinsicInstrCost"),
        clEnumValN(
            IntrinsicCostStrategy::TypeBasedIntrinsicCost,
            "type-based-intrinsic-cost",
            "Calculate the intrinsic cost based only on argument types")));

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

static InstructionCost getCost(Instruction &Inst, TTI::TargetCostKind CostKind,
                               TargetTransformInfo &TTI,
                               TargetLibraryInfo &TLI) {
  auto *II = dyn_cast<IntrinsicInst>(&Inst);
  if (II && IntrinsicCost != IntrinsicCostStrategy::InstructionCost) {
    IntrinsicCostAttributes ICA(
        II->getIntrinsicID(), *II, InstructionCost::getInvalid(),
        /*TypeBasedOnly=*/IntrinsicCost ==
            IntrinsicCostStrategy::TypeBasedIntrinsicCost,
        &TLI);
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
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  OS << "Printing analysis 'Cost Model Analysis' for function '" << F.getName() << "':\n";
  for (BasicBlock &B : F) {
    for (Instruction &Inst : B) {
      OS << "Cost Model: ";
      if (CostKind == OutputCostKind::All) {
        OS << "Found costs of ";
        InstructionCost RThru =
            getCost(Inst, TTI::TCK_RecipThroughput, TTI, TLI);
        InstructionCost CodeSize = getCost(Inst, TTI::TCK_CodeSize, TTI, TLI);
        InstructionCost Lat = getCost(Inst, TTI::TCK_Latency, TTI, TLI);
        InstructionCost SizeLat =
            getCost(Inst, TTI::TCK_SizeAndLatency, TTI, TLI);
        if (RThru == CodeSize && RThru == Lat && RThru == SizeLat)
          OS << RThru;
        else
          OS << "RThru:" << RThru << " CodeSize:" << CodeSize << " Lat:" << Lat
             << " SizeLat:" << SizeLat;
        OS << " for: " << Inst << "\n";
      } else {
        InstructionCost Cost =
            getCost(Inst, OutputCostKindToTargetCostKind(CostKind), TTI, TLI);
        if (auto CostVal = Cost.getValue())
          OS << "Found an estimated cost of " << *CostVal;
        else
          OS << "Invalid cost";
        OS << " for instruction: " << Inst << "\n";
      }
    }
  }
  return PreservedAnalyses::all();
}
