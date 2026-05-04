//===- StructuralHash.cpp - Function Hash Printing ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StructuralHashPrinterPass which is used to show
// the structural hash of all functions in a module and the module itself.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StructuralHash.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/Format.h"

using namespace llvm;

PreservedAnalyses StructuralHashPrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  OS << "Module Hash: "
     << format("%016" PRIx64,
               StructuralHash(M, Options != StructuralHashOptions::None))
     << "\n";
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (Options == StructuralHashOptions::CallTargetIgnored) {
      auto IgnoreOp = [&](const Instruction *I, unsigned OpndIdx) {
        return I->getOpcode() == Instruction::Call &&
               isa<Constant>(I->getOperand(OpndIdx));
      };
      auto FuncHashInfo = StructuralHashWithDifferences(F, IgnoreOp);
      OS << "Function " << F.getName()
         << " Hash: " << format("%016" PRIx64, FuncHashInfo.FunctionHash)
         << "\n";
      for (auto &[IndexPair, OpndHash] : *FuncHashInfo.IndexOperandHashMap) {
        auto [InstIndex, OpndIndex] = IndexPair;
        OS << "\tIgnored Operand Hash: " << format("%016" PRIx64, OpndHash)
           << " at (" << InstIndex << "," << OpndIndex << ")\n";
      }
    } else {
      OS << "Function " << F.getName() << " Hash: "
         << format(
                "%016" PRIx64,
                StructuralHash(F, Options == StructuralHashOptions::Detailed))
         << "\n";
    }
  }
  return PreservedAnalyses::all();
}
