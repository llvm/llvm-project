//===- llvm/CodeGen/MachineFunctionAnalysis.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachineFunctionAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTIONANALYSIS
#define LLVM_CODEGEN_MACHINEFUNCTIONANALYSIS

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineFunction;
class TargetMachine;

/// This analysis create MachineFunction for given Function.
/// To release the MachineFunction, users should invalidate it explicitly.
class MachineFunctionAnalysis
    : public AnalysisInfoMixin<MachineFunctionAnalysis> {
  friend AnalysisInfoMixin<MachineFunctionAnalysis>;

  LLVM_ABI static AnalysisKey Key;

  const TargetMachine *TM;

public:
  class Result {
    std::unique_ptr<MachineFunction> MF;

  public:
    Result(std::unique_ptr<MachineFunction> MF);
    MachineFunction &getMF() { return *MF; };
    LLVM_ABI bool invalidate(Function &, const PreservedAnalyses &PA,
                             FunctionAnalysisManager::Invalidator &);
  };

  MachineFunctionAnalysis(const TargetMachine *TM) : TM(TM) {};
  LLVM_ABI Result run(Function &F, FunctionAnalysisManager &FAM);
};

class FreeMachineFunctionPass : public PassInfoMixin<FreeMachineFunctionPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MachineFunctionAnalysis
