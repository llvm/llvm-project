//===- SourceExpressionAnalysis.h - Mapping LLVM Values to Source Level Expression -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file defines the LoadStoreSourceExpression class related to analyzing
// and generating source-level expressions for LLVM values by utilising the
// debug metadata.
//
// This analysis is useful for understanding memory access patterns, aiding optimization decisions,
// and providing more informative optimization reports.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H
#define LLVM_ANALYSIS_SOURCEEXPRESSIONANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <optional>
#include <string_view>
using namespace llvm;

namespace llvm {

class LoadStoreSourceExpression {
public:
  // Constructor that takes a Function reference.
  LoadStoreSourceExpression(const Function &F) : F(F) {}

  // Print out the values currently in the cache.
  void print(raw_ostream &OS) const;

  // Query the SourceExpressionMap For a Value
  std::string getSourceExpressionForValue(Value *Key) const {
    auto It = SourceExpressionsMap.find(Key);
    if (It != SourceExpressionsMap.end()) {
      return It->second;
    }

    return "Complex Expression or load and store get optimized out";
  }

  // Get the expression string corresponding to an opcode.
  std::string getExpressionFromOpcode(unsigned Opcode);

  // Process a StoreInst instruction and return its source-level expression.
  void processStoreInst(StoreInst *I);

  // Process a LoadInst instruction and update the sourceExpressionsMap.
  void processLoadInst(LoadInst *I);

private:
  // This map stores the source-level expressions for LLVM values.
  // The expressions are represented as strings and are associated with the
  // corresponding values. It is used to cache and retrieve source expressions
  // during the generation process.
  std::map<Value *, std::string> SourceExpressionsMap;

  // Process Debug Metadata associated with a stored value
  DILocalVariable *processDbgMetadata(Value *StoredValue);

  const Function &F;

  // Get the source-level expression for an LLVM value.
  std::string getSourceExpression(Value *Operand);

  // Get the source-level expression for a GetElementPtr instruction.
  std::string
  getSourceExpressionForGetElementPtr(GetElementPtrInst *GepInstruction);

  // Get the source-level expression for a BinaryOperator.
  std::string getSourceExpressionForBinaryOperator(BinaryOperator *BinaryOp,
                                                   Value *Operand);

  // Get the source-level expression for a SExtInst.
  std::string getSourceExpressionForSExtInst(SExtInst *SextInstruction);
};

class SourceExpressionAnalysis
    : public AnalysisInfoMixin<SourceExpressionAnalysis> {
  friend AnalysisInfoMixin<SourceExpressionAnalysis>;
  static AnalysisKey Key;

public:
  using Result = LoadStoreSourceExpression;
  Result run(Function &F, FunctionAnalysisManager &);
};

class SourceExpressionAnalysisPrinterPass
    : public PassInfoMixin<SourceExpressionAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit SourceExpressionAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif
