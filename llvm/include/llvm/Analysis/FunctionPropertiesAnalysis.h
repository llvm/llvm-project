//=- FunctionPropertiesAnalysis.h - Function Properties Analysis --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FUNCTIONPROPERTIESANALYSIS_H
#define LLVM_ANALYSIS_FUNCTIONPROPERTIESANALYSIS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class CallBase;
class DominatorTree;
class Function;
class LoopInfo;

class FunctionPropertiesInfo {
  friend class FunctionPropertiesUpdater;
  void updateForBB(const BasicBlock &BB, int64_t Direction);
  void updateAggregateStats(const Function &F, const LoopInfo &LI);
  void reIncludeBB(const BasicBlock &BB);

  ir2vec::Embedding FunctionEmbedding = ir2vec::Embedding(0.0);
  const ir2vec::Vocabulary *IR2VecVocab = nullptr;

public:
  LLVM_ABI static FunctionPropertiesInfo
  getFunctionPropertiesInfo(const Function &F, const DominatorTree &DT,
                            const LoopInfo &LI,
                            const ir2vec::Vocabulary *Vocabulary);

  LLVM_ABI static FunctionPropertiesInfo
  getFunctionPropertiesInfo(Function &F, FunctionAnalysisManager &FAM);

  LLVM_ABI bool operator==(const FunctionPropertiesInfo &FPI) const;

  bool operator!=(const FunctionPropertiesInfo &FPI) const {
    return !(*this == FPI);
  }

  LLVM_ABI void print(raw_ostream &OS) const;

  /// Number of basic blocks
  int64_t BasicBlockCount = 0;

  /// Number of blocks reached from a conditional instruction, or that are
  /// 'cases' of a SwitchInstr.
  // FIXME: We may want to replace this with a more meaningful metric, like
  // number of conditionally executed blocks:
  // 'if (a) s();' would be counted here as 2 blocks, just like
  // 'if (a) s(); else s2(); s3();' would.
  int64_t BlocksReachedFromConditionalInstruction = 0;

  /// Number of uses of this function, plus 1 if the function is callable
  /// outside the module.
  int64_t Uses = 0;

  /// Number of direct calls made from this function to other functions
  /// defined in this module.
  int64_t DirectCallsToDefinedFunctions = 0;

  // Load Instruction Count
  int64_t LoadInstCount = 0;

  // Store Instruction Count
  int64_t StoreInstCount = 0;

  // Maximum Loop Depth in the Function
  int64_t MaxLoopDepth = 0;

  // Number of Top Level Loops in the Function
  int64_t TopLevelLoopCount = 0;

  // All non-debug instructions
  int64_t TotalInstructionCount = 0;

  // Basic blocks grouped by number of successors.
  int64_t BasicBlocksWithSingleSuccessor = 0;
  int64_t BasicBlocksWithTwoSuccessors = 0;
  int64_t BasicBlocksWithMoreThanTwoSuccessors = 0;

  // Basic blocks grouped by number of predecessors.
  int64_t BasicBlocksWithSinglePredecessor = 0;
  int64_t BasicBlocksWithTwoPredecessors = 0;
  int64_t BasicBlocksWithMoreThanTwoPredecessors = 0;

  // Basic blocks grouped by size as determined by the number of non-debug
  // instructions that they contain.
  int64_t BigBasicBlocks = 0;
  int64_t MediumBasicBlocks = 0;
  int64_t SmallBasicBlocks = 0;

  // The number of cast instructions inside the function.
  int64_t CastInstructionCount = 0;

  // The number of floating point instructions inside the function.
  int64_t FloatingPointInstructionCount = 0;

  // The number of integer instructions inside the function.
  int64_t IntegerInstructionCount = 0;

  // Operand type couns
  int64_t ConstantIntOperandCount = 0;
  int64_t ConstantFPOperandCount = 0;
  int64_t ConstantOperandCount = 0;
  int64_t InstructionOperandCount = 0;
  int64_t BasicBlockOperandCount = 0;
  int64_t GlobalValueOperandCount = 0;
  int64_t InlineAsmOperandCount = 0;
  int64_t ArgumentOperandCount = 0;
  int64_t UnknownOperandCount = 0;

  // Additional CFG Properties
  int64_t CriticalEdgeCount = 0;
  int64_t ControlFlowEdgeCount = 0;
  int64_t UnconditionalBranchCount = 0;
  int64_t ConditionalBranchCount = 0;
  int64_t BranchInstructionCount = 0;
  int64_t BranchSuccessorCount = 0;
  int64_t SwitchInstructionCount = 0;
  int64_t SwitchSuccessorCount = 0;

  // Call related instructions
  int64_t IntrinsicCount = 0;
  int64_t DirectCallCount = 0;
  int64_t IndirectCallCount = 0;
  int64_t CallReturnsIntegerCount = 0;
  int64_t CallReturnsFloatCount = 0;
  int64_t CallReturnsPointerCount = 0;
  int64_t CallReturnsVectorIntCount = 0;
  int64_t CallReturnsVectorFloatCount = 0;
  int64_t CallReturnsVectorPointerCount = 0;
  int64_t CallWithManyArgumentsCount = 0;
  int64_t CallWithPointerArgumentCount = 0;

  const ir2vec::Embedding &getFunctionEmbedding() const {
    return FunctionEmbedding;
  }

  const ir2vec::Vocabulary *getIR2VecVocab() const { return IR2VecVocab; }

  // Helper intended to be useful for unittests
  void setFunctionEmbeddingForTest(const ir2vec::Embedding &Embedding) {
    FunctionEmbedding = Embedding;
  }
};

// Analysis pass
class FunctionPropertiesAnalysis
    : public AnalysisInfoMixin<FunctionPropertiesAnalysis> {

public:
  LLVM_ABI static AnalysisKey Key;

  using Result = const FunctionPropertiesInfo;

  LLVM_ABI FunctionPropertiesInfo run(Function &F,
                                      FunctionAnalysisManager &FAM);
};

/// Printer pass for the FunctionPropertiesAnalysis results.
class FunctionPropertiesPrinterPass
    : public PassInfoMixin<FunctionPropertiesPrinterPass> {
  raw_ostream &OS;

public:
  explicit FunctionPropertiesPrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// Statistics pass for the FunctionPropertiesAnalysis results.
struct FunctionPropertiesStatisticsPass
    : PassInfoMixin<FunctionPropertiesStatisticsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

/// Correctly update FunctionPropertiesInfo post-inlining. A
/// FunctionPropertiesUpdater keeps the state necessary for tracking the changes
/// llvm::InlineFunction makes. The idea is that inlining will at most modify
/// a few BBs of the Caller (maybe the entry BB and definitely the callsite BB)
/// and potentially affect exception handling BBs in the case of invoke
/// inlining.
class FunctionPropertiesUpdater {
public:
  LLVM_ABI FunctionPropertiesUpdater(FunctionPropertiesInfo &FPI, CallBase &CB);

  LLVM_ABI void finish(FunctionAnalysisManager &FAM) const;
  bool finishAndTest(FunctionAnalysisManager &FAM) const {
    finish(FAM);
    return isUpdateValid(Caller, FPI, FAM);
  }

private:
  FunctionPropertiesInfo &FPI;
  BasicBlock &CallSiteBB;
  Function &Caller;

  LLVM_ABI static bool isUpdateValid(Function &F,
                                     const FunctionPropertiesInfo &FPI,
                                     FunctionAnalysisManager &FAM);

  DominatorTree &getUpdatedDominatorTree(FunctionAnalysisManager &FAM) const;

  DenseSet<const BasicBlock *> Successors;
  DenseSet<const BasicBlock *> CallUsers;

  // Edges we might potentially need to remove from the dominator tree.
  SmallVector<DominatorTree::UpdateType, 2> DomTreeUpdates;
};
} // namespace llvm
#endif // LLVM_ANALYSIS_FUNCTIONPROPERTIESANALYSIS_H
