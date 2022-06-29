//
// Created by tanmay on 6/10/22.
//

#ifndef LLVM_ACINSTRUMENTATION_H
#define LLVM_ACINSTRUMENTATION_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/ErrorAnalysis/ComputationGraph.h"

using namespace llvm;

namespace atomiccondition {

class ACInstrumentation {
private:
//  ComputationGraph CG;
  Function *FunctionToInstrument;

  Function *ACInitFunction;
  Function *CGInitFunction;

  Function *ACfp32UnaryFunction;
  Function *ACfp64UnaryFunction;
  Function *ACfp32BinaryFunction;
  Function *ACfp64BinaryFunction;

  Function *CGCreateNode;

  Function *ACStoreFunction;
  Function *CGStoreFunction;
  Function *ACAnalysisFunction;

  // Additional Utilities
  Function *CGDotGraphFunction;

public:
  static int VarCounter;

  ACInstrumentation(Function *F);

  bool instrumentCallsForMemoryLoadOperation(Instruction* BaseInstruction,
                                             long int *NumInstrumentedInstructions);
  bool instrumentCallsForUnaryOperation(Instruction* BaseInstruction,
                           long int *NumInstrumentedInstructions);
  bool instrumentCallsForBinaryOperation(Instruction* BaseInstruction,
                            long int *NumInstrumentedInstructions);

  bool instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  bool instrumentMainFunction(Function *F);

  //// Helper Functions
  /// Instruction based functions
  // Categorized by operations
  static bool isMemoryLoadOperation(const Instruction *Inst);
  static bool isUnaryOperation(const Instruction *Inst);
  static bool isBinaryOperation(const Instruction *Inst);

  // Categorized by Data Type
  static bool isFPOperation(const Instruction *Inst);
  static bool isSingleFPOperation(const Instruction *Inst);
  static bool isDoubleFPOperation(const Instruction *Inst);

  /// Function based functions
  static bool isUnwantedFunction(const Function *Func);
};

} // namespace atomiccondition

#endif // LLVM_ACINSTRUMENTATION_H
