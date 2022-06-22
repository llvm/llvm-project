//
// Created by tanmay on 6/10/22.
//

#ifndef LLVM_ACINSTRUMENTATION_H
#define LLVM_ACINSTRUMENTATION_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/ErrorAnalysis/ComputationGraph.h"

using namespace llvm;

namespace atomiccondition {

class ACInstrumentation {
private:
//  ComputationGraph CG;
  Function *FunctionToInstrument;

  Function *ACInitFunction;

  Function *ACfp32UnaryFunction;
  Function *ACfp64UnaryFunction;
  Function *ACfp32BinaryFunction;
  Function *ACfp64BinaryFunction;

//  Function *ACStoreFunction;
//  Function *ACAnalysisFunction;

public:
  static int VarCounter;
  static int NodeCounter;

  ACInstrumentation(Function *F);

  bool instrumentUnaryCall(Instruction* BaseInstruction,
                           long int *NumInstrumentedInstructions);
  bool instrumentBinaryCall(Instruction* BaseInstruction,
                            long int *NumInstrumentedInstructions);

  bool instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  bool instrumentMainFunction(Function *F);

  //// Helper Functions
  /// Instruction based functions
  // Categorized by operations
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
