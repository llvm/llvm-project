//
// Created by tanmay on 6/10/22.
//

#ifndef LLVM_ACINSTRUMENTATION_H
#define LLVM_ACINSTRUMENTATION_H

#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace atomiccondition {

class ACInstrumentation {
private:
  Function *FunctionToInstrument;

  Function *ACInitFunction;

  Function *ACfp32UnaryFunction;
  Function *ACfp64UnaryFunction;
  Function *ACfp32BinaryFunction;
  Function *ACfp64BinaryFunction;

public:
  static int VarCounter;

  ACInstrumentation(Function *F);
  void instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  void instrumentMainFunction(Function *F);

  //// Helper Functions
  /// Instruction based functions
  // Categorized by operations
  static bool isUnaryOperation(const Instruction *Inst);
  static bool isBinaryOperation(const Instruction *Inst);
  static bool isCastOperation(const Instruction *Inst);

  // Categorized by Data Type
  static bool isFPOperation(const Instruction *Inst);
  static bool isSingleFPOperation(const Instruction *Inst);
  static bool isDoubleFPOperation(const Instruction *Inst);

  /// Function based functions
  static bool isUnwantedFunction(const Function *Func);
};

} // namespace atomiccondition

#endif // LLVM_ACINSTRUMENTATION_H
