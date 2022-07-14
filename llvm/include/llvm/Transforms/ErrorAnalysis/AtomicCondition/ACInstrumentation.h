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
  Function *FunctionToInstrument;

  Function *ACInitFunction;
  Function *CGInitFunction;
  Function *AFInitFunction;

  Function *ACfp32UnaryFunction;
  Function *ACfp64UnaryFunction;
  Function *ACfp32BinaryFunction;
  Function *ACfp64BinaryFunction;

  Function *CGRecordPHIInstruction;
  Function *CGRecordBasicBlock;
  Function *CGCreateNode;

  Function *ACStoreFunction;
  Function *CGStoreFunction;
  Function *AFStoreFunction;

  Function *AFfp32AnalysisFunction;
  Function *AFfp64AnalysisFunction;

  // Additional Utilities
  Function *CGDotGraphFunction;

public:
  ACInstrumentation(Function *F);

  void instrumentCallRecordingBasicBlock(BasicBlock* CurrentBB,
                                         long int *NumInstrumentedInstructions);
  void instrumentCallRecordingPHIInstructions(BasicBlock* CurrentBB,
                                              long int *NumInstrumentedInstructions);
  void instrumentCallsForMemoryLoadOperation(Instruction* BaseInstruction,
                                             long int *NumInstrumentedInstructions);
  void instrumentCallsForUnaryOperation(Instruction* BaseInstruction,
                                        long int *NumInstrumentedInstructions);
  void instrumentCallsForBinaryOperation(Instruction* BaseInstruction,
                                         long int *NumInstrumentedInstructions);
  void instrumentCallsForAFAnalysis(Instruction *BaseInstruction,
                                    Instruction *LocationToInstrument,
                                    long int *NumInstrumentedInstructions);

  void instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  void instrumentMainFunction(Function *F);

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

  // Utility Functions
  Value *createBBNameString(BasicBlock *BB);
  Value *createRegisterNameString(Instruction *Inst);
  Value *createInstructionString(Instruction *Inst);
};

} // namespace atomiccondition

#endif // LLVM_ACINSTRUMENTATION_H
