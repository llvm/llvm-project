//
// Created by tanmay on 6/10/22.
//

#ifndef LLVM_ACINSTRUMENTATION_H
#define LLVM_ACINSTRUMENTATION_H

#include <string>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"

using namespace llvm;

namespace atomiccondition {

class ACInstrumentation {
private:
  // Function to Instrument
  int Evaluations;
  GlobalVariable *ExecutionCounter;
  GlobalVariable *FunctionInstanceCounter;
  Function *FunctionToInstrument;

  Function *AFInitFunction;

  Function *ACComputingFunction;
  Function *AFComputingFunction;

  Function *ACStoreFunction;
  Function *AFStoreFunction;

  Function *AFStoreInFile;

  // Instruction to AC/AF Maps
  ValueMap<Value*, Value*> InstructionACMap;
  ValueMap<Value*, Value*> InstructionAFMap;

  // Result Functions
  Function *AFPrintTopAmplificationPaths;
  Function *AFPrintStatistics;

  // Additional Utilities

public:
  Value *EmptyValuePointer;

  ACInstrumentation(Function *F, std::string FunctionToAnalyze, int Evaluations);

  void instrumentCallsToAnalyzeInstruction(Instruction *BaseInstruction,
                                           BasicBlock::iterator *InstructionIterator,
                                           long int *NumInstrumentedInstructions);
  void instrumentCallsForACComputation(Instruction *BaseInstruction,
                                       BasicBlock::iterator *InstructionIterator,
                                       long int *NumInstrumentedInstructions,
                                       int FunctionType);
  void instrumentCallsForAFComputation(Instruction *BaseInstruction,
                                       BasicBlock::iterator *InstructionIterator,
                                       long int *NumInstrumentedInstructions,
                                       int FunctionType);

  Value *instrumentPhiNodeForAF(Value *OriginalPHI,
                                long *NumInstrumentedInstructions);
  Value *instrumentSelectForAF(Value *OriginalSelInstr,
                                BasicBlock::iterator *InstructionIterator,
                                long *NumInstrumentedInstructions);

  void instrumentForMarkedVariable(Value *BaseInstruction,
                                   BasicBlock::iterator *InstructionIterator,
                                   long int *NumInstrumentedInstructions);
  void instrumentMarkedFunction(BasicBlock::iterator *InstructionIterator,
                                long int *NumInstrumentedInstructions);

  void instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  void instrumentMainFunction(Function *F);
  void instrumentFunctionInitializationInsts(Function *F);

  //// Helper Functions
  Value *createArrayInIR(std::vector<Value*> ArrayOfValues,
                         IRBuilder<> *InstructionBuilder,
                         BasicBlock::iterator *InstructionIterator);

  /// Instruction based functions
  // Categorized by instructions
  static bool isPhiNode(const Instruction *Inst);
  static bool isMemoryLoadInstruction(const Instruction *Inst);
  static bool isIntegerToFloatCastInstruction(const Instruction *Inst);
  static bool isFloatToFloatCastInstruction(const Instruction *Inst);
  static bool isUnaryInstruction(const Instruction *Inst);
  static bool isBinaryInstruction(const Instruction *Inst);
  static bool isTernaryInstruction(const Instruction *Inst);
  static bool isNonACInstrinsicFunction(const Instruction *Inst);
  static bool isNonACFloatPointInstruction(const Instruction *Inst);

  // Categorized by Data Type
//  static bool isFPInstruction(const Instruction *Inst);
  static bool isSingleFPInstruction(const Instruction *Inst);
  static bool isDoubleFPInstruction(const Instruction *Inst);

  /// Function based functions
  static bool isUnwantedFunction(const Function *Func);

  // Other Utilities
  void mapFloatCastToAFValue(Instruction *Inst,
                             long int *NumInstrumentedInstructions);
};

} // namespace atomiccondition

#endif // LLVM_ACINSTRUMENTATION_H
