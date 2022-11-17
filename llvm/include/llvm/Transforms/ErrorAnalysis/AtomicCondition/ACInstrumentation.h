//
// Created by tanmay on 6/10/22.
//

#ifndef LLVM_ACINSTRUMENTATION_H
#define LLVM_ACINSTRUMENTATION_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"

using namespace llvm;

namespace atomiccondition {

class ACInstrumentation {
private:
  Function *FunctionToInstrument;

  Function *ACInitFunction;
  Function *AFInitFunction;

  Function *ACComputingFunction;
  Function *AFComputingFunction;

  Function *ACStoreFunction;
  Function *AFStoreFunction;
  Function *AFPrintTopAmplificationPaths;

  // Instruction to AC/AF Maps
  ValueMap<Value*, Value*> InstructionACMap;
  ValueMap<Value*, Value*> InstructionAFMap;

  // Additional Utilities

public:
  Value *EmptyValuePointer;

  ACInstrumentation(Function *F);

//  void instrumentCallsForMemoryLoadOperation(Instruction *BaseInstruction,
//                                             BasicBlock::iterator *InstructionIterator,
//                                             long int *NumInstrumentedInstructions);
//  void instrumentCallsForUnaryOperation(Instruction *BaseInstruction,
//                                        BasicBlock::iterator *InstructionIterator,
//                                        long int *NumInstrumentedInstructions);
//  void instrumentCallsForBinaryOperation(Instruction *BaseInstruction,
//                                         BasicBlock::iterator *InstructionIterator,
//                                         long int *NumInstrumentedInstructions);
//  void instrumentCallsForOtherOperation(Instruction *BaseInstruction,
//                                        BasicBlock::iterator *InstructionIterator,
//                                        long int *NumInstrumentedInstructions);
//  void instrumentCallsForAFAnalysis(Instruction *BaseInstruction,
//                                    Instruction *LocationToInstrument,
//                                    BasicBlock::iterator *InstructionIterator,
//                                    long int *NumInstrumentedInstructions);

//  void instrumentAFAnalysisForPrintsAndReturns(Instruction *BaseInstruction,
//                                               BasicBlock::iterator *InstructionIterator,
//                                               long int *NumInstrumentedInstructions);

  void instrumentCallsToAnalyzeInstruction(Instruction *BaseInstruction,
                                           BasicBlock::iterator *InstructionIterator,
                                           long int *NumInstrumentedInstructions);
  void instrumentCallsForACComputation(Instruction *BaseInstruction,
                                       BasicBlock::iterator *InstructionIterator,
                                       long int *NumInstrumentedInstructions,
                                       int FunctionType);
  void instrumentCallsForAFComputation(Instruction *BaseInstruction,
                                       BasicBlock::iterator *InstructionIterator,
                                       long int *NumInstrumentedInstructions);

  Value *instrumentPhiNodeForAF(Value *OriginalPHI,
                                long *NumInstrumentedInstructions);
  Value *instrumentSelectForAF(Value *OriginalSelInstr,
                                BasicBlock::iterator *InstructionIterator,
                                long *NumInstrumentedInstructions);

  void instrumentBasicBlock(BasicBlock *BB,
                            long int *NumInstrumentedInstructions);
  void instrumentMainFunction(Function *F);

  //// Helper Functions
  Value *createArrayInIR(std::vector<Value*> ArrayOfValues,
                         IRBuilder<> *InstructionBuilder,
                         BasicBlock::iterator *InstructionIterator);

  /// Instruction based functions
  // Instruction finders
  Instruction *getInstructionAfterInitializationCalls(BasicBlock *BB);

  // Categorized by operations
  static bool canHaveGraphNode(const Instruction *Inst);
  static bool isPhiNode(const Instruction *Inst);
  static bool isMemoryLoadOperation(const Instruction *Inst);
  static bool isIntegerToFloatCastOperation(const Instruction *Inst);
  static bool isUnaryOperation(const Instruction *Inst);
  static bool isBinaryOperation(const Instruction *Inst);
  static bool isOtherOperation(const Instruction *Inst);
  static bool isNonACInstrinsicFunction(const Instruction *Inst);
  static bool isNonACFloatPointInstruction(const Instruction *Inst);

  // Categorized by Data Type
//  static bool isFPOperation(const Instruction *Inst);
  static bool isSingleFPOperation(const Instruction *Inst);
  static bool isDoubleFPOperation(const Instruction *Inst);

  /// Function based functions
  static bool isUnwantedFunction(const Function *Func);
  static bool isFunctionOfInterest(const Function *Func);

  // Utility Functions
  Value *createBBNameGlobalString(BasicBlock *BB);
  Value *createRegisterNameGlobalString(Instruction *Inst);
  Value *createInstructionGlobalString(Instruction *Inst);
  Value *createStringRefGlobalString(StringRef StringObj, Instruction *Inst);
  std::string getInstructionAsString(Instruction *Inst);
  static bool isInstructionOfInterest(Instruction *Inst);
  int getFunctionEnum(Instruction *Inst);
};

} // namespace atomiccondition

#endif // LLVM_ACINSTRUMENTATION_H
