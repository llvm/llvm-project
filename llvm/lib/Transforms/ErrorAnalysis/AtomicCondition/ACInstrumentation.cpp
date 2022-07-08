//
// Created by tanmay on 6/10/22.
//

#include <string>
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h"


using namespace llvm;
using namespace atomiccondition;
using namespace std;

int ACInstrumentation::VarCounter = 0;

void confFunction(Function *FunctionToSave, Function **StorageLocation,
                  GlobalValue::LinkageTypes LinkageType)
{
  // Save the function pointer
  if(StorageLocation != nullptr)
    *StorageLocation = FunctionToSave;
  if (FunctionToSave->getLinkage() != LinkageType)
    FunctionToSave->setLinkage(LinkageType);
}

#define SET_ODR_LIKAGE(name) \
    if (CurrentFunction->getName().str().find(name) != std::string::npos) { \
      CurrentFunction->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage); \
    }

// Searches module for functions to mark and creates pointers for them.
ACInstrumentation::ACInstrumentation(Function *InstrumentFunction) : FunctionToInstrument(InstrumentFunction),
                                                                     ACInitFunction(nullptr),
                                                                     CGInitFunction(nullptr),
                                                                     ACfp32UnaryFunction(nullptr),
                                                                     ACfp64UnaryFunction(nullptr),
                                                                     ACfp32BinaryFunction(nullptr),
                                                                     ACfp64BinaryFunction(nullptr),
                                                                     CGRecordPHIInstruction(nullptr),
                                                                     CGRecordBasicBlock(nullptr),
                                                                     CGCreateNode(nullptr),
                                                                     ACStoreFunction(nullptr),
                                                                     CGStoreFunction(nullptr),
                                                                     ACAnalysisFunction(nullptr),
                                                                     CGDotGraphFunction(nullptr)
{
  // Find and configure instrumentation functions
  Module *M = FunctionToInstrument->getParent();

  // Configuring all runtime functions and saving pointers.
  for(Module::iterator F = M->begin(); F != M->end(); ++F) {
    Function *CurrentFunction = &*F;

    // Only configuring functions with certain prefixes
    if (CurrentFunction->getName().str().find("fACCreate") != std::string::npos) {
      confFunction(CurrentFunction, &ACInitFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fACfp32UnaryDriver") != std::string::npos) {
      confFunction(CurrentFunction, &ACfp32UnaryFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fACfp64UnaryDriver") != std::string::npos) {
      confFunction(CurrentFunction, &ACfp64UnaryFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fACfp32BinaryDriver") != std::string::npos) {
      confFunction(CurrentFunction, &ACfp32BinaryFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fACfp64BinaryDriver") != std::string::npos) {
      confFunction(CurrentFunction, &ACfp64BinaryFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGInitialize") != std::string::npos) {
      confFunction(CurrentFunction, &CGInitFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGrecordPHIInstruction") != std::string::npos) {
      confFunction(CurrentFunction, &CGRecordPHIInstruction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGrecordCurrentBasicBlock") != std::string::npos) {
      confFunction(CurrentFunction, &CGRecordBasicBlock,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGcreateNode") != std::string::npos) {
      confFunction(CurrentFunction, &CGCreateNode,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fACStoreResult") != std::string::npos) {
      confFunction(CurrentFunction, &ACStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGStoreResult") != std::string::npos) {
      confFunction(CurrentFunction, &CGStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFAnalysis") != std::string::npos) {
      confFunction(CurrentFunction, &ACAnalysisFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fCGDotGraph") != std::string::npos) {
      confFunction(CurrentFunction, &CGDotGraphFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
  }
}

void ACInstrumentation::instrumentCallRecordingBasicBlock(BasicBlock* CurrentBB,
                                                          long *NumInstrumentedInstructions) {
  BasicBlock::iterator NextInst(CurrentBB->getFirstNonPHIOrDbg());
  while(NextInst->getOpcode() == 56 &&
         (static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() != "fCGInitialize" ||
          static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() != "fACCreate"))
    NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );
  std::vector<Value *> Args;

  CallInst *NewCallInstruction = nullptr;

  string BasicBlockString;
  raw_string_ostream RawBasicBlockString(BasicBlockString);
  CurrentBB->printAsOperand(RawBasicBlockString, false);
  Constant *BBValue = ConstantDataArray::getString(CurrentBB->getModule()->getContext(),
                                                   RawBasicBlockString.str().c_str(),
                                                   true);

  Value *BBValuePointer = new GlobalVariable(*CurrentBB->getModule(),
                                             BBValue->getType(),
                                             true,
                                             GlobalValue::InternalLinkage,
                                             BBValue);

  Args.push_back(BBValuePointer);

  NewCallInstruction = InstructionBuilder.CreateCall(CGRecordBasicBlock, Args);

  *NumInstrumentedInstructions+=1;
  assert(NewCallInstruction && "Invalid call instruction!");
}

void ACInstrumentation::instrumentCallRecordingPHIInstructions(BasicBlock* CurrentBB,
                                                               long *NumInstrumentedInstructions) {
  BasicBlock::iterator NextInst(CurrentBB->getFirstNonPHIOrDbg());
  while(NextInst->getOpcode() == 56 &&
         (static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() != "fCGInitialize" ||
          static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() != "fACCreate"))
    NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );

  long int NumberOfCalls = 0;

  for (BasicBlock::phi_iterator_impl<> CurrPhi = CurrentBB->phis().begin();
       CurrPhi != CurrentBB->phis().end();
       CurrPhi++) {
    std::vector<Value *> Args;

    string InstructionString;
    raw_string_ostream RawInstructionString(InstructionString);
    RawInstructionString << *static_cast<Instruction*>(&*CurrPhi);
    unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
    string Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                          RawInstructionString.str().substr(NonEmptyPosition);

    Constant *InstructionValue = ConstantDataArray::getString(CurrPhi->getModule()->getContext(),
                                                              Initializer.c_str(),
                                                              true);

    Value *InstructionValuePointer = new GlobalVariable(*CurrPhi->getModule(),
                                                        InstructionValue->getType(),
                                                        true,
                                                        GlobalValue::InternalLinkage,
                                                        InstructionValue);

    string BasicBlockString;
    raw_string_ostream RawBasicBlockString(BasicBlockString);
    CurrentBB->printAsOperand(RawBasicBlockString, false);
    Constant *BasicBlockValue = ConstantDataArray::getString(CurrentBB->getModule()->getContext(),
                                                     RawBasicBlockString.str().c_str(),
                                                     true);

    Value *BasicBlockValuePointer = new GlobalVariable(*CurrPhi->getModule(),
                                                        BasicBlockValue->getType(),
                                                        true,
                                                        GlobalValue::InternalLinkage,
                                                        BasicBlockValue);

    Args.push_back(InstructionValuePointer);
    Args.push_back(BasicBlockValuePointer);
    ArrayRef<Value *> ArgsRef(Args);

    InstructionBuilder.CreateCall(CGRecordPHIInstruction, ArgsRef);

    NumberOfCalls+=1;
  }

  *NumInstrumentedInstructions+=NumberOfCalls;
}

// Creates a node for LLVM memory load instructions in the computation graph.
void ACInstrumentation::instrumentCallsForMemoryLoadOperation(
    Instruction *BaseInstruction, long *NumInstrumentedInstructions) {
  assert((CGCreateNode!=nullptr) && "Function not initialized!");

  BasicBlock::iterator NextInst(BaseInstruction);
  NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );
  std::vector<Value *> Args;

  CallInst *NewCallInstruction = nullptr;

  unsigned long NonEmptyPosition;
  string Initializer;

  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *BaseInstruction;
  NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawInstructionString.str().substr(NonEmptyPosition);

  Constant *InstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);

  Constant *EmptyValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                            "",
                                                            true);

  Value *EmptyValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                      EmptyValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      EmptyValue);

  Args.push_back(InstructionValuePointer);
  Args.push_back(EmptyValuePointer);
  Args.push_back(EmptyValuePointer);
  Args.push_back(InstructionBuilder.getInt32(NodeKind::Register));
  ArrayRef<Value *> ArgsRef(Args);

  NewCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, ArgsRef);
  *NumInstrumentedInstructions+=1;

  assert(NewCallInstruction && "Invalid call instruction!");
  return;
}

// Instruments a call to calculate atomic condition for unary floating point
// instructions and creates a node for this instruction in the computation graph.
void ACInstrumentation::instrumentCallsForUnaryOperation(Instruction* BaseInstruction,
                                            long *NumInstrumentedInstructions) {
  assert((CGCreateNode!=nullptr) && (ACfp32BinaryFunction!=nullptr) && "Function not initialized!");
  Operation OpType;
  string FunctionName;


  switch (BaseInstruction->getOpcode()) {
  case 45:
    assert(static_cast<FPTruncInst*>(BaseInstruction)->getDestTy()->isFloatTy());
    OpType = Operation::TruncToFloat;
    break;
  case 56:
    FunctionName = static_cast<CallInst*>(BaseInstruction)->getCalledFunction()->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
    if (FunctionName.find("asin") != std::string::npos)
      OpType = Operation::ArcSin;
    else if(FunctionName.find("acos") != std::string::npos)
      OpType = Operation::ArcCos;
    else if(FunctionName.find("atan") != std::string::npos)
      OpType = Operation::ArcTan;
    else if(FunctionName.find("sinh") != std::string::npos)
      OpType = Operation::Sinh;
    else if(FunctionName.find("cosh") != std::string::npos)
      OpType = Operation::Cosh;
    else if(FunctionName.find("tanh") != std::string::npos)
      OpType = Operation::Tanh;
    else if(FunctionName.find("sin") != std::string::npos)
      OpType = Operation::Sin;
    else if(FunctionName.find("cos") != std::string::npos)
      OpType = Operation::Cos;
    else if(FunctionName.find("tan") != std::string::npos)
      OpType = Operation::Tan;
    else if(FunctionName.find("exp") != std::string::npos)
      OpType = Operation::Exp;
    else if(FunctionName.find("log") != std::string::npos)
      OpType = Operation::Log;
    else if(FunctionName.find("sqrt") != std::string::npos)
      OpType = Operation::Sqrt;
    else
      return;
    break;
  default:
    errs() << BaseInstruction << " is not a Binary Instruction.\n";
    return;
  }

  BasicBlock::iterator NextInst(BaseInstruction);
  NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );

  //----------------------------------------------------------------------------
  //------------------ Instrumenting AC Calculating Function ------------------
  //----------------------------------------------------------------------------

  std::vector<Value *> ACArgs;

  string XString = BaseInstruction->getOperand(0)->getName().str();
  if(XString == "")
    XString = to_string(VarCounter);
  VarCounter++;
  Constant *XValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                  XString,
                                                  true);
  Value *X = new GlobalVariable(*BaseInstruction->getModule(),
                                XValue->getType(),
                                true,
                                GlobalValue::InternalLinkage,
                                XValue);

  ACArgs.push_back(X);
  ACArgs.push_back(BaseInstruction->getOperand(0));

  ACArgs.push_back(InstructionBuilder.getInt32(OpType));

  ArrayRef<Value *> ACArgsRef(ACArgs);

  CallInst *ACCallInstruction = nullptr;

  // Branch based on data type of operation
  if(isSingleFPOperation(BaseInstruction)) {
    ACCallInstruction =
        InstructionBuilder.CreateCall(ACfp32UnaryFunction, ACArgsRef);
    *NumInstrumentedInstructions+=1;
  }
  else if(isDoubleFPOperation(BaseInstruction)) {
    ACCallInstruction =
        InstructionBuilder.CreateCall(ACfp64UnaryFunction, ACArgsRef);
    *NumInstrumentedInstructions+=1;
  }

  //----------------------------------------------------------------------------
  //----------------- Instrumenting CG Node creating function -----------------
  //----------------------------------------------------------------------------
  std::vector<Value *> CGArgs;

  CallInst *CGCallInstruction = nullptr;

  unsigned long NonEmptyPosition;
  string Initializer;

  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *BaseInstruction;
  NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawInstructionString.str().substr(NonEmptyPosition);

  Constant *InstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);


  string LeftOpInstructionString;
  raw_string_ostream RawLeftOpInstructionString(LeftOpInstructionString);
  if (!isa<Constant>(BaseInstruction->getOperand(0)))
    RawLeftOpInstructionString << *BaseInstruction->getOperand(0);
  else
    RawLeftOpInstructionString << "";
  NonEmptyPosition= RawLeftOpInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawLeftOpInstructionString.str().substr(NonEmptyPosition);

  Constant *LeftOpInstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                                  Initializer.c_str(),
                                                                  true);

  Value *LeftOpInstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                            LeftOpInstructionValue->getType(),
                                                            true,
                                                            GlobalValue::InternalLinkage,
                                                            LeftOpInstructionValue);

  Constant *EmptyValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                      "",
                                                      true);

  Value *EmptyValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                EmptyValue->getType(),
                                                true,
                                                GlobalValue::InternalLinkage,
                                                EmptyValue);

  CGArgs.push_back(InstructionValuePointer);
  CGArgs.push_back(LeftOpInstructionValuePointer);
  CGArgs.push_back(EmptyValuePointer);
  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::UnaryInstruction));
  ArrayRef<Value *> CGArgsRef(CGArgs);

  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
  *NumInstrumentedInstructions+=1;
  assert(ACCallInstruction && CGCallInstruction && "Invalid call instruction!");
  return;
}

// Instruments a call to calculate atomic condition for binary floating point
// instructions and creates a node for this instruction in the computation graph.
void ACInstrumentation::instrumentCallsForBinaryOperation(Instruction* BaseInstruction,
                                             long *NumInstrumentedInstructions) {
  assert((CGCreateNode!=nullptr) && (ACfp64BinaryFunction!=nullptr) && "Function not initialized!");

  Operation OpType;
  switch (BaseInstruction->getOpcode()) {
  case 14:
    OpType = Operation::Add;
    break;
  case 16:
    OpType = Operation::Sub;
    break;
  case 18:
    OpType = Operation::Mul;
    break;
  case 21:
    OpType = Operation::Div;
    break;
  default:
    errs() << BaseInstruction << " is not a Binary Instruction.\n";
    return;
  }

  BasicBlock::iterator NextInst(BaseInstruction);
  NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );

  //----------------------------------------------------------------------------
  //------------------ Instrumenting AC Calculating Function ------------------
  //----------------------------------------------------------------------------

  std::vector<Value *> Args;

  string XString = BaseInstruction->getOperand(0)->getName().str();
  if(XString == "")
    XString = to_string(VarCounter);
  VarCounter++;
  Constant *XValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                  XString,
                                                  true);
  Value *X = new GlobalVariable(*BaseInstruction->getModule(),
                                XValue->getType(),
                                true,
                                GlobalValue::InternalLinkage,
                                XValue);

  string YString = BaseInstruction->getOperand(1)->getName().str();
  if(YString == "")
    YString = to_string(VarCounter);
  VarCounter++;
  Constant *YValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                  YString,
                                                  true);
  Value *Y = new GlobalVariable(*BaseInstruction->getModule(),
                                YValue->getType(),
                                true,
                                GlobalValue::InternalLinkage,
                                YValue);

  Args.push_back(X);
  Args.push_back(BaseInstruction->getOperand(0));
  Args.push_back(Y);
  Args.push_back(BaseInstruction->getOperand(1));
  Args.push_back(InstructionBuilder.getInt32(OpType));

  ArrayRef<Value *> ArgsRef(Args);

  CallInst *NewCallInstruction = nullptr;

  // Branch based on data type of operation
  if(isSingleFPOperation(BaseInstruction)) {
    NewCallInstruction =
        InstructionBuilder.CreateCall(ACfp32BinaryFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }
  else if(isDoubleFPOperation(BaseInstruction)) {
    NewCallInstruction =
        InstructionBuilder.CreateCall(ACfp64BinaryFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }

  //----------------------------------------------------------------------------
  //----------------- Instrumenting CG Node creating function -----------------
  //----------------------------------------------------------------------------
  std::vector<Value *> CGArgs;

  CallInst *CGCallInstruction = nullptr;

  unsigned long NonEmptyPosition;
  string Initializer;

  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *BaseInstruction;
  NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawInstructionString.str().substr(NonEmptyPosition);

  Constant *InstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);


  string LeftOpInstructionString;
  raw_string_ostream RawLeftOpInstructionString(LeftOpInstructionString);
  if (!isa<Constant>(BaseInstruction->getOperand(0)))
    RawLeftOpInstructionString << *BaseInstruction->getOperand(0);
  else
    RawLeftOpInstructionString << "";
  NonEmptyPosition= RawLeftOpInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawLeftOpInstructionString.str().substr(NonEmptyPosition);


  Constant *LeftOpInstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                                  Initializer.c_str(),
                                                                  true);

  Value *LeftOpInstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                            LeftOpInstructionValue->getType(),
                                                            true,
                                                            GlobalValue::InternalLinkage,
                                                            LeftOpInstructionValue);

  string RightOpInstructionString;
  raw_string_ostream RawRightOpInstructionString(RightOpInstructionString);
  if (!isa<Constant>(BaseInstruction->getOperand(1)))
    RawRightOpInstructionString << *BaseInstruction->getOperand(1);
  else
    RawRightOpInstructionString << "";
  NonEmptyPosition= RawRightOpInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                        RawRightOpInstructionString.str().substr(NonEmptyPosition);

  Constant *RightOpInstructionValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                                   Initializer.c_str(),
                                                                  true);

  Value *RightOpInstructionValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                            RightOpInstructionValue->getType(),
                                                            true,
                                                            GlobalValue::InternalLinkage,
                                                            RightOpInstructionValue);

  CGArgs.push_back(InstructionValuePointer);
  CGArgs.push_back(LeftOpInstructionValuePointer);
  CGArgs.push_back(RightOpInstructionValuePointer);
  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::BinaryInstruction));
  ArrayRef<Value *> CGArgsRef(CGArgs);

  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
  *NumInstrumentedInstructions+=1;

  assert(NewCallInstruction && CGCallInstruction && "Invalid call instruction!");
  return;
}


void ACInstrumentation::instrumentBasicBlock(BasicBlock *BB,
                                             long *NumInstrumentedInstructions) {
//  if (ACInstrumentation::isUnwantedFunction(BB->getParent()))
//    return;
  instrumentCallRecordingPHIInstructions(BB,
                                       &*NumInstrumentedInstructions);
  instrumentCallRecordingBasicBlock(BB,
                                    &*NumInstrumentedInstructions);

  for (BasicBlock::iterator I = BB->begin();
       I != BB->end(); ++I) {
    Instruction* CurrentInstruction = &*I;

    // Branch based on kind of Instruction
    if(isMemoryLoadOperation(CurrentInstruction)) {
      instrumentCallsForMemoryLoadOperation(CurrentInstruction,
                                            &*NumInstrumentedInstructions);
    }
    else if(isUnaryOperation(CurrentInstruction)) {
      instrumentCallsForUnaryOperation(CurrentInstruction,
                                       &*NumInstrumentedInstructions);
    }
    else if(isBinaryOperation(CurrentInstruction)) {
      instrumentCallsForBinaryOperation(CurrentInstruction,
                                        &*NumInstrumentedInstructions);
    }
  }

  return;
}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  assert((ACInitFunction!=nullptr) &&
         (CGInitFunction!=nullptr) &&
         (ACStoreFunction!=nullptr) &&
         (CGStoreFunction!=nullptr) &&
         (CGDotGraphFunction!=nullptr) &&
         (ACAnalysisFunction!=nullptr) &&
         "Function not initialized!");
  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);
  std::vector<Value *> ACInitCallArgs, CGInitCallArgs, AnalysisCallArgs;
  std::vector<Value *> ACStoreCallArgs, CGStoreCallArgs;
  std::vector<Value *> DotGraphCallArgs;

  CallInst *ACInitCallInstruction, *CGInitCallInstruction, *AnalysisCallInstruction;
  CallInst *StoreACTableCallInstruction, *StoreCGTableCallInstruction;
  CallInst *DotGraphCallInstruction;

  // Instrumenting Initialization call instruction
//  Args.push_back(InstructionBuilder.getInt64(1000));
  // Check if function has arguments or not
  // TODO: Handle the case if program has arguments by creating a function
  //  initializing everything in case of arguments.
//  if (F->arg_size() == 2) {
//    // Push parameters
//    for (Argument *I = F->arg_begin(); I != F->arg_end(); ++I) {
//      Value *V = &(*I);
//      Args.push_back(V);
//    }
//    ArrayRef<Value *> args_ref(Args);
//    CallInstruction = InstructionBuilder.CreateCall(fpc_init_args, args_ref);
//  } else
//  {
  ACInitCallInstruction = InstructionBuilder.CreateCall(ACInitFunction, ACInitCallArgs);
  CGInitCallInstruction = InstructionBuilder.CreateCall(CGInitFunction, CGInitCallArgs);
//  }

  // Instrument call to print table
  for (Function::iterator BBIter=F->begin(); BBIter != F->end(); ++BBIter) {
    for (BasicBlock::iterator InstIter=BBIter->begin(); InstIter != BBIter->end(); ++InstIter) {
      Instruction *CurrentInstruction = &(*InstIter);
      if (isa<ReturnInst>(CurrentInstruction) || isa<ResumeInst>(CurrentInstruction)) {
        ArrayRef<Value *> ACStoreCallArgsRef(ACStoreCallArgs);
        ArrayRef<Value *> CGStoreCallArgsRef(CGStoreCallArgs);
        InstructionBuilder.SetInsertPoint(CurrentInstruction);
        StoreACTableCallInstruction = InstructionBuilder.CreateCall(ACStoreFunction, ACStoreCallArgsRef);
        StoreCGTableCallInstruction = InstructionBuilder.CreateCall(CGStoreFunction, CGStoreCallArgsRef);

        ArrayRef<Value *> DotGraphCallArgsRef(DotGraphCallArgs);
        DotGraphCallInstruction = InstructionBuilder.CreateCall(CGDotGraphFunction, DotGraphCallArgsRef);

        ArrayRef<Value *> AnalysisCallArgsRef(AnalysisCallArgs);
        AnalysisCallInstruction = InstructionBuilder.CreateCall(ACAnalysisFunction, AnalysisCallArgsRef);
      }
    }
  }

  assert(ACInitCallInstruction && CGInitCallInstruction && AnalysisCallInstruction && "Invalid call instruction!");
  assert(StoreACTableCallInstruction && StoreCGTableCallInstruction && "Invalid call instruction!");
  assert(DotGraphCallInstruction && "Invalid call instruction!");
  return;
}

bool ACInstrumentation::isMemoryLoadOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::Alloca ||
         Inst->getOpcode() == Instruction::Load;
}

bool ACInstrumentation::isUnaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FPTrunc ||
         Inst->getOpcode() == Instruction::Call;
}

// TODO: Check for FRem case.
bool ACInstrumentation::isBinaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FAdd ||
         Inst->getOpcode() == Instruction::FSub ||
         Inst->getOpcode() == Instruction::FMul ||
         Inst->getOpcode() == Instruction::FDiv;
}


bool ACInstrumentation::isSingleFPOperation(const Instruction *Inst) {
  if (isUnaryOperation(Inst)) {
    switch (Inst->getOpcode()) {
    case 45:
      return static_cast<const FPTruncInst*>(Inst)->getSrcTy()->isFloatTy();
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst*>(Inst)->getArgOperand(0)->getType()->isFloatTy();
    default:
      errs() << "Not an FP32 operation.";
    }
  } else if(isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isFloatTy() &&
           Inst->getOperand(1)->getType()->isFloatTy();
  }

  return false;
}

bool ACInstrumentation::isDoubleFPOperation(const Instruction *Inst) {
  if (isUnaryOperation(Inst)) {
    switch (Inst->getOpcode()) {
    case 45:
      return static_cast<const FPTruncInst*>(Inst)->getSrcTy()->isDoubleTy();
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst*>(Inst)->getArgOperand(0)->getType()->isDoubleTy();
    default:
      errs() << "Not an FP64 operation.";
    }
  } else if(isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isDoubleTy() &&
           Inst->getOperand(1)->getType()->isDoubleTy();
  }

  return false;
}

bool ACInstrumentation::isUnwantedFunction(const Function *Func) {
  return Func->getName().str().find("fAC") != std::string::npos ||
         Func->getName().str().find("fCG") != std::string::npos ||
         Func->getName().str().find("fAF") != std::string::npos ||
         Func->getName().str().find("ACItem") != std::string::npos;
}
