//
// Created by tanmay on 6/10/22.
//

#include <string>
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h"
#include "llvm/Transforms/ErrorAnalysis/Utilities/FunctionMatchers.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/DebugInfo.h"

using namespace llvm;
using namespace atomiccondition;
using namespace std;

void confFunction(Function *FunctionToSave, Function **StorageLocation,
                  GlobalValue::LinkageTypes LinkageType) {
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
                                                                     AFInitFunction(nullptr),
                                                                     ACComputingFunction(nullptr),
                                                                     AFComputingFunction(nullptr),
//                                                                     ACUnaryFunction(nullptr),
//                                                                     ACBinaryFunction(nullptr),
                                                                     ACStoreFunction(nullptr),
                                                                     AFStoreFunction(nullptr)
//                                                                     AFPrintTopAmplificationPaths(nullptr),
//                                                                     AFAnalysisFunction(nullptr)
{
  // Find and configure instrumentation functions
  Module *M = FunctionToInstrument->getParent();

  // Configuring all runtime functions and saving pointers.
  for(Module::iterator F = M->begin(); F != M->end(); ++F) {
    Function *CurrentFunction = &*F;

    // Only configuring functions with certain prefixes
    if(!CurrentFunction->hasName()) {

    }
    else if (CurrentFunction->getName().str().find("fACCreate") != std::string::npos) {
      confFunction(CurrentFunction, &ACInitFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
        else if (CurrentFunction->getName().str().find("fAFInitialize") != std::string::npos) {
          confFunction(CurrentFunction, &AFInitFunction,
                       GlobalValue::LinkageTypes::LinkOnceODRLinkage);
        }
    else if (CurrentFunction->getName().str().find("fACComputeAC") != std::string::npos) {
      confFunction(CurrentFunction, &ACComputingFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFComputeAF") != std::string::npos) {
      confFunction(CurrentFunction, &AFComputingFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
//    else if (CurrentFunction->getName().str().find("fACUnaryDriver") != std::string::npos) {
//      confFunction(CurrentFunction, &ACUnaryFunction,
//                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
//    }
//    else if (CurrentFunction->getName().str().find("fACBinaryDriver") != std::string::npos) {
//      confFunction(CurrentFunction, &ACBinaryFunction,
//                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
//    }
    else if (CurrentFunction->getName().str().find("fACStoreACs") != std::string::npos) {
      confFunction(CurrentFunction, &ACStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFStoreAFs") != std::string::npos) {
      confFunction(CurrentFunction, &AFStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
//    else if (CurrentFunction->getName().str().find("fAFPrintTopAmplificationPaths") != std::string::npos) {
//      confFunction(CurrentFunction, &AFPrintTopAmplificationPaths,
//                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
//    }
//    else if (CurrentFunction->getName().str().find("fAFAnalysis") != std::string::npos) {
//      confFunction(CurrentFunction, &AFAnalysisFunction,
//                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
//    }
  }
  
  Constant *EmptyValue = ConstantDataArray::getString(InstrumentFunction->getContext(),
                                                      "",
                                                      true);
  
  EmptyValuePointer = new GlobalVariable(*InstrumentFunction->getParent(),
                                         EmptyValue->getType(),
                                         true,
                                         GlobalValue::InternalLinkage,
                                         EmptyValue);
}

// Instruments calls analyzing sensitivity of this instruction
void ACInstrumentation::instrumentCallsToAnalyzeInstruction(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  // Return if we cannot analyze this function
  int F = getFunctionEnum(BaseInstruction);
  if(F == -1)
    return ;

  instrumentCallsForACComputation(BaseInstruction,
                                  InstructionIterator,
                                  NumInstrumentedInstructions,
                                  F);

  instrumentCallsForAFComputation(BaseInstruction,
                                  InstructionIterator,
                                  NumInstrumentedInstructions);
}

// Instruments a call to calculate atomic condition
void ACInstrumentation::instrumentCallsForACComputation(
    Instruction *BaseInstruction,
    BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions,
    int FunctionType) {
  // Ensuring ACComputingFunction is linked
  assert((ACComputingFunction!=nullptr) && "Function not initialized!");
  assert((FunctionType != -1) && "Function cannot be analyzed");

  // Positioning the instruction builder
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> ACArgs;


  // Creating a Global string for the Register the Result of this instruction is
  // stored in.
  Value *ResultNamePointer = createRegisterNameGlobalString(
      static_cast<Instruction*>(BaseInstruction));

  std::vector<Value*> OpRegisterNamesArray;
  std::vector<Value*> OperandValuesArray;

  // Looping through operands of BaseInstruction
  for (int I = 0; I < (int)BaseInstruction->getNumOperands(); ++I) {
    // Creating a Global string representing the register name of operand if
    // if is not a constant.
    Value *OpRegisterNamePointer;
    if(!(isa<Constant>(BaseInstruction->getOperand(I)) ||
          isa<Argument>(BaseInstruction->getOperand(I))))
      OpRegisterNamePointer=createRegisterNameGlobalString(
          static_cast<Instruction*>(BaseInstruction->getOperand(I)));
    else
      OpRegisterNamePointer = EmptyValuePointer;

    OpRegisterNamesArray.push_back(OpRegisterNamePointer);

    Value *OperandValue = BaseInstruction->getOperand(I);
    // Create a Cast Instruction to double in case operation is float operation.
    if (isSingleFPOperation(&*BaseInstruction)) {
      OperandValue = InstructionBuilder.CreateFPCast(
          BaseInstruction->getOperand(I),
          Type::getDoubleTy(BaseInstruction->getModule()->getContext()));

      (*InstructionIterator)++;
      *NumInstrumentedInstructions++;
    }

    OperandValuesArray.push_back(OperandValue);
  }

  Value *AllocatedOpRegisterNamesArray = createArrayInIR(OpRegisterNamesArray,
                                                         &InstructionBuilder,
                                                         InstructionIterator);

  Value *AllocatedOperandValuesArray = createArrayInIR(OperandValuesArray,
                                                       &InstructionBuilder,
                                                       InstructionIterator);

  Value *FileNameValuePointer;
  int LineNumber;
  if(BaseInstruction->getDebugLoc()) {
    std::string FileLocation = BaseInstruction->getDebugLoc()->getDirectory().str() +
                               "/" +
                               BaseInstruction->getDebugLoc()->getFilename().str();
    FileNameValuePointer = createStringRefGlobalString(FileLocation, BaseInstruction);
    LineNumber = BaseInstruction->getDebugLoc().getLine();
  }
  else {
    FileNameValuePointer = EmptyValuePointer;
    LineNumber = -1;
  }

  // Creating array of parameters
  ACArgs.push_back(ResultNamePointer);
  ACArgs.push_back(AllocatedOpRegisterNamesArray);
  ACArgs.push_back(AllocatedOperandValuesArray);
  ACArgs.push_back(InstructionBuilder.getInt32(FunctionType));
  ACArgs.push_back(FileNameValuePointer);
  ACArgs.push_back(InstructionBuilder.getInt32(LineNumber));

  // Creating a call to ACComputingFunction with above parameters
  ArrayRef<Value *> ACArgsRef(ACArgs);
  CallInst *ACComputingCallInstruction =
      InstructionBuilder.CreateCall(ACComputingFunction, ACArgsRef,
                                    "AC");

  // 2*(Get Location + Store Value at Location) * NumOperands +
  // 2*(Allocate for Array + GetArrayLocation) + ACComputation Function Call
  *NumInstrumentedInstructions += 4*BaseInstruction->getNumOperands()+5;
  (*InstructionIterator)++;

  std::pair<Value*, Value*> InstructionACPair = std::make_pair(BaseInstruction,
                                                                 ACComputingCallInstruction);
  InstructionACMap.insert(InstructionACPair);

  assert(ACComputingCallInstruction && "Invalid call instruction!");
  return ;
}

void ACInstrumentation::instrumentCallsForAFComputation(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  // Ensuring AFComputingFunction is linked
  assert((ACComputingFunction!=nullptr) && "Function not initialized!");
  assert((getFunctionEnum(BaseInstruction) != -1) && "Function cannot be analyzed");

  // Positioning the instruction builder
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> AFArgs;

  std::vector<Value*> AFArray;

  for (int I = 0; I < (int)BaseInstruction->getNumOperands(); ++I) {
    if(getFunctionEnum(
            static_cast<Instruction *>(BaseInstruction->getOperand(I))) != -1)
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
    else
      AFArray.push_back(ConstantPointerNull::get(InstructionBuilder.getPtrTy()));
  }

  Value *AllocatedAFArray = createArrayInIR(AFArray,
                                            &InstructionBuilder,
                                            InstructionIterator);

  AFArgs.push_back(InstructionACMap[BaseInstruction]);
  AFArgs.push_back(AllocatedAFArray);
  AFArgs.push_back(InstructionBuilder.getInt32(BaseInstruction->getNumOperands()));

  // Creating a call to AFComputingFunction with above parameters
  ArrayRef<Value *> AFArgsRef(AFArgs);
  CallInst *AFComputingCallInstruction =
      InstructionBuilder.CreateCall(AFComputingFunction, AFArgsRef,
                                    "AF");

  // (Get Location + Store Value at Location) * NumOperands +
  // (Allocate for Array + GetArrayLocation) + AFComputation Function Call
  *NumInstrumentedInstructions += 2*BaseInstruction->getNumOperands()+3;
  (*InstructionIterator)++;

  std::pair<Value*, Value*> InstructionAFPair = std::make_pair(BaseInstruction,
                                                                 AFComputingCallInstruction);
  InstructionAFMap.insert(InstructionAFPair);

  assert(AFComputingCallInstruction && "Invalid call instruction!");
}

// Instruments a call to calculate atomic condition for unary floating point
// instructions and creates a node for this instruction in the computation graph.
//void ACInstrumentation::instrumentCallsForUnaryOperation(Instruction *BaseInstruction,
//                                                         BasicBlock::iterator *InstructionIterator,
//                                                         long *NumInstrumentedInstructions) {
//  assert((ACUnaryFunction!=nullptr) && "Function not initialized!");
//
//  Operation OpType;
//  string FunctionName = "";
//
//
//  switch (BaseInstruction->getOpcode()) {
//  case 12:
//    OpType = Operation::Neg;
//    break;
//  case 56:
//    if(static_cast<CallInst*>(BaseInstruction)->getCalledFunction() &&
//        static_cast<CallInst*>(BaseInstruction)->getCalledFunction()->hasName())
//      FunctionName = static_cast<CallInst*>(BaseInstruction)->getCalledFunction()->getName().str();
//    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
//    if (isASinFunction(FunctionName))
//      OpType = Operation::ArcSin;
//    else if(isACosFunction(FunctionName))
//      OpType = Operation::ArcCos;
//    else if(isATanFunction(FunctionName))
//      OpType = Operation::ArcTan;
//    else if(isSinhFunction(FunctionName))
//      OpType = Operation::Sinh;
//    else if(isCoshFunction(FunctionName))
//      OpType = Operation::Cosh;
//    else if(isTanhFunction(FunctionName))
//      OpType = Operation::Tanh;
//    else if(isSinFunction(FunctionName))
//      OpType = Operation::Sin;
//    else if(isCosFunction(FunctionName))
//      OpType = Operation::Cos;
//    else if(isTanFunction(FunctionName))
//      OpType = Operation::Tan;
//    else if(isExpFunction(FunctionName))
//      OpType = Operation::Exp;
//    else if(isLogFunction(FunctionName))
//      OpType = Operation::Log;
//    else if(isSqrtFunction(FunctionName))
//      OpType = Operation::Sqrt;
//    else
//      return;
//    break;
//  default:
////    errs() << BaseInstruction << " is not a Unary Instruction.\n";
//    return;
//  }
//
//  // Positioning the instruction builder
//  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
//
//
//  //----------------------------------------------------------------------------
//  //------------------ Instrumenting AC Calculating Function ------------------
//  //----------------------------------------------------------------------------
//
//  std::vector<Value *> ACArgs;
//
//  Value *LeftOpRegisterNamePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpRegisterNamePointer = createRegisterNameGlobalString(
//        static_cast<Instruction *>(LeftValue));
//  else
//    LeftOpRegisterNamePointer = EmptyValuePointer;
//
//  ACArgs.push_back(LeftOpRegisterNamePointer);
//
//  if (isSingleFPOperation(&*BaseInstruction)) {
//    Value *CastToDouble = InstructionBuilder.CreateFPCast(
//        BaseInstruction->getOperand(0),
//        Type::getDoubleTy(BaseInstruction->getModule()->getContext()));
//    ACArgs.push_back(CastToDouble);
//    (*InstructionIterator)++;
//  } else if (isDoubleFPOperation(BaseInstruction)) {
//    ACArgs.push_back(BaseInstruction->getOperand(0));
//  }
//
//  ACArgs.push_back(InstructionBuilder.getInt32(OpType));
//
//  ArrayRef<Value *> ACArgsRef(ACArgs);
//
//  CallInst *ACComputingCallInstruction = nullptr;
//
//  ACComputingCallInstruction =
//      InstructionBuilder.CreateCall(ACUnaryFunction, ACArgsRef,
//                                    "AC");
//  *NumInstrumentedInstructions += 1;
//  (*InstructionIterator)++;
//
//  assert(ACComputingCallInstruction && "Invalid call instruction!");
//
//  //----------------------------------------------------------------------------
//  //----------------- Instrumenting CG Node creating function -----------------
//  //----------------------------------------------------------------------------
//  std::vector<Value *> CGArgs;
//
//  CallInst *CGCallInstruction = nullptr;
//
//  Value *LeftOpInstructionValuePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpInstructionValuePointer = createInstructionGlobalString(
//        static_cast<Instruction*>(LeftValue));
//  else
//    LeftOpInstructionValuePointer = EmptyValuePointer;
//
//  Value *FileNameValuePointer;
//  int LineNumber;
//  if(BaseInstruction->getDebugLoc()) {
//    std::string FileLocation = BaseInstruction->getDebugLoc()->getDirectory().str() +
//                               "/" +
//                               BaseInstruction->getDebugLoc()->getFilename().str();
//    FileNameValuePointer = createStringRefGlobalString(FileLocation, BaseInstruction);
//    LineNumber = BaseInstruction->getDebugLoc().getLine();
//  }
//  else {
//    FileNameValuePointer = EmptyValuePointer;
//    LineNumber = -1;
//  }
//
//  CGArgs.push_back(createInstructionGlobalString(BaseInstruction));
//  CGArgs.push_back(LeftOpInstructionValuePointer);
//  CGArgs.push_back(EmptyValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::UnaryInstruction));
//  CGArgs.push_back(ACComputingCallInstruction);
//  CGArgs.push_back(FileNameValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(LineNumber));
//  ArrayRef<Value *> CGArgsRef(CGArgs);
//
//  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(CGCallInstruction && "Invalid call instruction!");
//  return;
//}

// Instruments a call to calculate atomic condition for binary floating point
// instructions and creates a node for this instruction in the computation graph.
//void ACInstrumentation::instrumentCallsForBinaryOperation(Instruction *BaseInstruction,
//                                                          BasicBlock::iterator *InstructionIterator,
//                                                          long *NumInstrumentedInstructions) {
//  assert((ACBinaryFunction!=nullptr) && "Function not initialized!");
//  assert((CGCreateNode!=nullptr) && "Function not initialized!");
//
//  Operation OpType;
//  switch (BaseInstruction->getOpcode()) {
//  case 14:
//    OpType = Operation::Add;
//    break;
//  case 16:
//    OpType = Operation::Sub;
//    break;
//  case 18:
//    OpType = Operation::Mul;
//    break;
//  case 21:
//    OpType = Operation::Div;
//    break;
//  default:
////    errs() << BaseInstruction << " is not a Binary Instruction.\n";
//    return;
//  }
//
//  // Positioning the instruction builder
//  IRBuilder<> InstructionBuilder( (*InstructionIterator)->getNextNode() );
//
//  Value *LeftValue = resolveToCGCorrespondingInstruction(BaseInstruction->getOperand(0));
//  Value *RightValue = resolveToCGCorrespondingInstruction(BaseInstruction->getOperand(1));
//
//  //----------------------------------------------------------------------------
//  //------------------ Instrumenting AC Calculating Function ------------------
//  //----------------------------------------------------------------------------
//
//  std::vector<Value *> Args;
//
//  Value *LeftOpRegisterNamePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpRegisterNamePointer = createRegisterNameGlobalString(
//        static_cast<Instruction*>(LeftValue));
//  else
//    LeftOpRegisterNamePointer = EmptyValuePointer;
//
//  Value *RightOpRegisterNamePointer;
//  if(RightValue != EmptyValuePointer)
//    RightOpRegisterNamePointer = createRegisterNameGlobalString(
//        static_cast<Instruction*>(RightValue));
//  else
//    RightOpRegisterNamePointer = EmptyValuePointer;
//
//  Args.push_back(LeftOpRegisterNamePointer);
//  if (isSingleFPOperation(BaseInstruction)) {
//    Value *CastToDouble = InstructionBuilder.CreateFPCast(
//        BaseInstruction->getOperand(0),
//        Type::getDoubleTy(BaseInstruction->getModule()->getContext()));
//    Args.push_back(CastToDouble);
//    (*InstructionIterator)++;
//  } else if (isDoubleFPOperation(BaseInstruction)) {
//    Args.push_back(BaseInstruction->getOperand(0));
//  }
//  Args.push_back(RightOpRegisterNamePointer);
//  if (isSingleFPOperation(BaseInstruction)) {
//    Value *CastToDouble = InstructionBuilder.CreateFPCast(
//        BaseInstruction->getOperand(1),
//        Type::getDoubleTy(BaseInstruction->getModule()->getContext()));
//    Args.push_back(CastToDouble);
//    (*InstructionIterator)++;
//  } else if (isDoubleFPOperation(BaseInstruction)) {
//    Args.push_back(BaseInstruction->getOperand(1));
//  }
//  Args.push_back(InstructionBuilder.getInt32(OpType));
//
//  ArrayRef<Value *> ArgsRef(Args);
//
//  CallInst *ACComputingCallInstruction = nullptr;
//
//  ACComputingCallInstruction = InstructionBuilder.CreateCall(ACBinaryFunction,
//                                                             ArgsRef,
//                                                             "AC");
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(ACComputingCallInstruction && "Invalid call instruction!");
//
//  //----------------------------------------------------------------------------
//  //----------------- Instrumenting CG Node creating function -----------------
//  //----------------------------------------------------------------------------
//  std::vector<Value *> CGArgs;
//
//  CallInst *CGCallInstruction = nullptr;
//
//  Value *LeftOpInstructionValuePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpInstructionValuePointer = createInstructionGlobalString(
//        static_cast<Instruction*>(LeftValue));
//  else
//    LeftOpInstructionValuePointer = EmptyValuePointer;
//
//  Value *RightOpInstructionValuePointer;
//  if(RightValue != EmptyValuePointer)
//    RightOpInstructionValuePointer = createInstructionGlobalString(
//        static_cast<Instruction*>(RightValue));
//  else
//    RightOpInstructionValuePointer = EmptyValuePointer;
//
//  Value *FileNameValuePointer;
//  int LineNumber;
//  if(BaseInstruction->getDebugLoc()) {
//    std::string FileLocation = BaseInstruction->getDebugLoc()->getDirectory().str() +
//                               "/" +
//                               BaseInstruction->getDebugLoc()->getFilename().str();
//    FileNameValuePointer = createStringRefGlobalString(FileLocation, BaseInstruction);
//    LineNumber = BaseInstruction->getDebugLoc().getLine();
//  }
//  else {
//    FileNameValuePointer = EmptyValuePointer;
//    LineNumber = -1;
//  }
//
//  CGArgs.push_back(createInstructionGlobalString(BaseInstruction));
//  CGArgs.push_back(LeftOpInstructionValuePointer);
//  CGArgs.push_back(RightOpInstructionValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::BinaryInstruction));
//  CGArgs.push_back(ACComputingCallInstruction);
//  CGArgs.push_back(FileNameValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(LineNumber));
//  ArrayRef<Value *> CGArgsRef(CGArgs);
//
//  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(CGCallInstruction && "Invalid call instruction!");
//  return;
//}

//void ACInstrumentation::instrumentCallsForOtherOperation(Instruction *BaseInstruction,
//                                                         BasicBlock::iterator *InstructionIterator,
//                                                         long *NumInstrumentedInstructions) {
//  assert((ACBinaryFunction!=nullptr) && "Function not initialized!");
//  assert((CGCreateNode!=nullptr) && "Function not initialized!");
//
//  // If its not an FMA/Other instruction, simply return without instrumenting
//  if (!(BaseInstruction->getOpcode() == Instruction::Call &&
//    static_cast<const CallInst*>(BaseInstruction)->getCalledFunction() &&
//    static_cast<const CallInst*>(BaseInstruction)->getCalledFunction()->hasName() &&
//    (static_cast<const CallInst*>(BaseInstruction)->getCalledFunction()->getName().str().find("llvm.fmuladd") !=
//         std::string::npos ||
//    static_cast<const CallInst*>(BaseInstruction)->getCalledFunction()->getName().str().find("llvm.fma") !=
//         std::string::npos)))
//    return ;
//
//  // Positioning the instruction builder
//  IRBuilder<> InstructionBuilder( (*InstructionIterator)->getNextNode() );
//
//  Value *LeftValue = resolveToCGCorrespondingInstruction(BaseInstruction->getOperand(1));
//  Value *RightValue = resolveToCGCorrespondingInstruction(BaseInstruction->getOperand(2));
//  //----------------------------------------------------------------------------
//  //------------------ Instrumenting AC Calculating Function ------------------
//  //----------------------------------------------------------------------------
//
//  std::vector<Value *> Args;
//
//  Value *LeftOpRegisterNamePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpRegisterNamePointer = createRegisterNameGlobalString(
//        static_cast<Instruction*>(LeftValue));
//  else
//    LeftOpRegisterNamePointer = EmptyValuePointer;
//
//  Value *RightOpRegisterNamePointer;
//  if(RightValue != EmptyValuePointer)
//    RightOpRegisterNamePointer = createRegisterNameGlobalString(
//        static_cast<Instruction *>(RightValue));
//  else
//    RightOpRegisterNamePointer = EmptyValuePointer;
//
//  Args.push_back(LeftOpRegisterNamePointer);
//  Args.push_back(static_cast<const CallInst*>(BaseInstruction)->getArgOperand(1));
//  Args.push_back(RightOpRegisterNamePointer);
//  Args.push_back(static_cast<const CallInst*>(BaseInstruction)->getArgOperand(2));
//  Args.push_back(InstructionBuilder.getInt32(Operation::Add));
//
//  ArrayRef<Value *> ArgsRef(Args);
//
//  CallInst *ACComputingCallInstruction = nullptr;
//
//  ACComputingCallInstruction =
//      InstructionBuilder.CreateCall(ACBinaryFunction, ArgsRef,
//                                    "AC");
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(ACComputingCallInstruction && "Invalid call instruction!");
//
//  //----------------------------------------------------------------------------
//  //----------------- Instrumenting CG Node creating function -----------------
//  //----------------------------------------------------------------------------
//  std::vector<Value *> CGArgs;
//
//  CallInst *CGCallInstruction = nullptr;
//
//  Value *LeftOpInstructionValuePointer;
//  if(LeftValue != EmptyValuePointer)
//    LeftOpInstructionValuePointer = createInstructionGlobalString(
//        static_cast<Instruction*>(LeftValue));
//  else
//    LeftOpInstructionValuePointer = EmptyValuePointer;
//
//  Value *RightOpInstructionValuePointer;
//  if(RightValue != EmptyValuePointer)
//    RightOpInstructionValuePointer = createInstructionGlobalString(
//        static_cast<Instruction*>(RightValue));
//  else
//    RightOpInstructionValuePointer = EmptyValuePointer;
//
//  Value *FileNameValuePointer;
//  int LineNumber;
//  if(BaseInstruction->getDebugLoc()) {
//    FileNameValuePointer = createStringRefGlobalString(
//        BaseInstruction->getDebugLoc()->getDirectory(), BaseInstruction);
//    LineNumber = BaseInstruction->getDebugLoc().getLine();
//  }
//  else {
//    FileNameValuePointer = EmptyValuePointer;
//    LineNumber = -1;
//  }
//
//  CGArgs.push_back(createInstructionGlobalString(BaseInstruction));
//  CGArgs.push_back(LeftOpInstructionValuePointer);
//  CGArgs.push_back(RightOpInstructionValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::BinaryInstruction));
//  CGArgs.push_back(ACComputingCallInstruction);
//  CGArgs.push_back(FileNameValuePointer);
//  CGArgs.push_back(InstructionBuilder.getInt32(LineNumber));
//  ArrayRef<Value *> CGArgsRef(CGArgs);
//
//  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(CGCallInstruction && "Invalid call instruction!");
//
//  return;
//}

//void ACInstrumentation::instrumentCallsForNonACIntrinsicFunction(
//    Instruction *BaseInstruction, long *NumInstrumentedInstructions) {
//  assert((CGCreateNode!=nullptr) && "Function not initialized!");
//
//  BasicBlock::iterator NextInst(BaseInstruction);
//  NextInst++;
//  IRBuilder<> InstructionBuilder( &(*NextInst) );
//  std::vector<Value *> Args;
//
//  CallInst *CGCallInstruction = nullptr;
//
//  Value *FileNameValuePointer;
//  int LineNumber;
//  if(BaseInstruction->getDebugLoc()) {
//    std::string FileLocation = BaseInstruction->getDebugLoc()->getDirectory().str() +
//                               "/" +
//                               BaseInstruction->getDebugLoc()->getFilename().str();
//    FileNameValuePointer = createStringRefGlobalString(FileLocation, BaseInstruction);
//    LineNumber = BaseInstruction->getDebugLoc().getLine();
//  }
//  else {
//    FileNameValuePointer = EmptyValuePointer;
//    LineNumber = -1;
//  }
//
//  Args.push_back(createInstructionGlobalString(BaseInstruction));
//  Args.push_back(EmptyValuePointer);
//  Args.push_back(EmptyValuePointer);
//  Args.push_back(InstructionBuilder.getInt32(NodeKind::Register));
//  Args.push_back(FileNameValuePointer);
//  Args.push_back(InstructionBuilder.getInt32(LineNumber));
//  ArrayRef<Value *> ArgsRef(Args);
//
//  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, ArgsRef);
//  *NumInstrumentedInstructions+=1;
//
//  assert(CGCallInstruction && "Invalid call instruction!");
//  return;
//}

// Instruments a call to create a node corresponding to nodes that do not have
// an Atomic condition. This is useful in case there are instructions that use
// these NonAC floating-point instructions.
//void ACInstrumentation::instrumentCallsForNonACFloatPointInstruction(
//    Instruction *BaseInstruction, long *NumInstrumentedInstructions) {
//  assert((CGCreateNode!=nullptr) && "Function not initialized!");
//
//  BasicBlock::iterator NextInst(BaseInstruction);
//  NextInst++;
//  IRBuilder<> InstructionBuilder( &(*NextInst) );
//  std::vector<Value *> Args;
//
//  CallInst *CGCallInstruction = nullptr;
//
//  Value *FileNameValuePointer;
//  int LineNumber;
//  if(BaseInstruction->getDebugLoc()) {
//    std::string FileLocation = BaseInstruction->getDebugLoc()->getDirectory().str() +
//                               "/" +
//                               BaseInstruction->getDebugLoc()->getFilename().str();
//    FileNameValuePointer = createStringRefGlobalString(FileLocation, BaseInstruction);
//    LineNumber = BaseInstruction->getDebugLoc().getLine();
//  }
//  else {
//    FileNameValuePointer = EmptyValuePointer;
//    LineNumber = -1;
//  }
//
//  Value *LeftOpInstructionValuePointer;
//  if(!isa<Constant>(BaseInstruction->getOperand(0)) && !isa<Argument>(BaseInstruction->getOperand(0)))
//    LeftOpInstructionValuePointer = createInstructionGlobalString(static_cast<Instruction*>(BaseInstruction->getOperand(0)));
//  else
//    LeftOpInstructionValuePointer = EmptyValuePointer;
//
//  Args.push_back(createInstructionGlobalString(BaseInstruction));
//  Args.push_back(LeftOpInstructionValuePointer);
//  Args.push_back(EmptyValuePointer);
//  Args.push_back(InstructionBuilder.getInt32(NodeKind::UnaryInstruction));
//  Args.push_back(FileNameValuePointer);
//  Args.push_back(InstructionBuilder.getInt32(LineNumber));
//  ArrayRef<Value *> ArgsRef(Args);
//
//  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, ArgsRef);
//  *NumInstrumentedInstructions+=1;
//
//  assert(CGCallInstruction && "Invalid call instruction!");
//  return;
//}

// Instruments a call to calculate Amplification Factors at the node corresponding
// this instruction.
//void ACInstrumentation::instrumentCallsForAFAnalysis(
//    Instruction *BaseInstruction, Instruction *LocationToInstrument,
//    BasicBlock::iterator *InstructionIterator, long *NumInstrumentedInstructions) {
//  assert((AFAnalysisFunction!=nullptr) && "Function not initialized!");
//
//  // Backtracking till you get a function of interest.
//  // NOTE: Assuming following the usedef chain of just the first operands back
//  // will give the function of interest
////  while(!isInstructionOfInterest(BaseInstruction) &&
////         static_cast<Instruction*>(BaseInstruction->getOperand(0))) {
////    BaseInstruction = static_cast<Instruction*>(BaseInstruction->getOperand(0));
////  }
//
//  // If Instruction is not of interest, ignore and move ahead.
//  if(!isInstructionOfInterest(BaseInstruction))
//    return ;
//
//  BasicBlock::iterator NextInst(LocationToInstrument);
//  NextInst++;
//  IRBuilder<> InstructionBuilder( &(*NextInst) );
//
//  std::vector<Value *> Args;
//  Args.push_back(createInstructionGlobalString(BaseInstruction));
//  ArrayRef<Value *> ArgsRef(Args);
//
//  CallInst *AFComputingCallInstruction = nullptr;
//  AFComputingCallInstruction =
//      InstructionBuilder.CreateCall(AFAnalysisFunction,ArgsRef);
//  *NumInstrumentedInstructions+=1;
//  (*InstructionIterator)++;
//
//  assert(AFComputingCallInstruction && "Invalid call instruction!");
//  return;
//}

// Instrument Amplification Factor Calculating Function in case there is a
// print function or a return call
//void ACInstrumentation::instrumentAFAnalysisForPrintsAndReturns(Instruction *BaseInstruction,
//                                                                BasicBlock::iterator *InstructionIterator,
//                                                                long int *NumInstrumentedInstructions) {
//  if(BaseInstruction->getOpcode() == Instruction::Ret ||
//      BaseInstruction->getOpcode() == Instruction::Call) {
//    if(BaseInstruction->getOpcode() == Instruction::Call) {
//      string FunctionName = "";
//      if(static_cast<CallInst*>(BaseInstruction)->getCalledFunction() &&
//          static_cast<CallInst*>(BaseInstruction)->getCalledFunction()->hasName())
//        FunctionName = static_cast<CallInst*>(BaseInstruction)->getCalledFunction()->getName().str();
//      transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
//      if(FunctionName.find("printf") != std::string::npos){
//        for (Function::op_iterator CurrResultOp = static_cast<CallInst*>(BaseInstruction)->data_operands_begin();
//             CurrResultOp != static_cast<CallInst*>(BaseInstruction)->data_operands_end();
//             ++CurrResultOp) {
//          //            errs() << "Operand: " << *CurrResultOp << "\n";
//          if(CurrResultOp->get()->getType()->isFloatingPointTy())
//            instrumentCallsForAFAnalysis(
//                static_cast<Instruction *>(
//                    resolveToCGCorrespondingInstruction(CurrResultOp->get())),
//                BaseInstruction,
//                InstructionIterator,
//                NumInstrumentedInstructions);
//        }
//      }
//    }
//    else {
//      Instruction* ResultInstruction;
//      if(isa_and_nonnull<Instruction>(static_cast<ReturnInst*>(BaseInstruction)->getReturnValue())) {
//        ResultInstruction =
//            static_cast<Instruction *>(static_cast<ReturnInst*>(BaseInstruction)->getReturnValue());
//        if(ResultInstruction->getType()->isFloatingPointTy())
//          instrumentCallsForAFAnalysis(
//              static_cast<Instruction *>(
//                  resolveToCGCorrespondingInstruction(ResultInstruction)),
//              BaseInstruction,
//              InstructionIterator,
//              NumInstrumentedInstructions);
//      }
//    }
//  }
//
//  return ;
//}

void ACInstrumentation::instrumentBasicBlock(BasicBlock *BB,
                                             long *NumInstrumentedInstructions) {
  if (ACInstrumentation::isUnwantedFunction(BB->getParent()))
    return;

  // Looping through the basic block stepping instruction by instruction. 'I'
  // marks the position to instrument calls. We want to avoid using instrumented
  // calls as base to instrument more calls
  for (BasicBlock::iterator I = BB->begin();
       I != BB->end(); ++I) {
    // The CurrentInstruction is the instruction using which we are
    // instrumenting additional instructions. This has to be updated to the
    // iterators position at the start of the loop.
    Instruction *CurrentInstruction = &*I;

    // Branch based on kind of Instruction
    if(isMemoryLoadOperation(&*I)) {
//      instrumentCallsForMemoryLoadOperation(CurrentInstruction,
//                                            &I,
//                                            NumInstrumentedInstructions);
    }
//    else if(isIntegerToFloatCastOperation(CurrentInstruction)) {
//      instrumentCallsForCastOperation(CurrentInstruction,
//                                      NumInstrumentedInstructions);
//    }
//    // isNonACInstrinsicFunction checks whether the intrinsic function is one from
//    // the list that we do not calculate AC for. Only the CG Node.
//    else if(CurrentInstruction->getOpcode() == Instruction::Call &&
//             isNonACInstrinsicFunction(CurrentInstruction)) {
//      instrumentCallsForNonACIntrinsicFunction(CurrentInstruction,
//                                               NumInstrumentedInstructions);
//    }
    else if(isOtherOperation(&*I)) {
//      instrumentCallsForOtherOperation(CurrentInstruction,
//                                       &I,
//                                       NumInstrumentedInstructions);
      instrumentCallsToAnalyzeInstruction(CurrentInstruction,
                                    &I,
                                    NumInstrumentedInstructions);
    }
    else if(isUnaryOperation(&*I)) {

//      instrumentCallsForUnaryOperation(CurrentInstruction,
//                                       &I,
//                                       NumInstrumentedInstructions);
      instrumentCallsToAnalyzeInstruction(CurrentInstruction,
                                    &I,
                                    NumInstrumentedInstructions);
    }
    else if(isBinaryOperation(&*I)) {
//      instrumentCallsForBinaryOperation(CurrentInstruction,
//                                        &I,
//                                        NumInstrumentedInstructions);
      instrumentCallsToAnalyzeInstruction(CurrentInstruction,
                                    &I,
                                    NumInstrumentedInstructions);
    }
//    else if(isNonACFloatPointInstruction(CurrentInstruction)) {
//      instrumentCallsForNonACFloatPointInstruction(CurrentInstruction,
//                                                   NumInstrumentedInstructions);
//    }

    // CurrentInstruction is updated to the BasicBlock iterators position as the
    // previous if-else ladder may have instrumented some instructions and we
    // want to avoid using the instrumented instructions as base for further
    // instrumentation.
    CurrentInstruction = &*I;

    // Instrument Amplification Factor Calculating Function
//    if(CurrentInstruction->getOpcode() == Instruction::Call) {
//
//      string FunctionName = "";
//      if(static_cast<CallInst*>(CurrentInstruction)->getCalledFunction() &&
//          static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->hasName())
//        FunctionName = static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->getName().str();
//      transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
//      if(FunctionName.find("markforresult") != std::string::npos) {
//        if(!isa<Constant>(static_cast<CallInst*>(CurrentInstruction)->data_operands_begin()->get()) &&
//            static_cast<CallInst*>(static_cast<CallInst*>(CurrentInstruction)->data_operands_begin()->get())->getOpcode() !=
//                Instruction::Load) {
//          instrumentCallsForAFAnalysis(
//              static_cast<Instruction *>(
//                  static_cast<CallInst *>(CurrentInstruction)
//                      ->data_operands_begin()
//                      ->get()), &I,
//              CurrentInstruction, NumInstrumentedInstructions);
//        } else {
//          errs() << "Value to be analyzed has been optimized into a constant\n";
//        }
//      }
//    }

//    instrumentAFAnalysisForPrintsAndReturns(CurrentInstruction,
//                                            &I,
//                                            NumInstrumentedInstructions);
  }

  return;
}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  assert((ACInitFunction!=nullptr) && "Function not initialized!");
  assert((AFInitFunction!= nullptr) && "Function not initialized!");
  assert((ACStoreFunction!=nullptr) && "Function not initialized!");
  assert((AFStoreFunction!= nullptr) && "Function not initialized!");
//  assert((AFPrintTopAmplificationPaths != nullptr) && "Function not initialized!");
//  assert((CGDotGraphFunction!=nullptr) && "Function not initialized!");

  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);
  std::vector<Value *> ACInitCallArgs, CGInitCallArgs, AFInitCallArgs, PrintAFPathsCallArgs;
  std::vector<Value *> ACStoreCallArgs, CGStoreCallArgs, AFStoreCallArgs;
  std::vector<Value *> DotGraphCallArgs;

  CallInst *ACInitCallInstruction, *AFInitCallInstruction, *PrintAFPathsCallInstruction;
  CallInst *StoreACTableCallInstruction, *StoreAFTableCallInstruction;

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
  AFInitCallInstruction = InstructionBuilder.CreateCall(AFInitFunction, AFInitCallArgs);
//  }

  // Instrument call to print table
  for (Function::iterator BBIter=F->begin(); BBIter != F->end(); ++BBIter) {
    for (BasicBlock::iterator InstIter=BBIter->begin(); InstIter != BBIter->end(); ++InstIter) {
      Instruction *CurrentInstruction = &(*InstIter);
      if (isa<ReturnInst>(CurrentInstruction) || isa<ResumeInst>(CurrentInstruction)) {
        ArrayRef<Value *> ACStoreCallArgsRef(ACStoreCallArgs);
        ArrayRef<Value *> AFStoreCallArgsRef(AFStoreCallArgs);
        ArrayRef<Value *> PrintAFPathsCallArgsRef(PrintAFPathsCallArgs);

        InstructionBuilder.SetInsertPoint(CurrentInstruction);
        StoreACTableCallInstruction = InstructionBuilder.CreateCall(ACStoreFunction, ACStoreCallArgsRef);
        StoreAFTableCallInstruction = InstructionBuilder.CreateCall(AFStoreFunction, AFStoreCallArgsRef);
//        PrintAFPathsCallInstruction = InstructionBuilder.CreateCall(AFPrintTopAmplificationPaths, PrintAFPathsCallArgsRef);
      }
    }
  }

  assert(ACInitCallInstruction && AFInitCallInstruction && "Invalid call instruction!");
  assert(StoreACTableCallInstruction && StoreAFTableCallInstruction &&
         PrintAFPathsCallInstruction && "Invalid call instruction!");
  return;
}

//Value *ACInstrumentation::resolveToCGCorrespondingInstruction(Value *Val) {
//  // Check whether Instruction is a constant or an argument. If so, just assign
//  // Empty Value and let program proceed till return.
//  Value *ResolvedInstruction;
//  if(isa<Constant>(Val) ||
//      isa<Argument>(Val))
//    ResolvedInstruction = EmptyValuePointer;
//  else
//    ResolvedInstruction = static_cast<Value*>(Val);
//
//  while(ResolvedInstruction != EmptyValuePointer &&
//         !isa<Constant>(ResolvedInstruction) &&
//         !isa<Argument>(ResolvedInstruction) &&
//         static_cast<Instruction*>(ResolvedInstruction)->getOpcode() != Instruction::PHI &&
//         !canHaveGraphNode(static_cast<Instruction*>(ResolvedInstruction))) {
//    switch (static_cast<Instruction*>(ResolvedInstruction)->getOpcode()) {
//    case Instruction::SIToFP:
//    case Instruction::UIToFP:
//      ResolvedInstruction = EmptyValuePointer;
//      break;
//    case Instruction::FPTrunc:
//    case Instruction::FPExt:
//      ResolvedInstruction = static_cast<Instruction*>(ResolvedInstruction)->getOperand(0);
//      break;
//    }
//
//    // Checks whether the switch-case resulted in Resolved Instruction becoming
//    // a Constant or Argument
//    if(isa<Constant>(ResolvedInstruction) ||
//        isa<Argument>(ResolvedInstruction))
//      ResolvedInstruction = EmptyValuePointer;
//  }
//
//  return ResolvedInstruction;
//}

Value *ACInstrumentation::createArrayInIR(vector<Value*> ArrayOfValues,
                                          IRBuilder<> *InstructionBuilder,
                                          BasicBlock::iterator *InstructionIterator) {
  // Inserting an Alloca Instruction to allocate memory for the array.
  Value *AllocatedArray =
      InstructionBuilder->CreateAlloca(ArrayType::get(ArrayOfValues[0]->getType(),
                                                     ArrayOfValues.size()));
  (*InstructionIterator)++;

  // Looping through operands of BaseInstruction
  for (long unsigned int I = 0; I < ArrayOfValues.size(); ++I) {
    // Setting Value in OperandNameArray
    Value *LocationInArray = InstructionBuilder->CreateGEP(ArrayOfValues[0]->getType(),
                                                            AllocatedArray,
                                                            InstructionBuilder->getInt32(I));
    (*InstructionIterator)++;
    InstructionBuilder->CreateStore(ArrayOfValues[I], LocationInArray);
    (*InstructionIterator)++;
  }

  // Get pointers to arrays
  Value *Array = InstructionBuilder->CreateGEP(ArrayOfValues[0]->getType(),
                                               AllocatedArray,
                                               InstructionBuilder->getInt32(0));
  (*InstructionIterator)++;

  return Array;
}

// Get Instruction after any Phi/Dbg instructions AND Atomic Condition and
// Computation Graph Initialization Calls.
Instruction *
ACInstrumentation::getInstructionAfterInitializationCalls(BasicBlock *BB) {
  BasicBlock::iterator NextInst(BB->getFirstNonPHIOrDbg());

  // Iterating through instructions till we skip any Initialization calls.
  while(NextInst->getOpcode() == 56 &&
         (!static_cast<CallInst*>(&*NextInst)->getCalledFunction() ||
          (static_cast<CallInst*>(&*NextInst)->getCalledFunction() &&
           (static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() == "fACCreate" ||
            static_cast<CallInst*>(&*NextInst)->getCalledFunction()->getName().str() == "fCGInitialize"))))
    NextInst++;

  return &(*NextInst);
}

bool ACInstrumentation::canHaveGraphNode(const Instruction *Inst) {
  return isMemoryLoadOperation(Inst) ||
         isUnaryOperation(Inst) ||
         isBinaryOperation(Inst) ||
         isOtherOperation(Inst);
}

bool ACInstrumentation::isMemoryLoadOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::Alloca ||
         Inst->getOpcode() == Instruction::Load;
}

bool ACInstrumentation::isIntegerToFloatCastOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::UIToFP ||
         Inst->getOpcode() == Instruction::SIToFP;
}

bool ACInstrumentation::isUnaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FNeg ||
         (Inst->getOpcode() == Instruction::Call &&
          (static_cast<const CallInst*>(Inst)->getCalledFunction() &&
           isFunctionOfInterest(
              static_cast<const CallInst*>(Inst)->getCalledFunction())));
}

// TODO: Check for FRem case.
bool ACInstrumentation::isBinaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FAdd ||
         Inst->getOpcode() == Instruction::FSub ||
         Inst->getOpcode() == Instruction::FMul ||
         Inst->getOpcode() == Instruction::FDiv;
}

bool ACInstrumentation::isOtherOperation(const Instruction *Inst) {
  return (Inst->getOpcode() == Instruction::Call &&
          static_cast<const CallInst*>(Inst)->getCalledFunction() &&
          static_cast<const CallInst*>(Inst)->getCalledFunction()->hasName() &&
          (static_cast<const CallInst*>(Inst)->getCalledFunction()->getName().str().find("llvm.fmuladd") !=
              std::string::npos ||
          static_cast<const CallInst*>(Inst)->getCalledFunction()->getName().str().find("llvm.fma") !=
             std::string::npos));
}

bool ACInstrumentation::isNonACInstrinsicFunction(const Instruction *Inst) {
  assert(Inst->getOpcode() == Instruction::Call);
  if(static_cast<const CallInst*>(Inst)->getCalledFunction() &&
      static_cast<const CallInst*>(Inst)->getCalledFunction()->hasName())
    // Change this return to true when you have some function name that you check.
    return false;

  return false;
}

bool ACInstrumentation::isNonACFloatPointInstruction(const Instruction *Inst) {
  return false;
}

bool ACInstrumentation::isSingleFPOperation(const Instruction *Inst) {
  if (isUnaryOperation(Inst)) {
    switch (Inst->getOpcode()) {
    case 12:
      return Inst->getOperand(0)->getType()->isFloatTy();
    case 46:
      return static_cast<const FPExtInst*>(Inst)->getSrcTy()->isFloatTy();
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst*>(Inst)->getArgOperand(0)->getType()->isFloatTy();
    default:
//      errs() << "Not an FP32 operation.\n";
      break;
    }
  } else if(isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isFloatTy() &&
           Inst->getOperand(1)->getType()->isFloatTy();
  } else if(Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode*>(Inst)->getType()->isFloatTy();
  }

  return false;
}

bool ACInstrumentation::isDoubleFPOperation(const Instruction *Inst) {
  if (isUnaryOperation(Inst)) {
    switch (Inst->getOpcode()) {
    case 12:
      return Inst->getOperand(0)->getType()->isDoubleTy();
    case 45:
      return static_cast<const FPTruncInst*>(Inst)->getSrcTy()->isDoubleTy();
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst*>(Inst)->getArgOperand(0)->getType()->isDoubleTy();
    default:
//      errs() << "Not an FP64 operation.\n";
      break;
    }
  } else if(isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isDoubleTy() &&
           Inst->getOperand(1)->getType()->isDoubleTy();
  } else if(Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode*>(Inst)->getType()->isDoubleTy();
  }

  return false;
}

bool ACInstrumentation::isUnwantedFunction(const Function *Func) {
  assert(Func->hasName());
  return Func->getName().str().find("fAC") != std::string::npos ||
         Func->getName().str().find("fCG") != std::string::npos ||
         Func->getName().str().find("fAF") != std::string::npos ||
         Func->getName().str().find("fRS") != std::string::npos ||
         Func->getName().str().find("fURT") != std::string::npos ||
         Func->getName().str().find("ACItem") != std::string::npos;
}

bool ACInstrumentation::isFunctionOfInterest(const Function *Func) {
  if(Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return FunctionName.find("sin") != std::string::npos ||
           FunctionName.find("cos") != std::string::npos ||
           FunctionName.find("tan") != std::string::npos ||
           FunctionName.find("exp") != std::string::npos ||
           FunctionName.find("log") != std::string::npos ||
           FunctionName.find("sqrt") != std::string::npos;
  }
  return false;
}

Value *ACInstrumentation::createBBNameGlobalString(BasicBlock *BB) {
  string BasicBlockString;
  raw_string_ostream RawBasicBlockString(BasicBlockString);
  BB->printAsOperand(RawBasicBlockString, false);
  Constant *BBValue = ConstantDataArray::getString(BB->getModule()->getContext(),
                                                   RawBasicBlockString.str().c_str(),
                                                   true);

  Value *BBValuePointer = new GlobalVariable(*BB->getModule(),
                                             BBValue->getType(),
                                             true,
                                             GlobalValue::InternalLinkage,
                                             BBValue);

  return BBValuePointer;
}

Value *ACInstrumentation::createRegisterNameGlobalString(Instruction *Inst) {
  string RegisterString;
  raw_string_ostream RawRegisterString(RegisterString);
  Inst->printAsOperand(RawRegisterString, false);

  Constant *RegisterValue = ConstantDataArray::getString(Inst->getModule()->getContext(),
                                                            RawRegisterString.str().c_str(),
                                                            true);

  Value *RegisterValuePointer = new GlobalVariable(*Inst->getModule(),
                                                   RegisterValue->getType(),
                                                   true,
                                                   GlobalValue::InternalLinkage,
                                                   RegisterValue);

  return RegisterValuePointer;
}

Value *ACInstrumentation::createInstructionGlobalString(Instruction *Inst) {
  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *Inst;
  unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  string Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                               RawInstructionString.str().substr(NonEmptyPosition);

  Constant *InstructionValue = ConstantDataArray::getString(Inst->getModule()->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*Inst->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);

  return InstructionValuePointer;
}

Value *ACInstrumentation::createStringRefGlobalString(StringRef StringObj, Instruction *Inst) {
  string StringRefString;
  raw_string_ostream RawInstructionString(StringRefString);
  RawInstructionString << StringObj;
  unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  string Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                               RawInstructionString.str().substr(NonEmptyPosition);

  Constant *StringObjValue = ConstantDataArray::getString(Inst->getModule()->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *StringObjValuePointer = new GlobalVariable(*Inst->getModule(),
                                                      StringObjValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      StringObjValue);

  return StringObjValuePointer;
}

string ACInstrumentation::getInstructionAsString(Instruction *Inst) {
  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *Inst;
  unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  string InstructionAsString = (NonEmptyPosition == std::string::npos) ? "" :
                                                               RawInstructionString.str().substr(NonEmptyPosition);
  return InstructionAsString;
}

bool ACInstrumentation::isInstructionOfInterest(Instruction *Inst) {
  switch (Inst->getOpcode()) {
  case 14:
  case 16:
  case 18:
  case 21:
  case 31:
  case 32:
  case 55:
  case 56:
    return true;
  }
  return false;
}

int ACInstrumentation::getFunctionEnum(Instruction *Inst) {
  string FunctionName = "";

  switch (Inst->getOpcode()) {
  case 12:
    return Func::Neg;
  case 14:
    return Func::Add;
  case 16:
    return Func::Sub;
  case 18:
    return Func::Mul;
  case 21:
    return Func::Div;
  case 56:
    if(static_cast<CallInst*>(Inst)->getCalledFunction() &&
        static_cast<CallInst*>(Inst)->getCalledFunction()->hasName())
      FunctionName = static_cast<CallInst*>(Inst)->getCalledFunction()->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
    if (isASinFunction(FunctionName))
      return Func::ArcSin;
    if(isACosFunction(FunctionName))
      return Func::ArcCos;
    if(isATanFunction(FunctionName))
      return Func::ArcTan;
    if(isSinhFunction(FunctionName))
      return Func::Sinh;
    if(isCoshFunction(FunctionName))
      return Func::Cosh;
    if(isTanhFunction(FunctionName))
      return Func::Tanh;
    if(isSinFunction(FunctionName))
      return Func::Sin;
    if(isCosFunction(FunctionName))
      return Func::Cos;
    if(isTanFunction(FunctionName))
      return Func::Tan;
    if(isExpFunction(FunctionName))
      return Func::Exp;
    if(isLogFunction(FunctionName))
      return Func::Log;
    if(isSqrtFunction(FunctionName))
      return Func::Sqrt;
    if(isFMAFuncton(FunctionName))
      return Func::FMA;
    return -1;
  default:
    //    errs() << BaseInstruction << " is not a Binary Instruction.\n";
    return -1;
  }
}
