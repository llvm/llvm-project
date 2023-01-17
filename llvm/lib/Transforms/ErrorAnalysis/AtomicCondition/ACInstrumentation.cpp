#include <string>
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/Utilities/Utilities.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/DebugInfo.h"

using namespace llvm;
using namespace atomiccondition;
using namespace std;

namespace atomiccondition {

void confFunction(Function *FunctionToSave, Function **StorageLocation,
                  GlobalValue::LinkageTypes LinkageType) {
  // Save the function pointer
  if (StorageLocation != nullptr)
    *StorageLocation = FunctionToSave;
  if (FunctionToSave->getLinkage() != LinkageType)
    FunctionToSave->setLinkage(LinkageType);
}

// Searches module for functions to mark and creates pointers for them.
ACInstrumentation::ACInstrumentation(Function *InstrumentFunction)
    : FunctionToInstrument(InstrumentFunction), AFInitFunction(nullptr),
      ACComputingFunction(nullptr), AFComputingFunction(nullptr),
      ACStoreFunction(nullptr), AFStoreFunction(nullptr),
      AFPrintTopAmplificationPaths(nullptr) {
  // Find and configure instrumentation functions
  Module *M = FunctionToInstrument->getParent();

  // Configuring all runtime functions and saving pointers.
  for (Module::iterator F = M->begin(); F != M->end(); ++F) {
    Function *CurrentFunction = &*F;

    // Only configuring functions with certain prefixes
    if (!CurrentFunction->hasName()) {

    } else if (CurrentFunction->getName().str().find("fAFInitialize") !=
               std::string::npos) {
      confFunction(CurrentFunction, &AFInitFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    } else if (CurrentFunction->getName().str().find("fACComputeAC") !=
               std::string::npos) {
      confFunction(CurrentFunction, &ACComputingFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    } else if (CurrentFunction->getName().str().find("fAFComputeAF") !=
               std::string::npos) {
      confFunction(CurrentFunction, &AFComputingFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    } else if (CurrentFunction->getName().str().find("fACStoreACs") !=
               std::string::npos) {
      confFunction(CurrentFunction, &ACStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    } else if (CurrentFunction->getName().str().find("fAFStoreAFs") !=
               std::string::npos) {
      confFunction(CurrentFunction, &AFStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    } else if (CurrentFunction->getName().str().find("fAFStoreInFile") !=
               std::string::npos) {
      confFunction(CurrentFunction, &AFStoreInFile,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    //    else if (CurrentFunction->getName().str().find("fAFPrintTopAmplificationPaths") != std::string::npos) {
    else if (CurrentFunction->getName().str().find(
                 "fAFPrintTopFromAllAmplificationPaths") != std::string::npos) {
      confFunction(CurrentFunction, &AFPrintTopAmplificationPaths,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
  }

  Constant *EmptyValue =
      ConstantDataArray::getString(InstrumentFunction->getContext(), "", true);

  EmptyValuePointer = new GlobalVariable(
      *InstrumentFunction->getParent(), EmptyValue->getType(), true,
      GlobalValue::InternalLinkage, EmptyValue);
}

// Instruments calls analyzing sensitivity of this instruction
void ACInstrumentation::instrumentCallsToAnalyzeInstruction(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  // Return if we cannot analyze this function
  int F = getFunctionEnum(BaseInstruction);
  if (F == -1)
    return;

  instrumentCallsForACComputation(BaseInstruction, InstructionIterator,
                                  NumInstrumentedInstructions, F);

  instrumentCallsForAFComputation(BaseInstruction, InstructionIterator,
                                  NumInstrumentedInstructions);
}

// Instruments a call to calculate atomic condition
void ACInstrumentation::instrumentCallsForACComputation(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions, int FunctionType) {
  // Ensuring ACComputingFunction is linked
  assert((ACComputingFunction != nullptr) && "Function not initialized!");
  assert((FunctionType != -1) && "Function cannot be analyzed");

  // Positioning the instruction builder
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> ACArgs;
  int NumOperands = (int)BaseInstruction->getNumOperands();
  if (BaseInstruction->getOpcode() == Instruction::Call &&
      static_cast<CallInst *>(BaseInstruction)
          ->getCalledFunction()
          ->isIntrinsic())
    NumOperands--;

  // Creating a Global string for the Register the Result of this instruction is
  // stored in.
  Value *ResultNamePointer = createRegisterNameGlobalString(
      static_cast<Instruction *>(BaseInstruction));

  std::vector<Value *> OpRegisterNamesArray;
  std::vector<Value *> OperandValuesArray;

  // Looping through operands of BaseInstruction
  for (int I = 0; I < NumOperands; ++I) {
    // Creating a Global string representing the register name of operand if
    // if is not a constant.
    Value *OpRegisterNamePointer;
    if (!(isa<Constant>(BaseInstruction->getOperand(I)) ||
          isa<Argument>(BaseInstruction->getOperand(I))))
      OpRegisterNamePointer = createRegisterNameGlobalString(
          static_cast<Instruction *>(BaseInstruction->getOperand(I)));
    else
      OpRegisterNamePointer = EmptyValuePointer;


    OpRegisterNamesArray.push_back(OpRegisterNamePointer);

    // TODO: If a constant if a float, manually insert a double of the same
    //  constant.
    Value *OperandValue = BaseInstruction->getOperand(I);
    // Create a Cast Instruction to double in case operation is float operation.
    if (OpRegisterNamePointer != EmptyValuePointer &&
        isSingleFPOperation(&*BaseInstruction)) {
      OperandValue = InstructionBuilder.CreateFPCast(
          BaseInstruction->getOperand(I),
          Type::getDoubleTy(BaseInstruction->getModule()->getContext()));

      (*InstructionIterator)++;
      (*NumInstrumentedInstructions)++;
    }

    OperandValuesArray.push_back(OperandValue);
  }

  Value *AllocatedOpRegisterNamesArray = createArrayInIR(
      OpRegisterNamesArray, &InstructionBuilder, InstructionIterator);

  Value *AllocatedOperandValuesArray = createArrayInIR(
      OperandValuesArray, &InstructionBuilder, InstructionIterator);

  Value *FileNameValuePointer;
  int LineNumber;
  if (BaseInstruction->getDebugLoc()) {
    std::string FileLocation =
        BaseInstruction->getDebugLoc()->getDirectory().str() + "/" +
        BaseInstruction->getDebugLoc()->getFilename().str();
    FileNameValuePointer =
        createStringRefGlobalString(FileLocation, BaseInstruction);
    LineNumber = BaseInstruction->getDebugLoc().getLine();
  } else {
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
      InstructionBuilder.CreateCall(ACComputingFunction, ACArgsRef);

  // 2*(Get Location + Store Value at Location) * NumOperands +
  // 2*(Allocate for Array + GetArrayLocation) + ACComputation Function Call
  (*NumInstrumentedInstructions) += 4 * NumOperands + 5;
  (*InstructionIterator)++;

  std::pair<Value *, Value *> InstructionACPair =
      std::make_pair(BaseInstruction, ACComputingCallInstruction);
  InstructionACMap.insert(InstructionACPair);

  assert(ACComputingCallInstruction && "Invalid call instruction!");
  return;
}

void ACInstrumentation::instrumentCallsForAFComputation(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  // Ensuring AFComputingFunction is linked
  assert((ACComputingFunction != nullptr) && "Function not initialized!");
  assert((getFunctionEnum(BaseInstruction) != -1) &&
         "Function cannot be analyzed");

  // Positioning the instruction builder
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> AFArgs;
  CallInst *AFComputingCallInstruction;

  std::vector<Value *> AFArray;

  int NumOperands = (int)BaseInstruction->getNumOperands();
//  if (BaseInstruction->getOpcode() == Instruction::Call &&
//      static_cast<CallInst *>(BaseInstruction)
//          ->getCalledFunction()
//          ->isIntrinsic())
  if (BaseInstruction->getOpcode() == Instruction::Call)
    NumOperands--;

  // Incomplete phi nodes are those that are to be assigned an incoming value
  // which is not generated yet.
  std::vector<Value *> IncompletePHINodes;
  for (int I = 0; I < NumOperands; ++I) {
    if (getFunctionEnum(
            static_cast<Instruction *>(BaseInstruction->getOperand(I))) != -1) {
      assert(InstructionAFMap.count(BaseInstruction->getOperand(I)) == 1);
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
      IncompletePHINodes.push_back(NULL);
    } else if (isFloatToFloatCastOperation(static_cast<const Instruction *>(
                   BaseInstruction->getOperand(I))) &&
               InstructionAFMap.count(BaseInstruction->getOperand(I)) == 1) {
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
      IncompletePHINodes.push_back(NULL);
    } else if (isa<Instruction>(BaseInstruction->getOperand(I)) &&
               static_cast<Instruction *>(BaseInstruction->getOperand(I))
                       ->getOpcode() == Instruction::PHI &&
               InstructionAFMap.count(BaseInstruction->getOperand(I)) == 0) {
      // Case when operand is a phi node.
      IncompletePHINodes.push_back(instrumentPhiNodeForAF(
          BaseInstruction->getOperand(I), NumInstrumentedInstructions));
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
    } else {
      // Case for when the operand cannot have an associated AF Object.
      // eg: When operand is constant.
      AFArray.push_back(
          ConstantPointerNull::get(InstructionBuilder.getPtrTy()));
      IncompletePHINodes.push_back(NULL);
    }
  }

  Value *AllocatedAFArray =
      createArrayInIR(AFArray, &InstructionBuilder, InstructionIterator);

  AFArgs.push_back(InstructionACMap[BaseInstruction]);
  AFArgs.push_back(AllocatedAFArray);
  AFArgs.push_back(InstructionBuilder.getInt32(NumOperands));

  // Creating a call to AFComputingFunction with above parameters
  ArrayRef<Value *> AFArgsRef(AFArgs);
  AFComputingCallInstruction =
      InstructionBuilder.CreateCall(AFComputingFunction, AFArgsRef);

  // Setting IncomingValue for the BasicBlock-IncomingValue pair coming from the
  // Current BasicBlock.
  for (int I = 0; I < NumOperands; ++I) {
    if (IncompletePHINodes[I] != NULL &&
        static_cast<PHINode *>(IncompletePHINodes[I])
                ->getBasicBlockIndex(AFComputingCallInstruction->getParent()) !=
            -1)
      static_cast<PHINode *>(IncompletePHINodes[I])
          ->setIncomingValueForBlock(AFComputingCallInstruction->getParent(),
                                     AFComputingCallInstruction);
  }

  // (Get Location + Store Value at Location) * NumOperands +
  // (Allocate for Array + GetArrayLocation) + AFComputation Function Call
  *NumInstrumentedInstructions += 2 * NumOperands + 3;
  (*InstructionIterator)++;

  std::pair<Value *, Value *> InstructionAFPair =
      std::make_pair(BaseInstruction, AFComputingCallInstruction);
  InstructionAFMap.insert(InstructionAFPair);

  assert(AFComputingCallInstruction && "Invalid call instruction!");
  return;
}

// Instrumenting a PhiNode for propagating AFItem
Value *
ACInstrumentation::instrumentPhiNodeForAF(Value *OriginalPHI,
                                          long *NumInstrumentedInstructions) {
  assert(isa<Instruction>(*OriginalPHI) &&
         static_cast<Instruction *>(OriginalPHI)->getOpcode() ==
             Instruction::PHI);

  IRBuilder<> InstructionBuilder(
      static_cast<Instruction *>(OriginalPHI)->getParent()->getFirstNonPHI());

  Value *AFPhi = InstructionBuilder.CreatePHI(
      InstructionBuilder.getPtrTy(),
      static_cast<PHINode *>(OriginalPHI)->getNumIncomingValues(), "Fi");

  // Looping through the Incoming Values and setting the IncomingBlocks for the new PhiNode
  for (unsigned int I = 0;
       I < static_cast<PHINode *>(OriginalPHI)->getNumIncomingValues(); ++I) {
    BasicBlock *IncomingBlock =
        static_cast<PHINode *>(OriginalPHI)->getIncomingBlock(I);
    Value *IncomingValue =
        static_cast<PHINode *>(OriginalPHI)->getIncomingValue(I);

    if (InstructionAFMap.count(IncomingValue) == 1)
      static_cast<PHINode *>(AFPhi)->addIncoming(
          InstructionAFMap[IncomingValue], IncomingBlock);
    else
      static_cast<PHINode *>(AFPhi)->addIncoming(
          ConstantPointerNull::get(InstructionBuilder.getPtrTy()),
          IncomingBlock);
  }

  (*NumInstrumentedInstructions)++;

  std::pair<Value *, Value *> InstructionAFPair =
      std::make_pair(OriginalPHI, AFPhi);
  InstructionAFMap.insert(InstructionAFPair);

  return AFPhi;
}

// Instrumenting a Select Instruction for propagating AFItem
Value *ACInstrumentation::instrumentSelectForAF(
    Value *OriginalSelInstr, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  assert(isa<Instruction>(*OriginalSelInstr) &&
         static_cast<Instruction *>(OriginalSelInstr)->getOpcode() ==
             Instruction::Select);

  Value *TrueValue = static_cast<SelectInst *>(OriginalSelInstr)->getTrueValue();
  Value *FalseValue = static_cast<SelectInst *>(OriginalSelInstr)->getFalseValue();

  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());

  // Setting the AF Values to propagate for the True and False cases.
  Value *TrueAFValue, *FalseAFValue;
  if (getFunctionEnum(static_cast<Instruction *>(TrueValue)) != -1) {
    assert(InstructionAFMap.count(TrueValue) == 1  &&
           "InstructionAFMap does not contain value for this key.");
    TrueAFValue = InstructionAFMap[TrueValue];
  } else if(static_cast<Instruction*>(TrueValue)->getOpcode() == Instruction::PHI &&
        InstructionAFMap.count(TrueValue) == 0) {
    instrumentPhiNodeForAF(TrueValue, NumInstrumentedInstructions);
    assert(InstructionAFMap.count(TrueValue) == 1  &&
           "InstructionAFMap does not contain value for this key.");
    TrueAFValue = InstructionAFMap[TrueValue];
  }
  else
    TrueAFValue = ConstantPointerNull::get(InstructionBuilder.getPtrTy());

  if (getFunctionEnum(static_cast<Instruction *>(FalseValue)) != -1) {
    assert(InstructionAFMap.count(FalseValue) == 1  &&
           "InstructionAFMap does not contain value for this key.");
    FalseAFValue = InstructionAFMap[FalseValue];
  } else if(static_cast<Instruction*>(FalseValue)->getOpcode() == Instruction::PHI &&
        InstructionAFMap.count(FalseValue) == 0) {
    instrumentPhiNodeForAF(FalseValue, NumInstrumentedInstructions);
    assert(InstructionAFMap.count(FalseValue) == 1  &&
           "InstructionAFMap does not contain value for this key.");
    FalseAFValue = InstructionAFMap[FalseValue];
  }
  else
    FalseAFValue = ConstantPointerNull::get(InstructionBuilder.getPtrTy());

  // Instrumenting Select instruction, incrementing instrumented instructions
  // and incrementing insert pointer.
  Value *AFSel = InstructionBuilder.CreateSelect(
      static_cast<SelectInst *>(OriginalSelInstr)->getCondition(), TrueAFValue,
      FalseAFValue);
  (*NumInstrumentedInstructions)++;
  (*InstructionIterator)++;

  return AFSel;
}

void ACInstrumentation::instrumentForMarkedVariable(
    Value *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions) {
  assert((AFStoreInFile != nullptr) && "Function not initialized!");

  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> Args;

  Args.push_back(InstructionAFMap[BaseInstruction]);

  // Creating a call to AFStoreInFile function with above parameters
  ArrayRef<Value *> ArgsRef(Args);
  Value *StoreInFileCall =
      InstructionBuilder.CreateCall(AFStoreInFile, ArgsRef);

  assert((StoreInFileCall != nullptr) && "Function not initialized!");
}

void ACInstrumentation::instrumentBasicBlock(
    BasicBlock *BB, long *NumInstrumentedInstructions) {
  if (ACInstrumentation::isUnwantedFunction(BB->getParent()))
    return;

  // Looping through the basic block stepping instruction by instruction. 'I'
  // marks the position to instrument calls. We want to avoid using instrumented
  // calls as base to instrument more calls
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    // The CurrentInstruction is the instruction using which we are
    // instrumenting additional instructions. This has to be updated to the
    // iterators position at the start of the loop.
    Instruction *CurrentInstruction = &*I;

    // Branch based on kind of Instruction
    if (isFloatToFloatCastOperation(&*I)) {
      mapFloatCastToAFValue(&*I);
    } else if (isOtherOperation(&*I)) {
      instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                            NumInstrumentedInstructions);
    } else if (isUnaryOperation(&*I)) {
      instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                          NumInstrumentedInstructions);
    } else if (isBinaryOperation(&*I)) {
      instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                          NumInstrumentedInstructions);
    } else if ((&*I)->getOpcode() == Instruction::Select) {
      std::pair<Value *, Value *> InstructionAFPair =
          std::make_pair(CurrentInstruction,
                         instrumentSelectForAF(CurrentInstruction, &I,
                                               NumInstrumentedInstructions));
      InstructionAFMap.insert(InstructionAFPair);
    }

    // CurrentInstruction is updated to the BasicBlock iterators position as the
    // previous if-else ladder may have instrumented some instructions and we
    // want to avoid using the instrumented instructions as base for further
    // instrumentation.
    CurrentInstruction = &*I;

    if(CurrentInstruction->getOpcode() == Instruction::Call) {
      string FunctionName = "";
      if(static_cast<CallInst*>(CurrentInstruction)->getCalledFunction() &&
          static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->hasName())
        FunctionName = static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->getName().str();
      transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
      if(FunctionName.find("markforresult") != std::string::npos) {
        if(!isa<Constant>(static_cast<CallInst*>(CurrentInstruction)->data_operands_begin()->get()) &&
            static_cast<CallInst*>(static_cast<CallInst*>(CurrentInstruction)->data_operands_begin()->get())->getOpcode() !=
                Instruction::Load) {
          instrumentForMarkedVariable(static_cast<CallInst *>(CurrentInstruction)
                                          ->data_operands_begin()
                                          ->get(), &I,
                                      NumInstrumentedInstructions);
        } else {
          errs() << "Value to be analyzed has been optimized into a constant\n";
        }
      }
    }
  }

  return;
}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  assert((AFInitFunction != nullptr) && "Function not initialized!");
  assert((ACStoreFunction != nullptr) && "Function not initialized!");
  assert((AFStoreFunction != nullptr) && "Function not initialized!");
  assert((AFPrintTopAmplificationPaths != nullptr) &&
         "Function not initialized!");

  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);
  std::vector<Value *> ACInitCallArgs, AFInitCallArgs, PrintAFPathsCallArgs;
  std::vector<Value *> ACStoreCallArgs, AFStoreCallArgs;

  CallInst *AFInitCallInstruction, *PrintAFPathsCallInstruction;
  CallInst *StoreACTableCallInstruction, *StoreAFTableCallInstruction;

  // Instrumenting Initialization call instruction
  AFInitCallInstruction =
      InstructionBuilder.CreateCall(AFInitFunction, AFInitCallArgs);

  // Instrument call to print table
  for (Function::iterator BBIter = F->begin(); BBIter != F->end(); ++BBIter) {
    for (BasicBlock::iterator InstIter = BBIter->begin();
         InstIter != BBIter->end(); ++InstIter) {
      Instruction *CurrentInstruction = &(*InstIter);
      if (isa<ReturnInst>(CurrentInstruction) ||
          isa<ResumeInst>(CurrentInstruction)) {
        ArrayRef<Value *> ACStoreCallArgsRef(ACStoreCallArgs);
        ArrayRef<Value *> AFStoreCallArgsRef(AFStoreCallArgs);
        ArrayRef<Value *> PrintAFPathsCallArgsRef(PrintAFPathsCallArgs);

        InstructionBuilder.SetInsertPoint(CurrentInstruction);
        StoreACTableCallInstruction =
            InstructionBuilder.CreateCall(ACStoreFunction, ACStoreCallArgsRef);
        StoreAFTableCallInstruction =
            InstructionBuilder.CreateCall(AFStoreFunction, AFStoreCallArgsRef);
        PrintAFPathsCallInstruction = InstructionBuilder.CreateCall(
            AFPrintTopAmplificationPaths, PrintAFPathsCallArgsRef);
      }
    }
  }

  assert(AFInitCallInstruction && "Invalid call instruction!");
  assert(StoreACTableCallInstruction && StoreAFTableCallInstruction &&
         PrintAFPathsCallInstruction && "Invalid call instruction!");
  return;
}

Value *
ACInstrumentation::createArrayInIR(vector<Value *> ArrayOfValues,
                                   IRBuilder<> *InstructionBuilder,
                                   BasicBlock::iterator *InstructionIterator) {
  // Inserting an Alloca Instruction to allocate memory for the array.
  Value *AllocatedArray = InstructionBuilder->CreateAlloca(
      ArrayType::get(InstructionBuilder->getPtrTy(), ArrayOfValues.size()));
  (*InstructionIterator)++;

  // Looping through operands of BaseInstruction
  for (long unsigned int I = 0; I < ArrayOfValues.size(); ++I) {
    // Setting Value in OperandNameArray
    Value *LocationInArray = InstructionBuilder->CreateGEP(
        InstructionBuilder->getPtrTy(), AllocatedArray,
        InstructionBuilder->getInt32(I));
    (*InstructionIterator)++;
    InstructionBuilder->CreateStore(ArrayOfValues[I], LocationInArray);
    (*InstructionIterator)++;
  }

  // Get pointers to arrays
  Value *Array = InstructionBuilder->CreateGEP(InstructionBuilder->getPtrTy(),
                                               AllocatedArray,
                                               InstructionBuilder->getInt32(0));
  (*InstructionIterator)++;

  return Array;
}

bool ACInstrumentation::canHaveGraphNode(const Instruction *Inst) {
  return isMemoryLoadOperation(Inst) || isUnaryOperation(Inst) ||
         isBinaryOperation(Inst) || isOtherOperation(Inst);
}

bool ACInstrumentation::isPhiNode(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::PHI;
}

bool ACInstrumentation::isMemoryLoadOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::Alloca ||
         Inst->getOpcode() == Instruction::Load;
}

bool ACInstrumentation::isIntegerToFloatCastOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::UIToFP ||
         Inst->getOpcode() == Instruction::SIToFP;
}

bool ACInstrumentation::isFloatToFloatCastOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FPTrunc ||
         Inst->getOpcode() == Instruction::FPExt;
}

bool ACInstrumentation::isUnaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FNeg ||
         (Inst->getOpcode() == Instruction::Call &&
          (static_cast<const CallInst *>(Inst)->getCalledFunction() &&
           isFunctionOfInterest(
               static_cast<const CallInst *>(Inst)->getCalledFunction())));
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
          static_cast<const CallInst *>(Inst)->getCalledFunction() &&
          static_cast<const CallInst *>(Inst)->getCalledFunction()->hasName() &&
          (static_cast<const CallInst *>(Inst)
                   ->getCalledFunction()
                   ->getName()
                   .str()
                   .find("llvm.fmuladd") != std::string::npos ||
           static_cast<const CallInst *>(Inst)
                   ->getCalledFunction()
                   ->getName()
                   .str()
                   .find("llvm.fma") != std::string::npos));
}

bool ACInstrumentation::isNonACInstrinsicFunction(const Instruction *Inst) {
  assert(Inst->getOpcode() == Instruction::Call);
  if (static_cast<const CallInst *>(Inst)->getCalledFunction() &&
      static_cast<const CallInst *>(Inst)->getCalledFunction()->hasName())
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
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst *>(Inst)
          ->getArgOperand(0)
          ->getType()
          ->isFloatTy();
    default:
      //      errs() << "Not an FP32 operation.\n";
      break;
    }
  } else if (isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isFloatTy() &&
           Inst->getOperand(1)->getType()->isFloatTy();
  } else if (Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode *>(Inst)->getType()->isFloatTy();
  }

  return false;
}

bool ACInstrumentation::isDoubleFPOperation(const Instruction *Inst) {
  if (isUnaryOperation(Inst)) {
    switch (Inst->getOpcode()) {
    case 12:
      return Inst->getOperand(0)->getType()->isDoubleTy();
    case 56:
      // Assuming that operand 0 for this call instruction contains the operand
      // used to calculate the AC.
      return static_cast<const CallInst *>(Inst)
          ->getArgOperand(0)
          ->getType()
          ->isDoubleTy();
    default:
      //      errs() << "Not an FP64 operation.\n";
      break;
    }
  } else if (isBinaryOperation(Inst)) {
    return Inst->getOperand(0)->getType()->isDoubleTy() &&
           Inst->getOperand(1)->getType()->isDoubleTy();
  } else if (Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode *>(Inst)->getType()->isDoubleTy();
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
  if (Func->hasName()) {
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

void ACInstrumentation::mapFloatCastToAFValue(Instruction *Inst) {
  if (Inst->getOpcode() == Instruction::FPTrunc &&
      static_cast<const FPTruncInst *>(Inst)->getSrcTy()->isDoubleTy() &&
      InstructionAFMap.count(static_cast<const FPTruncInst *>(Inst)->getOperand(0)) == 1) {
    std::pair<Value *, Value *> InstructionAFPair =
        std::make_pair((Value *)Inst, InstructionAFMap[static_cast<const FPTruncInst *>(Inst)->getOperand(0)]);
    InstructionAFMap.insert(InstructionAFPair);
  } else if(Inst->getOpcode() == Instruction::FPExt &&
             static_cast<const FPExtInst *>(Inst)->getSrcTy()->isDoubleTy() &&
             InstructionAFMap.count(static_cast<const FPExtInst *>(Inst)->getOperand(0)) == 1) {
    std::pair<Value *, Value *> InstructionAFPair =
        std::make_pair((Value *)Inst, InstructionAFMap[static_cast<const FPExtInst *>(Inst)->getOperand(0)]);
    InstructionAFMap.insert(InstructionAFPair);
  }
}

}  // namespace atomiccondition
