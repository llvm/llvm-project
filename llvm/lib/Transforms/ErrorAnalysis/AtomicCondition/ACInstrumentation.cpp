#include <string>
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/Utilities/CompileTimeUtilities.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/DebugInfo.h"

using namespace llvm;
using namespace atomiccondition;
using namespace std;

namespace atomiccondition {

// Global
int InstructionID = 0;

void confFunction(Function *FunctionToSave, Function **StorageLocation,
                  GlobalValue::LinkageTypes LinkageType) {
  // Save the function pointer
  if (StorageLocation != nullptr)
    *StorageLocation = FunctionToSave;
  if (FunctionToSave->getLinkage() != LinkageType)
    FunctionToSave->setLinkage(LinkageType);
}

// Searches module for functions to mark and creates pointers for them.
ACInstrumentation::ACInstrumentation(Function *InstrumentFunction, string FunctionToAnalyze, int Evaluations)
    : Evaluations(Evaluations), FunctionToInstrument(nullptr), AFInitFunction(nullptr),
      ACComputingFunction(nullptr), AFComputingFunction(nullptr),
      ACStoreFunction(nullptr), AFStoreFunction(nullptr),
      AFPrintTopAmplificationPaths(nullptr) {
  // Find and configure instrumentation functions
  Module *M = InstrumentFunction->getParent();

  for (auto &Global : M->getGlobalList()) {
    if (Global.getName().str().find("InstructionExecutionCounter") !=
        std::string::npos) {
      ExecutionCounter = &Global;
    }

    if (Global.getName().str().find("FunctionInstanceCounter") !=
            std::string::npos) {
      FunctionInstanceCounter = &Global;
    }
  }

  // Configuring all runtime functions and saving pointers.
  for (Module::iterator F = M->begin(); F != M->end(); ++F) {
    Function *CurrentFunction = &*F;

    // Only configuring functions with certain prefixes
    if (!CurrentFunction->hasName()) {

    } else if (CurrentFunction->getName().str().find(FunctionToAnalyze) !=
               std::string::npos) {
      confFunction(CurrentFunction, &FunctionToInstrument,
                   GlobalValue::LinkageTypes::ExternalLinkage);
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
    else if (CurrentFunction->getName().str().find("fAFPrintStatistics") != std::string::npos) {
      confFunction(CurrentFunction, &AFPrintStatistics,
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
                                  NumInstrumentedInstructions, F);
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
  int NumOperands = getFunctionNumOperands(FunctionType);

  bool IsCpFloatFunction = false;
  if(BaseInstruction->getOpcode() == Instruction::Call &&
      (static_cast<const CallInst *>(BaseInstruction)->getCalledFunction() &&
       isCPFloatFunction(
           static_cast<const CallInst *>(BaseInstruction)->getCalledFunction())))
    IsCpFloatFunction = true;

  Value *ResultNamePointer;
  // Creating a Global string for the Register the Result of this instruction is
  // stored in.
  if(IsCpFloatFunction) {
    ResultNamePointer = createRegisterNameGlobalString(static_cast<Instruction *>(BaseInstruction->getOperand(0)),
                                                              BaseInstruction->getModule());
  } else {
    ResultNamePointer = createRegisterNameGlobalString(
        static_cast<Instruction *>(BaseInstruction),
        BaseInstruction->getModule());
  }

  std::vector<Value *> OpRegisterNamesArray;
  std::vector<Value *> OperandValuesArray;

  // Looping through operands of BaseInstruction
  for (int I = 0; I < NumOperands; ++I) {
    // Creating a Global string representing the register name of operand if
    // it is not a constant.
    Value *OpRegisterNamePointer;
    if (!(isa<Constant>(BaseInstruction->getOperand(I)))) {
      OpRegisterNamePointer = createRegisterNameGlobalString(
          static_cast<Instruction *>(BaseInstruction->getOperand(I)),
          BaseInstruction->getModule());
      if(IsCpFloatFunction)
        OpRegisterNamePointer = createRegisterNameGlobalString(static_cast<Instruction *>(BaseInstruction->getOperand(I+1)),
                                                              BaseInstruction->getModule());
    }
    else
      OpRegisterNamePointer = EmptyValuePointer;


    OpRegisterNamesArray.push_back(OpRegisterNamePointer);

    Value *OperandValue;
    if(IsCpFloatFunction) {
      OperandValue = InstructionBuilder.CreateLoad(InstructionBuilder.getDoubleTy(),
                                                          BaseInstruction->getOperand(I+1));
      (*InstructionIterator)++;
      (*NumInstrumentedInstructions)++;
    } else {
      OperandValue = BaseInstruction->getOperand(I);
    }
    // Create a Cast Instruction to double in case Instruction is float Instruction.
    if (isSingleFPInstruction(&*BaseInstruction)) {
      if (OpRegisterNamePointer != EmptyValuePointer) {
        OperandValue = InstructionBuilder.CreateFPCast(
            BaseInstruction->getOperand(I),
            Type::getDoubleTy(BaseInstruction->getModule()->getContext()),
            "AFCast");

        (*InstructionIterator)++;
        (*NumInstrumentedInstructions)++;
      } else if(isa<ConstantFP>(BaseInstruction->getOperand(I))) {
        OperandValue = ConstantFP::get(InstructionBuilder.getDoubleTy(),
                                       APFloat(static_cast<ConstantFP*>(BaseInstruction->getOperand(I))->getValue().convertToDouble()));
      }
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
      InstructionBuilder.CreateCall(ACComputingFunction, ACArgsRef, "AFACCall");

  // 2*(Get Location + Store Value at Location) * NumOperands +
  // 2*(Allocate for Array + GetArrayLocation) + ACComputation Function Call
  (*NumInstrumentedInstructions) += 4 * NumOperands + 5;
  (*InstructionIterator)++;

  if(IsCpFloatFunction)
      InstructionACMap[BaseInstruction->getOperand(0)] = ACComputingCallInstruction;
  else
    InstructionACMap[BaseInstruction] = ACComputingCallInstruction;

  assert(ACComputingCallInstruction && "Invalid call instruction!");
  return;
}

void ACInstrumentation::instrumentCallsForAFComputation(
    Instruction *BaseInstruction, BasicBlock::iterator *InstructionIterator,
    long *NumInstrumentedInstructions, int FunctionType) {
  // Ensuring AFComputingFunction is linked
  assert((ACComputingFunction != nullptr) && "Function not initialized!");
  assert((FunctionType != -1) &&
         "Function cannot be analyzed");

  // Positioning the instruction builder
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());
  std::vector<Value *> AFArgs;
  CallInst *AFComputingCallInstruction;

  std::vector<Value *> AFArray;

  int NumOperands = getFunctionNumOperands(FunctionType);

  bool IsCpFloatFunction = false;
  if(BaseInstruction->getOpcode() == Instruction::Call &&
      (static_cast<const CallInst *>(BaseInstruction)->getCalledFunction() &&
       isCPFloatFunction(
           static_cast<const CallInst *>(BaseInstruction)->getCalledFunction())))
    IsCpFloatFunction = true;

  for (int I = 0; I < NumOperands; ++I) {
    if (IsCpFloatFunction) {
      if (InstructionAFMap.find(BaseInstruction->getOperand(I+1)) != InstructionAFMap.end())
        AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I+1)]);
      else
        AFArray.push_back(
            ConstantPointerNull::get(InstructionBuilder.getPtrTy()));
    }
    // Case when operand is another floating-point instruction for which we can
    // and must have instrumented an AFComputingFunction call.
    else if (getFunctionEnum(
            static_cast<Instruction *>(BaseInstruction->getOperand(I))) != -1) {
      assert(InstructionAFMap.find(BaseInstruction->getOperand(I)) != InstructionAFMap.end() &&
             "AFComputingFunction call not found for operand!");
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
    }
    // Case when operand is a float-to-float cast instruction for which we do not
    // instrument an AFComputingFunction call but it is mapped to an AFComputingFunction
    // call instrumented for ITS operand.
    else if (isFloatToFloatCastInstruction(static_cast<const Instruction *>(
                   BaseInstruction->getOperand(I))) &&
               InstructionAFMap.find(BaseInstruction->getOperand(I)) != InstructionAFMap.end()) {
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
    }
    // Case when operand is a phi node and we need to instrument a phi node for
    // AFComputingFunctions corresponding to it.
    else if (isa<Instruction>(BaseInstruction->getOperand(I)) &&
               static_cast<Instruction *>(BaseInstruction->getOperand(I))
                       ->getOpcode() == Instruction::PHI) {
      // Case when operand is a phi node.
      if(InstructionAFMap.find(BaseInstruction->getOperand(I)) == InstructionAFMap.end()) {
        // If Phi node is not yet instrumented, instrument it.
        instrumentPhiNodeForAF(BaseInstruction->getOperand(I), NumInstrumentedInstructions);
      }
      AFArray.push_back(InstructionAFMap[BaseInstruction->getOperand(I)]);
    } else {
      // Case for when the operand cannot have an associated AF Object.
      // eg: When operand is constant.
      AFArray.push_back(
          ConstantPointerNull::get(InstructionBuilder.getPtrTy()));
    }
  }

  Value *AllocatedAFArray =
      createArrayInIR(AFArray, &InstructionBuilder, InstructionIterator);

  Value *FunctionInstanceIdLoader = InstructionBuilder.CreateLoad(InstructionBuilder.getInt32Ty(),
                                                                  FunctionInstanceCounter);
  (*InstructionIterator)++;

  if (IsCpFloatFunction)
    AFArgs.push_back(InstructionACMap[BaseInstruction->getOperand(0)]);
  else
    AFArgs.push_back(InstructionACMap[BaseInstruction]);
  AFArgs.push_back(AllocatedAFArray);
  AFArgs.push_back(InstructionBuilder.getInt32(NumOperands));
  AFArgs.push_back(FunctionInstanceIdLoader);
  AFArgs.push_back(InstructionBuilder.getInt32(InstructionID));
  AFArgs.push_back(InstructionBuilder.getInt32(Evaluations+1));
  InstructionID++;

  // Creating a call to AFComputingFunction with above parameters
  ArrayRef<Value *> AFArgsRef(AFArgs);
  AFComputingCallInstruction =
      InstructionBuilder.CreateCall(AFComputingFunction, AFArgsRef, "AFCall");

  // (Get Location + Store Value at Location) * NumOperands +
  // (Allocate for Array + GetArrayLocation) + AFComputation Function Call
  *NumInstrumentedInstructions += 2 * NumOperands + 4;
  (*InstructionIterator)++;

  if(IsCpFloatFunction)
    InstructionAFMap[BaseInstruction->getOperand(0)] = AFComputingCallInstruction;
  else
    InstructionAFMap[BaseInstruction] = AFComputingCallInstruction;

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
      static_cast<PHINode *>(OriginalPHI)->getNumIncomingValues(), "AFFi");

  // Looping through the Incoming Values and setting the IncomingBlocks for the new PhiNode
  for (unsigned int I = 0;
       I < static_cast<PHINode *>(OriginalPHI)->getNumIncomingValues(); ++I) {
    BasicBlock *IncomingBlock =
        static_cast<PHINode *>(OriginalPHI)->getIncomingBlock(I);
    Value *IncomingValue =
        static_cast<PHINode *>(OriginalPHI)->getIncomingValue(I);

    if (InstructionAFMap.find(IncomingValue) != InstructionAFMap.end())
      static_cast<PHINode *>(AFPhi)->addIncoming(
          InstructionAFMap[IncomingValue], IncomingBlock);
    else
      static_cast<PHINode *>(AFPhi)->addIncoming(
          ConstantPointerNull::get(InstructionBuilder.getPtrTy()),
          IncomingBlock);
  }

  (*NumInstrumentedInstructions)++;

  InstructionAFMap[OriginalPHI] = AFPhi;

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
    assert(InstructionAFMap.find(TrueValue) != InstructionAFMap.end()  &&
           "InstructionAFMap does not contain value for this key.");
    TrueAFValue = InstructionAFMap[TrueValue];
  } else if(static_cast<Instruction*>(TrueValue)->getOpcode() == Instruction::PHI &&
        InstructionAFMap.find(TrueValue) == InstructionAFMap.end()) {
    instrumentPhiNodeForAF(TrueValue, NumInstrumentedInstructions);
    assert(InstructionAFMap.find(TrueValue) != InstructionAFMap.end()  &&
           "InstructionAFMap does not contain value for this key.");
    TrueAFValue = InstructionAFMap[TrueValue];
  }
  else
    TrueAFValue = ConstantPointerNull::get(InstructionBuilder.getPtrTy());

  if (getFunctionEnum(static_cast<Instruction *>(FalseValue)) != -1) {
    assert(InstructionAFMap.find(FalseValue) != InstructionAFMap.end()  &&
           "InstructionAFMap does not contain value for this key.");
    FalseAFValue = InstructionAFMap[FalseValue];
  } else if(static_cast<Instruction*>(FalseValue)->getOpcode() == Instruction::PHI &&
        InstructionAFMap.find(FalseValue) == InstructionAFMap.end()) {
    instrumentPhiNodeForAF(FalseValue, NumInstrumentedInstructions);
    assert(InstructionAFMap.find(FalseValue) != InstructionAFMap.end()  &&
           "InstructionAFMap does not contain value for this key.");
    FalseAFValue = InstructionAFMap[FalseValue];
  }
  else
    FalseAFValue = ConstantPointerNull::get(InstructionBuilder.getPtrTy());

  // Instrumenting Select instruction, incrementing instrumented instructions
  // and incrementing insert pointer.
  Value *AFSel = InstructionBuilder.CreateSelect(
      static_cast<SelectInst *>(OriginalSelInstr)->getCondition(), TrueAFValue,
      FalseAFValue, "AFSel");
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

  (*NumInstrumentedInstructions)++;
  (*InstructionIterator)++;

  assert((StoreInFileCall != nullptr) && "Function not initialized!");
}

void ACInstrumentation::instrumentMarkedFunction(BasicBlock::iterator *InstructionIterator,
                                                 long int *NumInstrumentedInstructions) {
  assert((FunctionToInstrument != nullptr) && "Function not initialized!");
  IRBuilder<> InstructionBuilder((*InstructionIterator)->getNextNode());


  // Providing a seed value
  Instruction *BaseInstruction = &(**InstructionIterator);
  srand((unsigned) time(NULL));

  for (int I = 0; I < Evaluations; ++I) {
    std::vector<Value *> Args;
    // Loop through the arguments
    for (Function::arg_iterator ArgumentIterator = FunctionToInstrument->arg_begin(),
                                LastArgument = FunctionToInstrument->arg_end();
         ArgumentIterator != LastArgument; ++ArgumentIterator) {
      // If the argument is a float type, generate a random floating point number and add it to the argument list.
      if (ArgumentIterator->getType()->isFloatTy()) {
        Value *RandomFloat = InstructionBuilder.CreateFAdd(BaseInstruction->getOperand(ArgumentIterator->getArgNo()),
                                                           ConstantFP::get(Type::getFloatTy(ArgumentIterator->getContext()),
                                                                           static_cast<float >(rand()) /
                                                                               static_cast<float>(RAND_MAX) * pow(2, -14)));
        if(!isa<ConstantFP>(BaseInstruction->getOperand(ArgumentIterator->getArgNo()))) {
          (*NumInstrumentedInstructions)++;
          (*InstructionIterator)++;
        }
//        Value *RandomFloat = ConstantFP::get(Type::getFloatTy(ArgumentIterator->getContext()),
//                                             static_cast<float >(rand()) /
//                                                 static_cast<float>(RAND_MAX));
        Args.push_back(RandomFloat);
      }
      // If the argument is a double type, generate a random floating point number and add it to the argument list.
      else if (ArgumentIterator->getType()->isDoubleTy()) {
        Value *RandomFloat = InstructionBuilder.CreateFAdd(BaseInstruction->getOperand(ArgumentIterator->getArgNo()),
                                                           ConstantFP::get(Type::getDoubleTy(ArgumentIterator->getContext()),
                                                                           static_cast<double >(rand()) /
                                                                               static_cast<double>(RAND_MAX) * pow(2, -14)));
        if(!isa<ConstantFP>(BaseInstruction->getOperand(ArgumentIterator->getArgNo()))) {
          (*NumInstrumentedInstructions)++;
          (*InstructionIterator)++;
        }
//        Value *RandomFloat = ConstantFP::get(Type::getDoubleTy(ArgumentIterator->getContext()),
//                                             static_cast<double>(rand()) /
//                                                 static_cast<double>(RAND_MAX));
        Args.push_back(RandomFloat);
      }
      // If the argument is an integer type, generate a random integer number
      // and add it to the argument list.
      else if (ArgumentIterator->getType()->isIntegerTy()) {
        Value *RandomInt =
            ConstantInt::get(Type::getInt32Ty(ArgumentIterator->getContext()), rand());
        Args.push_back(RandomInt);
      }
    }

    // Creating a call instruction with above parameters
    ArrayRef<Value *> ArgsRef(Args);
    InstructionBuilder.CreateCall(FunctionToInstrument, ArgsRef);

    (*NumInstrumentedInstructions)++;
    (*InstructionIterator)++;
  }
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

    if (this->FunctionToInstrument &&
        BB->getParent()->getName().compare(this->FunctionToInstrument->getName()) == 0) {
      // Branch based on kind of Instruction
      if (isFloatToFloatCastInstruction(&*I)) {
        mapFloatCastToAFValue(&*I, NumInstrumentedInstructions);
      } else if (isTernaryInstruction(&*I)) {
        instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                            NumInstrumentedInstructions);
      } else if (isUnaryInstruction(&*I)) {
        instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                            NumInstrumentedInstructions);
      } else if (isBinaryInstruction(&*I)) {
        instrumentCallsToAnalyzeInstruction(CurrentInstruction, &I,
                                            NumInstrumentedInstructions);
      } else if ((&*I)->getOpcode() == Instruction::Select) {
        assert(InstructionAFMap.find(CurrentInstruction) == InstructionAFMap.end() &&
               "InstructionAFMap already contains this instruction.");
        InstructionAFMap[CurrentInstruction] = instrumentSelectForAF(CurrentInstruction, &I,
                                                                     NumInstrumentedInstructions);
      }

      // CurrentInstruction is updated to the BasicBlock iterators position as the
      // previous if-else ladder may have instrumented some instructions and we
      // want to avoid using the instrumented instructions as base for further
      // instrumentation.
      CurrentInstruction = &*I;

      if (CurrentInstruction->getOpcode() == Instruction::Call) {
        string FunctionName = "";
        if (static_cast<CallInst *>(CurrentInstruction)
                ->getCalledFunction() &&
            static_cast<CallInst *>(CurrentInstruction)
                ->getCalledFunction()
                ->hasName())
          FunctionName = static_cast<CallInst *>(CurrentInstruction)
                             ->getCalledFunction()
                             ->getName()
                             .str();
        transform(FunctionName.begin(), FunctionName.end(),
                  FunctionName.begin(), ::tolower);
        if (FunctionName.find("markforresult") != std::string::npos) {
          if (!isa<Constant>(static_cast<CallInst *>(CurrentInstruction)
                                 ->data_operands_begin()
                                 ->get()) &&
              static_cast<CallInst *>(
                  static_cast<CallInst *>(CurrentInstruction)
                      ->data_operands_begin()
                      ->get())
                      ->getOpcode() != Instruction::Load) {
            instrumentForMarkedVariable(
                static_cast<CallInst *>(CurrentInstruction)
                    ->data_operands_begin()
                    ->get(),
                &I, NumInstrumentedInstructions);
          } else {
            errs() << "Value to be analyzed has been optimized into a constant\n";
          }
        }
      }
    }

    // Creating additional calls
    if (CurrentInstruction->getOpcode() == Instruction::Call &&
      static_cast<CallInst *>(CurrentInstruction)
            ->getCalledFunction() &&
      static_cast<CallInst *>(CurrentInstruction)
          ->getCalledFunction()
          ->hasName() &&
      FunctionToInstrument &&
      static_cast<CallInst *>(CurrentInstruction)
          ->getCalledFunction()->getName().compare(this->FunctionToInstrument->getName()) == 0) {
        instrumentMarkedFunction(&I, NumInstrumentedInstructions);
    }
  }

  // Finding and Completing Incomplete Phi Nodes
  if (this->FunctionToInstrument &&
      BB->getParent()->getName().compare(this->FunctionToInstrument->getName()) == 0) {
    for (BasicBlock::phi_iterator I = BB->phis().begin(); I != BB->phis().end(); ++I) {
      Value *OriginalPHI = &*I;

      if(InstructionAFMap.find(OriginalPHI) != InstructionAFMap.end()) {
        Value *AFPhi = InstructionAFMap[OriginalPHI];

        // Iterate through all the incoming values of AFPhi and replace null values
        // with the mapped value of corresponding OriginalPHI from InstructionAFMap
        for (unsigned J = 0; J < static_cast<PHINode *>(OriginalPHI)->getNumIncomingValues(); J++) {
          if (isa<ConstantPointerNull>(static_cast<PHINode *>(AFPhi)->getIncomingValue(J))
              && InstructionAFMap.find(static_cast<PHINode *>(OriginalPHI)->getIncomingValue(J)) != InstructionAFMap.end()) {
            Value *IncomingValue = InstructionAFMap[static_cast<PHINode *>(OriginalPHI)->getIncomingValue(J)];
            static_cast<PHINode *>(AFPhi)->setIncomingValue(J, IncomingValue);
          }
        }
      }
    }
  }

  return;
}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  assert((AFInitFunction != nullptr) && "AFInitFunction not initialized!");
  assert((ACStoreFunction != nullptr) && "ACStoreFunction not initialized!");
  assert((AFStoreFunction != nullptr) && "AFStoreFunction not initialized!");
  assert((AFPrintTopAmplificationPaths != nullptr) &&
         "AFPrintTopAmplificationPaths not initialized!");
  assert((AFPrintStatistics != nullptr) && "AFPrintStatistics not initialized!");

  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);

  std::vector<Value *> ACInitCallArgs, AFInitCallArgs;

  CallInst *AFInitCallInstruction, *PrintAFPathsCallInstruction,
      *PrintStatisticsCallInstruction;
  CallInst *StoreACTableCallInstruction, *StoreAFTableCallInstruction;

  InstructionBuilder.CreateStore(InstructionBuilder.getInt32(0), FunctionInstanceCounter);

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
        std::vector<Value *> PrintAFPathsCallArgs, ACStoreCallArgs, AFStoreCallArgs,
            PrintStatisticsCallArgs;

        ArrayRef<Value *> ACStoreCallArgsRef(ACStoreCallArgs);
        ArrayRef<Value *> AFStoreCallArgsRef(AFStoreCallArgs);
        ArrayRef<Value *> PrintAFPathsCallArgsRef(PrintAFPathsCallArgs);

        PrintStatisticsCallArgs.push_back(InstructionBuilder.getInt32(Evaluations+1));
        ArrayRef<Value *> PrintStatisticsCallArgsRef(PrintStatisticsCallArgs);

        InstructionBuilder.SetInsertPoint(CurrentInstruction);
        StoreACTableCallInstruction =
            InstructionBuilder.CreateCall(ACStoreFunction, ACStoreCallArgsRef);
        StoreAFTableCallInstruction =
            InstructionBuilder.CreateCall(AFStoreFunction, AFStoreCallArgsRef);
        PrintAFPathsCallInstruction = InstructionBuilder.CreateCall(
            AFPrintTopAmplificationPaths, PrintAFPathsCallArgsRef);
        PrintStatisticsCallInstruction = InstructionBuilder.CreateCall(
            AFPrintStatistics, PrintStatisticsCallArgsRef);
      }
    }
  }

  assert(AFInitCallInstruction && "Invalid call instruction!");
  assert(StoreACTableCallInstruction && StoreAFTableCallInstruction &&
         "Invalid call instruction!");
  assert(PrintAFPathsCallInstruction && PrintStatisticsCallInstruction &&
         "Invalid call instruction!");
  return;
}

void ACInstrumentation::instrumentFunctionInitializationInsts(Function *F) {
  assert((ExecutionCounter != nullptr) &&
         (FunctionInstanceCounter != nullptr) && "GlobalVariable pointers not initialized!");

  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();

  IRBuilder<> InstructionBuilder(Inst);

  InstructionBuilder.CreateStore(InstructionBuilder.getInt32(0), ExecutionCounter);

  Value *FunctionInstanceIdLoader = InstructionBuilder.CreateLoad(InstructionBuilder.getInt32Ty(),
                                                                  FunctionInstanceCounter);

  Value *Result = InstructionBuilder.CreateAdd(FunctionInstanceIdLoader, InstructionBuilder.getInt32(1));
  InstructionBuilder.CreateStore(Result,FunctionInstanceCounter);

}

Value *
ACInstrumentation::createArrayInIR(vector<Value *> ArrayOfValues,
                                   IRBuilder<> *InstructionBuilder,
                                   BasicBlock::iterator *InstructionIterator) {
  // Inserting an Alloca Instruction to allocate memory for the array.
  Value *AllocatedArray = InstructionBuilder->CreateAlloca(
      ArrayType::get(InstructionBuilder->getPtrTy(), ArrayOfValues.size()),
      NULL, "AFAlloca");
  (*InstructionIterator)++;

  // Looping through operands of BaseInstruction
  for (long unsigned int I = 0; I < ArrayOfValues.size(); ++I) {
    // Setting Value in OperandNameArray
    Value *LocationInArray = InstructionBuilder->CreateGEP(
        InstructionBuilder->getPtrTy(), AllocatedArray,
        InstructionBuilder->getInt32(I), "AFGep");
    (*InstructionIterator)++;
    InstructionBuilder->CreateStore(ArrayOfValues[I], LocationInArray, "AFStore");
    (*InstructionIterator)++;
  }

  // Get pointers to arrays
  Value *Array = InstructionBuilder->CreateGEP(InstructionBuilder->getPtrTy(),
                                               AllocatedArray,
                                               InstructionBuilder->getInt32(0),
                                               "AFGep");
  (*InstructionIterator)++;

  return Array;
}

bool ACInstrumentation::isPhiNode(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::PHI;
}

bool ACInstrumentation::isMemoryLoadInstruction(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::Alloca ||
         Inst->getOpcode() == Instruction::Load;
}

bool ACInstrumentation::isIntegerToFloatCastInstruction(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::UIToFP ||
         Inst->getOpcode() == Instruction::SIToFP;
}

bool ACInstrumentation::isFloatToFloatCastInstruction(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FPTrunc ||
         Inst->getOpcode() == Instruction::FPExt;
}

bool ACInstrumentation::isUnaryInstruction(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FNeg ||
         (Inst->getOpcode() == Instruction::Call &&
          (static_cast<const CallInst *>(Inst)->getCalledFunction() &&
           isUnaryFunction(
               static_cast<const CallInst *>(Inst)->getCalledFunction())));
}

// TODO: Check for FRem case.
bool ACInstrumentation::isBinaryInstruction(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FAdd ||
         Inst->getOpcode() == Instruction::FSub ||
         Inst->getOpcode() == Instruction::FMul ||
         Inst->getOpcode() == Instruction::FDiv ||
         (Inst->getOpcode() == Instruction::Call &&
          (static_cast<const CallInst *>(Inst)->getCalledFunction() &&
           isBinaryFunction(
               static_cast<const CallInst *>(Inst)->getCalledFunction())));
}

bool ACInstrumentation::isTernaryInstruction(const Instruction *Inst) {
  return (Inst->getOpcode() == Instruction::Call &&
          static_cast<const CallInst *>(Inst)->getCalledFunction() &&
          static_cast<const CallInst *>(Inst)->getCalledFunction()->hasName() &&
          isTernaryFunction(
              static_cast<const CallInst *>(Inst)->getCalledFunction()));
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

bool ACInstrumentation::isSingleFPInstruction(const Instruction *Inst) {
  return Inst->getType()->isFloatTy();
}

bool ACInstrumentation::isDoubleFPInstruction(const Instruction *Inst) {
  return Inst->getType()->isDoubleTy();
}

bool ACInstrumentation::isUnwantedFunction(const Function *Func) {
  assert(Func->hasName());
  return Func->getName().str().find("fAC") != std::string::npos ||
         Func->getName().str().find("fCG") != std::string::npos ||
         Func->getName().str().find("fAF") != std::string::npos ||
         Func->getName().str().find("fRS") != std::string::npos ||
         Func->getName().str().find("fURT") != std::string::npos ||
         Func->getName().str().find("fST") != std::string::npos ||
         Func->getName().str().find("ACItem") != std::string::npos;
}

void ACInstrumentation::mapFloatCastToAFValue(Instruction *Inst,
                                              long *NumInstrumentedInstructions) {
  if (Inst->getOpcode() == Instruction::FPTrunc) {
    if (InstructionAFMap.find(static_cast<const FPTruncInst *>(Inst)->getOperand(0)) == InstructionAFMap.end()
        && isPhiNode(static_cast<const Instruction *>(static_cast<const FPTruncInst *>(Inst)->getOperand(0)))) {
      // If the operand of the FPTruncInst is a PHINode, then we need to instrument
      // a corresponding PHINode for the AF value.
      instrumentPhiNodeForAF(static_cast<const FPTruncInst *>(Inst)->getOperand(0),
                             NumInstrumentedInstructions);
    }
    if (InstructionAFMap.find(static_cast<const FPTruncInst *>(Inst)->getOperand(0)) != InstructionAFMap.end())
      InstructionAFMap[Inst] = InstructionAFMap[static_cast<const FPTruncInst *>(Inst)->getOperand(0)];
  } else if(Inst->getOpcode() == Instruction::FPExt) {
    if(InstructionAFMap.find(static_cast<const FPExtInst *>(Inst)->getOperand(0)) == InstructionAFMap.end()
        && isPhiNode(static_cast<const Instruction *>(static_cast<const FPExtInst *>(Inst)->getOperand(0)))) {
      // If the operand of the FPExtInst is a PHINode, then we need to instrument
      // a corresponding PHINode for the AF value.
      instrumentPhiNodeForAF(static_cast<const FPExtInst *>(Inst)->getOperand(0),
                                 NumInstrumentedInstructions);
    }
    if(InstructionAFMap.find(static_cast<const FPExtInst *>(Inst)->getOperand(0)) != InstructionAFMap.end())
      InstructionAFMap[Inst] = InstructionAFMap[static_cast<const FPExtInst *>(Inst)->getOperand(0)];
  }
}

}  // namespace atomiccondition
