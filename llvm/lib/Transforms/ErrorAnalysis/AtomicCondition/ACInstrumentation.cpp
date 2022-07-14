//
// Created by tanmay on 6/10/22.
//

#include <string>
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h"


using namespace llvm;
using namespace atomiccondition;
using namespace std;

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
                                                                     AFInitFunction(nullptr),
                                                                     ACfp32UnaryFunction(nullptr),
                                                                     ACfp64UnaryFunction(nullptr),
                                                                     ACfp32BinaryFunction(nullptr),
                                                                     ACfp64BinaryFunction(nullptr),
                                                                     CGRecordPHIInstruction(nullptr),
                                                                     CGRecordBasicBlock(nullptr),
                                                                     CGCreateNode(nullptr),
                                                                     ACStoreFunction(nullptr),
                                                                     CGStoreFunction(nullptr),
                                                                     AFStoreFunction(nullptr),
                                                                     AFfp32AnalysisFunction(nullptr),
                                                                     AFfp64AnalysisFunction(nullptr),
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
    else if (CurrentFunction->getName().str().find("fCGInitialize") != std::string::npos) {
      confFunction(CurrentFunction, &CGInitFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFInitialize") != std::string::npos) {
      confFunction(CurrentFunction, &AFInitFunction,
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
    }else if (CurrentFunction->getName().str().find("fAFStoreResult") != std::string::npos) {
      confFunction(CurrentFunction, &AFStoreFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFfp32Analysis") != std::string::npos) {
      confFunction(CurrentFunction, &AFfp32AnalysisFunction,
                   GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (CurrentFunction->getName().str().find("fAFfp64Analysis") != std::string::npos) {
      confFunction(CurrentFunction, &AFfp64AnalysisFunction,
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

  CallInst *BBRecordingCallInstruction = nullptr;

  Args.push_back(createBBNameString(CurrentBB));

  BBRecordingCallInstruction = InstructionBuilder.CreateCall(CGRecordBasicBlock, Args);

  *NumInstrumentedInstructions+=1;
  assert(BBRecordingCallInstruction && "Invalid call instruction!");
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
    Args.push_back(createInstructionString(&*CurrPhi));
    Args.push_back(createBBNameString(CurrentBB));
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

  CallInst *CGCallInstruction = nullptr;

  Constant *EmptyValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                            "",
                                                            true);

  Value *EmptyValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                      EmptyValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      EmptyValue);

  Args.push_back(createInstructionString(BaseInstruction));
  Args.push_back(EmptyValuePointer);
  Args.push_back(EmptyValuePointer);
  Args.push_back(InstructionBuilder.getInt32(NodeKind::Register));
  ArrayRef<Value *> ArgsRef(Args);

  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, ArgsRef);
  *NumInstrumentedInstructions+=1;

  assert(CGCallInstruction && "Invalid call instruction!");
  return;
}

// Instruments a call to calculate atomic condition for unary floating point
// instructions and creates a node for this instruction in the computation graph.
void ACInstrumentation::instrumentCallsForUnaryOperation(Instruction* BaseInstruction,
                                            long *NumInstrumentedInstructions) {
  assert((CGCreateNode!=nullptr) && (ACfp32UnaryFunction!=nullptr) &&
         (ACfp64UnaryFunction!=nullptr) && "Function not initialized!");
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

  Constant *EmptyValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                      "",
                                                      true);

  Value *EmptyValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                EmptyValue->getType(),
                                                true,
                                                GlobalValue::InternalLinkage,
                                                EmptyValue);

  Value *LeftOpRegisterNamePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(0)))
    LeftOpRegisterNamePointer = createRegisterNameString(static_cast<Instruction*>(BaseInstruction->getOperand(0)));
  else
    LeftOpRegisterNamePointer = EmptyValuePointer;

  ACArgs.push_back(LeftOpRegisterNamePointer);
  ACArgs.push_back(BaseInstruction->getOperand(0));

  ACArgs.push_back(InstructionBuilder.getInt32(OpType));

  ArrayRef<Value *> ACArgsRef(ACArgs);

  CallInst *ACComputingCallInstruction = nullptr;

  // Branch based on data type of operation
  if(isSingleFPOperation(BaseInstruction)) {
    ACComputingCallInstruction =
        InstructionBuilder.CreateCall(ACfp32UnaryFunction, ACArgsRef);
    *NumInstrumentedInstructions+=1;
  }
  else if(isDoubleFPOperation(BaseInstruction)) {
    ACComputingCallInstruction =
        InstructionBuilder.CreateCall(ACfp64UnaryFunction, ACArgsRef);
    *NumInstrumentedInstructions+=1;
  }

  //----------------------------------------------------------------------------
  //----------------- Instrumenting CG Node creating function -----------------
  //----------------------------------------------------------------------------
  std::vector<Value *> CGArgs;

  CallInst *CGCallInstruction = nullptr;

  Value *LeftOpInstructionValuePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(0)))
    LeftOpInstructionValuePointer = createInstructionString(static_cast<Instruction*>(BaseInstruction->getOperand(0)));
  else
    LeftOpInstructionValuePointer = EmptyValuePointer;

  CGArgs.push_back(createInstructionString(BaseInstruction));
  CGArgs.push_back(LeftOpInstructionValuePointer);
  CGArgs.push_back(EmptyValuePointer);
  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::UnaryInstruction));
  ArrayRef<Value *> CGArgsRef(CGArgs);

  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
  *NumInstrumentedInstructions+=1;
  assert(ACComputingCallInstruction && CGCallInstruction && "Invalid call instruction!");
  return;
}

// Instruments a call to calculate atomic condition for binary floating point
// instructions and creates a node for this instruction in the computation graph.
void ACInstrumentation::instrumentCallsForBinaryOperation(Instruction* BaseInstruction,
                                             long *NumInstrumentedInstructions) {
  assert((CGCreateNode!=nullptr) && (ACfp32BinaryFunction!=nullptr) &&
         (ACfp64BinaryFunction!=nullptr) && "Function not initialized!");

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

  Constant *EmptyValue = ConstantDataArray::getString(BaseInstruction->getModule()->getContext(),
                                                      "",
                                                      true);

  Value *EmptyValuePointer = new GlobalVariable(*BaseInstruction->getModule(),
                                                EmptyValue->getType(),
                                                true,
                                                GlobalValue::InternalLinkage,
                                                EmptyValue);

  Value *LeftOpRegisterNamePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(0)))
    LeftOpRegisterNamePointer = createRegisterNameString(static_cast<Instruction*>(BaseInstruction->getOperand(0)));
  else
    LeftOpRegisterNamePointer = EmptyValuePointer;

  Value *RightOpRegisterNamePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(1)))
    RightOpRegisterNamePointer = createRegisterNameString(static_cast<Instruction*>(BaseInstruction->getOperand(1)));
  else
    RightOpRegisterNamePointer = EmptyValuePointer;

  Args.push_back(LeftOpRegisterNamePointer);
  Args.push_back(BaseInstruction->getOperand(0));
  Args.push_back(RightOpRegisterNamePointer);
  Args.push_back(BaseInstruction->getOperand(1));
  Args.push_back(InstructionBuilder.getInt32(OpType));

  ArrayRef<Value *> ArgsRef(Args);

  CallInst *ACComputingCallInstruction = nullptr;

  // Branch based on data type of operation
  if(isSingleFPOperation(BaseInstruction)) {
    ACComputingCallInstruction =
        InstructionBuilder.CreateCall(ACfp32BinaryFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }
  else if(isDoubleFPOperation(BaseInstruction)) {
    ACComputingCallInstruction =
        InstructionBuilder.CreateCall(ACfp64BinaryFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }

  //----------------------------------------------------------------------------
  //----------------- Instrumenting CG Node creating function -----------------
  //----------------------------------------------------------------------------
  std::vector<Value *> CGArgs;

  CallInst *CGCallInstruction = nullptr;

  Value *LeftOpInstructionValuePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(0)))
    LeftOpInstructionValuePointer = createInstructionString(static_cast<Instruction*>(BaseInstruction->getOperand(0)));
  else
    LeftOpInstructionValuePointer = EmptyValuePointer;

  Value *RightOpInstructionValuePointer;
  if(!isa<Constant>(BaseInstruction->getOperand(1)))
    RightOpInstructionValuePointer = createInstructionString(static_cast<Instruction*>(BaseInstruction->getOperand(1)));
  else
    RightOpInstructionValuePointer = EmptyValuePointer;

  CGArgs.push_back(createInstructionString(BaseInstruction));
  CGArgs.push_back(LeftOpInstructionValuePointer);
  CGArgs.push_back(RightOpInstructionValuePointer);
  CGArgs.push_back(InstructionBuilder.getInt32(NodeKind::BinaryInstruction));
  ArrayRef<Value *> CGArgsRef(CGArgs);

  CGCallInstruction = InstructionBuilder.CreateCall(CGCreateNode, CGArgsRef);
  *NumInstrumentedInstructions+=1;

  assert(ACComputingCallInstruction && CGCallInstruction && "Invalid call instruction!");
  return;
}

// Instruments a call to calculate Amplification Factor of this Nodes
void ACInstrumentation::instrumentCallsForAFAnalysis(
    Instruction *BaseInstruction, Instruction *LocationToInstrument,
    long *NumInstrumentedInstructions) {
  assert((AFfp32AnalysisFunction!=nullptr) &&
         (AFfp64AnalysisFunction!=nullptr) &&
         "Function not initialized!");

  BasicBlock::iterator NextInst(LocationToInstrument);
  NextInst++;
  IRBuilder<> InstructionBuilder( &(*NextInst) );

  std::vector<Value *> Args;
  Args.push_back(createInstructionString(BaseInstruction));
  ArrayRef<Value *> ArgsRef(Args);

  CallInst *AFComputingCallInstruction = nullptr;
  if(isSingleFPOperation(BaseInstruction)) {
    AFComputingCallInstruction =
        InstructionBuilder.CreateCall(AFfp32AnalysisFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }
  else if(isDoubleFPOperation(BaseInstruction)) {
    AFComputingCallInstruction =
        InstructionBuilder.CreateCall(AFfp64AnalysisFunction, ArgsRef);
    *NumInstrumentedInstructions+=1;
  }

  assert(AFComputingCallInstruction && "Invalid call instruction!");
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


    // Instrument Amplification Factor Calculating Function
    if(CurrentInstruction->getOpcode() == Instruction::Call) {
      string FunctionName = static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->getName().str();
      transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
      if(FunctionName.find("markforresult") != std::string::npos) {
        instrumentCallsForAFAnalysis(static_cast<Instruction*>(static_cast<CallInst*>(CurrentInstruction)->data_operands_begin()->get()),
                                     CurrentInstruction,
                                     &*NumInstrumentedInstructions);
        *NumInstrumentedInstructions += 1;
      }
    }

    // Instrument Amplification Factor Calculating Function in case there is a
    // print function or a return call
//    if(CurrentInstruction->getOpcode() == Instruction::Ret ||
//        CurrentInstruction->getOpcode() == Instruction::Call) {
//      if(CurrentInstruction->getOpcode() == Instruction::Call) {
//        string FunctionName = static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->getName().str();
//        transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(), ::tolower);
//        if(FunctionName.find("print") != std::string::npos){
//
//          // For loop below wont work since getCalledFunction gets the
            // SIGNATURE of the function rather that the called instance. Use
            // data_operands instead
//          for (Function::op_iterator CurrResultOp = static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->op_begin();
//               CurrResultOp != static_cast<CallInst*>(CurrentInstruction)->getCalledFunction()->op_end();
//               ++CurrResultOp) {
//            errs() << *CurrResultOp << "\n";
//            Instruction* ResultInstruction;
//            if(isa<Instruction>(CurrResultOp->get())) {
//              ResultInstruction =
//                  static_cast<Instruction *>(CurrResultOp->get());
//              if(ResultInstruction->getType()->isFloatingPointTy()) {
//                while(!isUnaryOperation(ResultInstruction) &&
//                       !isBinaryOperation(ResultInstruction) &&
//                       ResultInstruction->getOpcode() != Instruction::PHI) {
//                  if (isa<Instruction>(ResultInstruction->getOperand(0)))
//                    ResultInstruction =
//                        (Instruction *)ResultInstruction->getOperand(0);
//                  else
//                    exit(1);
//                }
//                instrumentCallsForAFAnalysis(ResultInstruction,
//                                             &*NumInstrumentedInstructions);
//                *NumInstrumentedInstructions+=1;
//              }
//            }
//          }
//        }
//      }
//      else {
//        Instruction* ResultInstruction;
//        if(isa<Instruction>(static_cast<ReturnInst*>(CurrentInstruction)->getReturnValue())) {
//          ResultInstruction =
//              static_cast<Instruction *>(static_cast<ReturnInst*>(CurrentInstruction)->getReturnValue());
//          if(ResultInstruction->getType()->isFloatingPointTy()) {
//            instrumentCallsForAFAnalysis(ResultInstruction,
//                                         &*NumInstrumentedInstructions);
//            *NumInstrumentedInstructions+=1;
//          }
//        }
//      }
//    }
  }

  return;
}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  assert((ACInitFunction!=nullptr) &&
         (CGInitFunction!=nullptr) &&
         (AFInitFunction!= nullptr) &&
         (ACStoreFunction!=nullptr) &&
         (CGStoreFunction!=nullptr) &&
         (AFStoreFunction!= nullptr) &&
         (CGDotGraphFunction!=nullptr) &&
         "Function not initialized!");
  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);
  std::vector<Value *> ACInitCallArgs, CGInitCallArgs, AFInitCallArgs,
      AnalysisCallArgs;
  std::vector<Value *> ACStoreCallArgs, CGStoreCallArgs, AFStoreCallArgs;
  std::vector<Value *> DotGraphCallArgs;

  CallInst *ACInitCallInstruction, *CGInitCallInstruction, *AFInitCallInstruction,
      *AnalysisCallInstruction;
  CallInst *StoreACTableCallInstruction, *StoreCGTableCallInstruction,
      *StoreAFTableCallInstruction;
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
  AFInitCallInstruction = InstructionBuilder.CreateCall(AFInitFunction, AFInitCallArgs);
//  }

  // Instrument call to print table
  for (Function::iterator BBIter=F->begin(); BBIter != F->end(); ++BBIter) {
    for (BasicBlock::iterator InstIter=BBIter->begin(); InstIter != BBIter->end(); ++InstIter) {
      Instruction *CurrentInstruction = &(*InstIter);
      if (isa<ReturnInst>(CurrentInstruction) || isa<ResumeInst>(CurrentInstruction)) {
        ArrayRef<Value *> ACStoreCallArgsRef(ACStoreCallArgs);
        ArrayRef<Value *> CGStoreCallArgsRef(CGStoreCallArgs);
        ArrayRef<Value *> AFStoreCallArgesRef(AFStoreCallArgs);

        InstructionBuilder.SetInsertPoint(CurrentInstruction);
        StoreACTableCallInstruction = InstructionBuilder.CreateCall(ACStoreFunction, ACStoreCallArgsRef);
        StoreCGTableCallInstruction = InstructionBuilder.CreateCall(CGStoreFunction, CGStoreCallArgsRef);
        StoreAFTableCallInstruction = InstructionBuilder.CreateCall(AFStoreFunction, AFStoreCallArgesRef);

        ArrayRef<Value *> DotGraphCallArgsRef(DotGraphCallArgs);
        DotGraphCallInstruction = InstructionBuilder.CreateCall(CGDotGraphFunction, DotGraphCallArgsRef);
      }
    }
  }

  assert(ACInitCallInstruction && CGInitCallInstruction &&
         AFInitCallInstruction && AnalysisCallInstruction && "Invalid call instruction!");
  assert(StoreACTableCallInstruction && StoreCGTableCallInstruction &&
         StoreAFTableCallInstruction && "Invalid call instruction!");
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
  } else if(Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode*>(Inst)->getType()->isFloatTy();
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
  } else if(Inst->getOpcode() == Instruction::PHI) {
    return static_cast<const PHINode*>(Inst)->getType()->isDoubleTy();
  }

  return false;
}

bool ACInstrumentation::isUnwantedFunction(const Function *Func) {
  return Func->getName().str().find("fAC") != std::string::npos ||
         Func->getName().str().find("fCG") != std::string::npos ||
         Func->getName().str().find("fAF") != std::string::npos ||
         Func->getName().str().find("ACItem") != std::string::npos;
}

Value *ACInstrumentation::createBBNameString(BasicBlock *BB) {
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

Value *ACInstrumentation::createRegisterNameString(Instruction *Inst) {
  string InstructionString;
  raw_string_ostream RawRegisterString(InstructionString);
  Inst->printAsOperand(RawRegisterString, false);

  Constant *InstructionValue = ConstantDataArray::getString(Inst->getModule()->getContext(),
                                                            RawRegisterString.str().c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*Inst->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);

  return InstructionValuePointer;
}

Value *ACInstrumentation::createInstructionString(Instruction *Inst) {
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
