//
// Created by tanmay on 6/10/22.
//

#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h"

using namespace llvm;
using namespace atomiccondition;

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


ACInstrumentation::ACInstrumentation(Function *InstrumentFunction) : FunctionToInstrument(InstrumentFunction),
                                                                     ACInitFunction(nullptr),
                                                                     ACfp32UnaryFunction(nullptr),
                                                                     ACfp64UnaryFunction(nullptr),
                                                                     ACfp32BinaryFunction(nullptr),
                                                                     ACfp64BinaryFunction(nullptr){
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

    // Setting linkage of runtime functions
  }
}


void ACInstrumentation::instrumentBasicBlock(BasicBlock *BB,
                                             long *NumInstrumentedInstructions) {

//  if (ACInstrumentation::isUnwantedFunction(BB->getParent()))
//    return;

//  assert((ACfp32UnaryFunction!=nullptr) && "Function not initialized!");
  assert((ACfp32BinaryFunction!=nullptr) && "Function not initialized!");
  //  assert((ACfp64UnaryFunction!=nullptr) && "Function not initialized!");
  assert((ACfp64BinaryFunction!=nullptr) && "Function not initialized!");

  long int NuminstrumentedOps = 0;
  for (BasicBlock::iterator I = BB->begin();
       I != BB->end(); ++I) {
    Instruction* CurrentInstruction = &*I;

    // Branch based on kind of operation
    if(isUnaryOperation(CurrentInstruction)) {

    }
    else if(isBinaryOperation(CurrentInstruction)) {
      BasicBlock::iterator NextInst(CurrentInstruction);
      NextInst++;
      IRBuilder<> InstructionBuilder( &(*NextInst) );

      std::vector<Value *> Call1Args;
      std::vector<Value *> Call2Args;

      string XString = CurrentInstruction->getOperand(0)->getName().str();
      if(XString == "")
        XString = to_string(VarCounter);
      VarCounter++;
      Constant *XValue = ConstantDataArray::getString(BB->getModule()->getContext(),
                                                      XString,
                                                      true);
      Value *X = new GlobalVariable(*BB->getModule(),
                                    XValue->getType(),
                                    true,
                                    GlobalValue::InternalLinkage,
                                    XValue);

      string YString = CurrentInstruction->getOperand(1)->getName().str();
      if(YString == "")
        YString = to_string(VarCounter);
      VarCounter++;
      Constant *YValue = ConstantDataArray::getString(BB->getModule()->getContext(),
                                                      YString,
                                                      true);
      Value *Y = new GlobalVariable(*BB->getModule(),
                                    XValue->getType(),
                                    true,
                                    GlobalValue::InternalLinkage,
                                    YValue);

      Call1Args.push_back(X);
      Call1Args.push_back(CurrentInstruction->getOperand(0));
      Call1Args.push_back(Y);
      Call1Args.push_back(CurrentInstruction->getOperand(1));

      Call2Args.push_back(X);
      Call2Args.push_back(CurrentInstruction->getOperand(0));
      Call2Args.push_back(Y);
      Call2Args.push_back(CurrentInstruction->getOperand(1));

      Operation OpType;
      switch (CurrentInstruction->getOpcode()) {
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
        errs() << CurrentInstruction << " is not a Binary Instruction";
        break;
      }

      Call1Args.push_back(InstructionBuilder.getInt32(OpType));
      Call2Args.push_back(InstructionBuilder.getInt32(OpType));

      Call1Args.push_back(InstructionBuilder.getInt32(1));
      Call2Args.push_back(InstructionBuilder.getInt32(2));


      ArrayRef<Value *> Call1ArgsRef(Call1Args);
      ArrayRef<Value *> Call2ArgsRef(Call2Args);

      CallInst *NewCall1Instruction;
      CallInst *NewCall2Instruction;

      // Branch based on data type of operation
      if(isSingleFPOperation(CurrentInstruction)) {
        NewCall1Instruction =
            InstructionBuilder.CreateCall(ACfp32BinaryFunction, Call1ArgsRef);
        NewCall2Instruction =
            InstructionBuilder.CreateCall(ACfp32BinaryFunction, Call2ArgsRef);
      }
      else if(isDoubleFPOperation(CurrentInstruction)) {
        NewCall1Instruction =
            InstructionBuilder.CreateCall(ACfp64BinaryFunction, Call1ArgsRef);
        NewCall2Instruction =
            InstructionBuilder.CreateCall(ACfp64BinaryFunction, Call2ArgsRef);
      }

      NuminstrumentedOps++;

      assert(NewCall1Instruction && NewCall2Instruction && "Invalid call instruction!");
    }
    else if(isCastOperation(CurrentInstruction)) {
      
    }

  }
  *NumInstrumentedInstructions = NuminstrumentedOps;

}

void ACInstrumentation::instrumentMainFunction(Function *F) {
  BasicBlock *BB = &(*(F->begin()));
  Instruction *Inst = BB->getFirstNonPHIOrDbg();
  IRBuilder<> InstructionBuilder(Inst);
  std::vector<Value *> Args;

  CallInst *CallInstruction = nullptr;
  Args.push_back(InstructionBuilder.getInt64(1000));
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
  CallInstruction = InstructionBuilder.CreateCall(ACInitFunction, Args);
//  }

  assert(CallInstruction && "Invalid call instruction!");
}

// TODO: Figure out which unary instructions you would want the atomic condition
//  for and return true for those below.
bool ACInstrumentation::isUnaryOperation(const Instruction *Inst) {
  return false;
}

// TODO: Check for FRem case.
bool ACInstrumentation::isBinaryOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FAdd ||
         Inst->getOpcode() == Instruction::FSub ||
         Inst->getOpcode() == Instruction::FMul ||
         Inst->getOpcode() == Instruction::FDiv;
}

// TODO: Figure out which cast instructions you would want the atomic condition
//  for and return true for those below.
bool ACInstrumentation::isCastOperation(const Instruction *Inst) {
  return false;
}

bool ACInstrumentation::isFPOperation(const Instruction *Inst) {
  return Inst->getOpcode() == Instruction::FMul ||
         Inst->getOpcode() == Instruction::FDiv ||
         Inst->getOpcode() == Instruction::FAdd ||
         Inst->getOpcode() == Instruction::FSub;
}

bool ACInstrumentation::isSingleFPOperation(const Instruction *Inst) {
  return isFPOperation(Inst) &&
         Inst->getOperand(0)->getType()->isFloatTy() &&
         Inst->getOperand(1)->getType()->isFloatTy();
}

bool ACInstrumentation::isDoubleFPOperation(const Instruction *Inst) {
  return isFPOperation(Inst) &&
         Inst->getOperand(0)->getType()->isDoubleTy() &&
         Inst->getOperand(1)->getType()->isDoubleTy();
}

bool ACInstrumentation::isUnwantedFunction(const Function *Func) {
  return Func->getName().str().find("fAC") != std::string::npos ||
         Func->getName().str().find("ACItem") != std::string::npos;
}
