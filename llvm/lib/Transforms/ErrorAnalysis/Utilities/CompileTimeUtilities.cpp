#include <string>
#include "llvm/Transforms/ErrorAnalysis/Utilities/CompileTimeUtilities.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h"
#include "llvm/Transforms/ErrorAnalysis/Utilities/FunctionMatchers.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DebugInfo.h"

using namespace llvm;
using namespace atomiccondition;
using namespace std;

namespace atomiccondition {

// Get Instruction after any Phi/Dbg instructions AND Atomic Condition and
// Computation Graph Initialization Calls.
Instruction *getInstructionAfterInitializationCalls(BasicBlock *BB) {
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

Value *createBBNameGlobalString(BasicBlock *BB) {
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

Value *createRegisterNameGlobalString(Instruction *Inst, Module *M) {
  string RegisterString;
  raw_string_ostream RawRegisterString(RegisterString);
  Inst->printAsOperand(RawRegisterString, false);

  Constant *RegisterValue = ConstantDataArray::getString(Inst->getContext(),
                                                         RawRegisterString.str().c_str(),
                                                         true);

  Value *RegisterValuePointer = new GlobalVariable(*M,
                                                   RegisterValue->getType(),
                                                   true,
                                                   GlobalValue::InternalLinkage,
                                                   RegisterValue);

  return RegisterValuePointer;
}

Value *createInstructionGlobalString(Instruction *Inst) {
  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *Inst;
  unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  string Initializer = (NonEmptyPosition == std::string::npos) ? "" :
                                                               RawInstructionString.str().substr(NonEmptyPosition);

  Constant *InstructionValue = ConstantDataArray::getString(Inst->getContext(),
                                                            Initializer.c_str(),
                                                            true);

  Value *InstructionValuePointer = new GlobalVariable(*Inst->getModule(),
                                                      InstructionValue->getType(),
                                                      true,
                                                      GlobalValue::InternalLinkage,
                                                      InstructionValue);

  return InstructionValuePointer;
}

Value *createStringRefGlobalString(StringRef StringObj, Instruction *Inst) {
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

string getInstructionAsString(Instruction *Inst) {
  string InstructionString;
  raw_string_ostream RawInstructionString(InstructionString);
  RawInstructionString << *Inst;
  unsigned long NonEmptyPosition= RawInstructionString.str().find_first_not_of(" \n\r\t\f\v");
  string InstructionAsString = (NonEmptyPosition == std::string::npos) ? "" :
                                                                       RawInstructionString.str().substr(NonEmptyPosition);
  return InstructionAsString;
}

bool isInstructionOfInterest(Instruction *Inst) {
  switch (Inst->getOpcode()) {
  case 14:
  case 16:
  case 18:
  case 21:
  case 31:
  case 32:
  case 55:
  case 56:
  case 57:
    return true;
  }
  return false;
}

bool isUnaryFunction(const Function *Func) {
  if (Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return isASinFunction(FunctionName) ||
           isACosFunction(FunctionName) ||
           isATanFunction(FunctionName) ||
           isSinFunction(FunctionName) ||
           isCosFunction(FunctionName) ||
           isTanFunction(FunctionName) ||
           isSinhFunction(FunctionName) ||
           isCoshFunction(FunctionName) ||
           isTanhFunction(FunctionName) ||
           isExpFunction(FunctionName) ||
           isLogFunction(FunctionName) ||
           isSqrtFunction(FunctionName);
  }
  return false;
}

bool isBinaryFunction(const Function *Func) {
  if (Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return isAddFunction(FunctionName) ||
           isSubFunction(FunctionName) ||
           isMulFunction(FunctionName) ||
           isDivFunction(FunctionName);
  }
  return false;
}

bool isTernaryFunction(const Function *Func) {
  if (Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return isFMAFuncton(FunctionName);
  }
  return false;
}

bool isFunctionOfInterest(const Function *Func) {
  if (Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return isUnaryFunction(Func) ||
           isBinaryFunction(Func) ||
           isTernaryFunction(Func);
  }
  return false;
}

bool isCPFloatFunction(const Function *Func) {
  if (Func->hasName()) {
    string FunctionName = Func->getName().str();
    transform(FunctionName.begin(), FunctionName.end(), FunctionName.begin(),
              ::tolower);
    return isCPFloatFunction(FunctionName);
  }
  return false;
}


// Functions defined in AtomicCondition library
int getFunctionEnum(Instruction *Inst) {
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
    if(isASinFunction(FunctionName))
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
    if(isAddFunction(FunctionName))
      return Func::Add;
    if(isSubFunction(FunctionName))
      return Func::Sub;
    if(isMulFunction(FunctionName))
      return Func::Mul;
    if(isDivFunction(FunctionName))
      return Func::Div;
    if(isFMAFuncton(FunctionName))
      return Func::FMA;
    return -1;
  default:
    //    errs() << BaseInstruction << " is not a Binary Instruction.\n";
    return -1;
  }
}

int getFunctionNumOperands(int FunctionEnum) {
  switch (FunctionEnum) {
    case Func::Neg:
    case Func::ArcSin:
    case Func::ArcCos:
    case Func::ArcTan:
    case Func::Sinh:
    case Func::Cosh:
    case Func::Tanh:
    case Func::Sin:
    case Func::Cos:
    case Func::Tan:
    case Func::Exp:
    case Func::Log:
    case Func::Sqrt:
      return 1;
    case Func::Add:
    case Func::Sub:
    case Func::Mul:
    case Func::Div:
      return 2;
    case Func::FMA:
      return 3;
    default:
      return -1;
  }
}

} // namespace atomiccondition