//
// Created by tanmay on 10/19/22.
//

#ifndef LLVM_COMPILETIMEUTILITIES_H
#define LLVM_COMPILETIMEUTILITIES_H

#include "llvm/IR/ValueMap.h"

using namespace llvm;

namespace atomiccondition {

// Instruction finders
Instruction *getInstructionAfterInitializationCalls(BasicBlock *BB);

Value *createBBNameGlobalString(BasicBlock *BB);
Value *createRegisterNameGlobalString(Instruction *Inst, Module *M);
Value *createInstructionGlobalString(Instruction *Inst);
Value *createStringRefGlobalString(StringRef StringObj, Instruction *Inst);
std::string getInstructionAsString(Instruction *Inst);
bool isInstructionOfInterest(Instruction *Inst);
int getFunctionEnum(Instruction *Inst);

} // namespace atomiccondition

#endif // LLVM_COMPILETIMEUTILITIES_H
