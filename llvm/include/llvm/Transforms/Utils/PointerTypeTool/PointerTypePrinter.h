#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPEPRINTER_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPEPRINTER_H

#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/PointerTypeTool/FlowAnalyzer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class PointerTypePrinter {
  raw_ostream &OS;
  FlowAnalyzer &FA;
  void printFunction(Function &F);
  void printBasicBlock(BasicBlock &B);
  void printInstruction(Instruction *I);
  void printValue(Value *V, bool typed = false, BasicBlock *B = nullptr);
  void printType(Value *V, BasicBlock *B = nullptr);
  void printCommonInst(Instruction *I);
  void printReturnInst(ReturnInst *I);
  void printSwitchInst(SwitchInst *I);
  void printIndirectBrInst(IndirectBrInst *I);
  void printBinaryOpInst(Instruction *I);
  void printExtractValueInst(ExtractValueInst *I);
  void printInsertValueInst(InsertValueInst *I);
  void printAllocaInst(AllocaInst *I);
  void printLoadInst(LoadInst *I);
  void printAtomicRMWInst(AtomicRMWInst *I);
  void printGetElementPtrInst(GetElementPtrInst *I);
  void printTurnToInst(Instruction *I);
  void printCmpInst(Instruction *I);
  void printPHINode(PHINode *I);

public:
  PointerTypePrinter(raw_ostream &Out, FlowAnalyzer &A) : OS(Out), FA(A) {}
  void printModule(Module &M);
};

}
#endif