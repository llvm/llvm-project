#include "llvm/Transforms/Utils/PointerTypePrinter.h"

using namespace llvm;

void PointerTypePrinter::printModule(Module& M) {
  OS << "target datalayout = \"" << M.getDataLayoutStr() << "\"\n";
  OS << "target triple = \"" << M.getTargetTriple().str() << "\"\n\n";

  for (Function &F : M) {
    printFunction(F);
  }
}

void PointerTypePrinter::printFunction(Function& F) {
  if (F.isDeclaration()) {
    OS << "declare ";
  } else {
    OS << "define ";
  }

  printType(&F, F.getReturnType());
  OS << " @" << F.getName() << "(";

  for (auto Arg = F.arg_begin(); Arg != F.arg_end(); ++Arg) {
    if (Arg != F.arg_begin())
      OS << ", ";
    printType(&*Arg, Arg->getType());
    OS << " %" << Arg->getName();
  }
  OS << ")";

  if (F.isDeclaration()) {
    OS << "\n\n";
  } else {
    OS << " {\n";
    for (BasicBlock &B : F) {
      printBasicBlock(B);
    }
    OS << "}\n\n";
  }
}

void PointerTypePrinter::printBasicBlock(BasicBlock& B) {
  B.printAsOperand(OS, false);
  OS << ":\n";
  for (Instruction &I : B) {
    printInstruction(I);
  }
}

void PointerTypePrinter::printInstruction(Instruction& I) {
  OS << "  ";

  if (!I.getType()->isVoidTy()) {
    Value *v = cast<Value>(&I);
    printValue(v);
    OS << " = ";
  }

  OS << I.getOpcodeName() << " ";

  if (!I.getType()->isVoidTy()) {
    if (I.getOpcode() == Instruction::GetElementPtr) {
      auto *GI = cast<GetElementPtrInst>(&I);
      OS << MyTy::ptr_cast<MyPointerTy>(pointerTypeMap[GI->getPointerOperand()])
                ->getInner()->to_string();
    } else if (I.getOpcode() == Instruction::Alloca) {
      Value *v = cast<Value>(&I);
      OS << MyTy::ptr_cast<MyPointerTy>(pointerTypeMap[v])
                ->getInner()->to_string();
    } else {
      Value *v = cast<Value>(&I);
      printType(v, I.getType());
    }
    OS << ", ";
  }

  for (unsigned i = 0; i < I.getNumOperands(); i++) {
    if (i > 0)
      OS << ", ";
    Value *Operand = I.getOperand(i);
    if (I.getOpcode() != Instruction::Alloca) {
        printType(Operand, Operand->getType());
        OS << " ";
    }
    printValue(Operand);
  }

  OS << "\n";
}

void PointerTypePrinter::printValue(Value *V) {
  V->printAsOperand(OS, false);
}

void PointerTypePrinter::printType(Value *V, Type *T) {
  if (pointerTypeMap.contains(V)) {
    OS << pointerTypeMap[V]->to_string();
  } else {
    T->print(OS);
  }
}

void PointerTypePrinter::loadPointerTypeMap(
    DenseMap< Value *, std::shared_ptr<MyTy> > m) {
  for (auto &entry : m) {
    pointerTypeMap.insert(entry);
  }
}