#include "llvm/Transforms/Utils/PointerTypePrinter.h"

using namespace llvm;

void PointerTypePrinter::printModule(Module& M) {
  OS << "target datalayout = \"" << M.getDataLayoutStr() << "\"\n";
  OS << "target triple = \"" << M.getTargetTriple().str() << "\"\n\n";

  for (auto st : M.getIdentifiedStructTypes()) {
    st->print(OS, false, true);
    OS << " = type { ";
    auto mst = MyTy::ptr_cast<MyStructTy>(helper.getStructInfo()[st]);
    auto cnt = mst->getElementCnt();
    if (cnt != 0) {
      OS << mst->getElementTy()->to_string();
      for (auto i = 1u; i < cnt; i++) {
        OS << ", " << mst->getElementTy(i)->to_string();
      }
      OS << " ";
    }
    OS << "}\n";
  }
  
  if (M.getIdentifiedStructTypes().size()) {
    OS << "\n";
  }

  for (auto &gl : M.globals()) {
    printValue(&gl);
    OS << " = ";
    printType(&gl, gl.getType());
    OS << " ";
    if (gl.hasInitializer()) {
      printValue(gl.getInitializer());
    }
    OS << "\n";
  }

  if (M.global_size()) {
    OS << "\n";
  }

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
  OS << ") ";

  if (F.isDeclaration()) {
    OS << "\n\n";
  } else {
    auto begin = true;
    for (BasicBlock &B : F) {
      B.printAsOperand(OS, false);
      if (begin) {
        OS << " {\n";
        begin = false;
      } else {
        OS << ":\n";
      }
      printBasicBlock(B);
    }
    OS << "}\n\n";
  }
}

void PointerTypePrinter::printBasicBlock(BasicBlock& B) {
  for (Instruction &I : B) {
    printInstruction(I);
  }
}

void PointerTypePrinter::printInstruction(Instruction& I) {
  OS << "  ";

  if (!I.getType()->isVoidTy()) {
    Value *v = static_cast<Value *>(&I);
    printValue(v);
    OS << " = ";
  }

  OS << I.getOpcodeName() << " ";

  if (!I.getType()->isVoidTy()) {
    auto ptm = helper.getPtm();
    if (I.getOpcode() == Instruction::GetElementPtr) {
      auto *GI = static_cast<GetElementPtrInst *>(&I);
      OS << MyTy::ptr_cast<MyPointerTy>(ptm[GI->getPointerOperand()])
                ->getInner()->to_string();
    } else if (I.getOpcode() == Instruction::Alloca) {
      Value *v = static_cast<Value *>(&I);
      OS << MyTy::ptr_cast<MyPointerTy>(ptm[v])
                ->getInner()->to_string();
    } else {
      Value *v = static_cast<Value *>(&I);
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
  auto ptm = helper.getPtm();
  if (ptm.contains(V)) {
    OS << ptm[V]->to_string();
  } else {
    T->print(OS);
  }
}

void PointerTypePrinter::load(PointerTypeHelpers helper) {
  this->helper = helper;
}