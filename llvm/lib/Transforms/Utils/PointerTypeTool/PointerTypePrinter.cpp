#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypePrinter.h"

using namespace llvm;

void PointerTypePrinter::printModule(Module& M) {
  OS << "target datalayout = \"" << M.getDataLayoutStr() << "\"\n";
  OS << "target triple = \"" << M.getTargetTriple().str() << "\"\n\n";

  for (auto st : M.getIdentifiedStructTypes()) {
    st->print(OS, false, true);
    OS << " = type ";
    if (st->isOpaque()) {
      OS << "opaque\n";
    } else {
      OS << "{ ";
      auto &mst = FA.getStructInfo().at(st);
      auto cnt = mst->getElementCnt();
      if (cnt != 0) {
        OS << mst->getElementTy()->toString();
        for (auto i = 1u; i < cnt; i++) {
          OS << ", " << mst->getElementTy(i)->toString();
        }
        OS << " ";
      }
      OS << "}\n";
    }
  }
  
  if (M.getIdentifiedStructTypes().size()) {
    OS << "\n";
  }

  for (auto &gl : M.globals()) {
    printValue(&gl);
    OS << " = ";
    if (gl.hasInitializer()) {
      printValue(gl.getInitializer(), true);
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

  // Todo: Print function type.
  OS << " @" << F.getName() << "(";

  for (auto Arg = F.arg_begin(); Arg != F.arg_end(); ++Arg) {
    if (Arg != F.arg_begin())
      OS << ", ";
    printValue(&*Arg, true, nullptr);
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
    printInstruction(&I);
  }
}

void PointerTypePrinter::printInstruction(Instruction *I) {
  OS << "  ";
  switch (I->getOpcode()) {
  case Instruction::Ret:
    printReturnInst(static_cast<ReturnInst *>(I));
    break;
  case Instruction::Switch:
    printSwitchInst(static_cast<SwitchInst *>(I));
    break;
  case Instruction::IndirectBr:
    printIndirectBrInst(static_cast<IndirectBrInst *>(I));
    break;
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    printBinaryOpInst(I);
    break;
  case Instruction::ExtractValue:
    printExtractValueInst(static_cast<ExtractValueInst *>(I));
    break;
  case Instruction::InsertValue:
    printInsertValueInst(static_cast<InsertValueInst *>(I));
    break;
  case Instruction::Alloca:
    printAllocaInst(static_cast<AllocaInst *>(I));
    break;
  case Instruction::Load:
    printLoadInst(static_cast<LoadInst *>(I));
    break;
  case Instruction::AtomicRMW:
    printAtomicRMWInst(static_cast<AtomicRMWInst *>(I));
    break;
  case Instruction::GetElementPtr:
    printGetElementPtrInst(static_cast<GetElementPtrInst *>(I));
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    printTurnToInst(I);
    break;
  case Instruction::ICmp:
  case Instruction::FCmp:
    printCmpInst(I);
    break;
  case Instruction::PHI:
    printPHINode(static_cast<PHINode *>(I));
    break;
  default:
    printCommonInst(I);
  }
  OS << "\n";
}

void PointerTypePrinter::printValue(Value *V, bool typed, BasicBlock *B) {
  if (typed) {
    printType(V, B);
    OS << " ";
  }
  V->printAsOperand(OS, false);
}

void PointerTypePrinter::printType(Value* V, BasicBlock* B) {
  if (B == nullptr) {
    V->getType()->print(OS, true);
    return;
  } else {
    auto &typeInfo = FA.getTypeInfo(B);
    assert(typeInfo.contains(V));
    OS << typeInfo.at(V)->toString();
  }
}

void PointerTypePrinter::printCommonInst(Instruction* I) {
  if (!I->getType()->isVoidTy()) {
    printValue(static_cast<Value *>(I));
    OS << " = ";
  }
  OS << I->getOpcodeName();
  for (auto i = 0u; i < I->getNumOperands(); i++) {
    if (i > 0) {
      OS << ",";
    }
    OS << " ";
    printValue(I->getOperand(i), true, I->getParent());
  }
}

void PointerTypePrinter::printReturnInst(ReturnInst *I) {
  OS << "ret ";
  auto val = I->getReturnValue();
  if (val == nullptr) {
    OS << "void";
  } else {
    printValue(val, true, I->getParent());
  }
}

void PointerTypePrinter::printSwitchInst(SwitchInst *I) {
  auto B = I->getParent();
  OS << "switch ";
  printValue(I->getCondition(), true, B);
  OS << ", label ";
  printValue(I->getDefaultDest());
  OS << " [ ";
  for (auto &C : I->cases()) {
    printValue(C.getCaseValue(), true, B);
    OS << ", label ";
    printValue(C.getCaseSuccessor());
    OS << " ";
  }
  OS << "]";
}

void PointerTypePrinter::printIndirectBrInst(IndirectBrInst *I) {
  OS << "indirectbr ";
  printValue(I->getAddress(), true, I->getParent());
  OS << ", [";
  for (auto i = 0u; i < I->getNumSuccessors(); i++) {
    if (i != 0) {
      OS << ",";
    }
    OS << " label ";
    printValue(I->getSuccessor(i));
  }
}

void PointerTypePrinter::printBinaryOpInst(Instruction* I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " ";
  printValue(I->getOperand(0), true, I->getParent());
  OS << ", ";
  printValue(I->getOperand(1));
}

void PointerTypePrinter::printExtractValueInst(ExtractValueInst* I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " ";
  printValue(I->getAggregateOperand(), true, I->getParent());
  for (auto i : I->indices()) {
    OS << ", " << i;
  }
}

void PointerTypePrinter::printInsertValueInst(InsertValueInst* I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " ";
  printValue(I->getAggregateOperand(), true, I->getParent());
  printValue(I->getInsertedValueOperand(), true, I->getParent());
  for (auto i : I->indices()) {
    OS << ", " << i;
  }
}

void PointerTypePrinter::printAllocaInst(AllocaInst *I) {
  auto res = static_cast<Value *>(I);
  printValue(res);
  OS << " = " << I->getOpcodeName() << " ";
  printType(res, I->getParent());
  int size = static_cast<ConstantInt *>(I->getArraySize())->getZExtValue();
  if (size > 1) {
    OS << ", " << size;
  }
}

void PointerTypePrinter::printLoadInst(LoadInst* I) {
  auto res = static_cast<Value *>(I);
  printValue(res);
  OS << " = " << I->getOpcodeName() << " ";
  printType(res, I->getParent());
  OS << ", ";
  printValue(I->getPointerOperand(), true, I->getParent());
}

void PointerTypePrinter::printAtomicRMWInst(AtomicRMWInst* I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " "
     << AtomicRMWInst::getOperationName(I->getOperation());
  for (auto i = 0u; i < I->getNumOperands(); i++) {
    if (i > 0) {
      OS << ",";
    }
    OS << " ";
    printValue(I->getOperand(i), true, I->getParent());
  }
}

void PointerTypePrinter::printGetElementPtrInst(GetElementPtrInst* I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " ";
  auto &type = FA.getTypeInfo(I->getParent()).at(I->getPointerOperand());
  if (type->isVector()) {
    auto ptrTy = MyTy::ptr_cast<MyVectorTy>(type)->getElementTy();
    OS << ptrTy->getPointeeTyAsPtr()->toString();
  } else {
    OS << type->getPointeeTyAsPtr()->toString();
  }
  for (auto i = 0u; i < I->getNumOperands(); i++) {
    OS << ", ";
    printValue(I->getOperand(i), true, I->getParent());
  }
}

void PointerTypePrinter::printTurnToInst(Instruction* I) {
  auto res = static_cast<Value *>(I);
  printValue(res);
  OS << " = " << I->getOpcodeName() << " ";
  printValue(I->getOperand(0), true, I->getParent());
  OS << " to ";
  printType(res, I->getParent());
}

void PointerTypePrinter::printCmpInst(Instruction *I) {
  printValue(static_cast<Value *>(I));
  OS << " = " << I->getOpcodeName() << " ";
  auto CI = static_cast<CmpInst *>(I);
  OS << CI->getPredicate() << " ";
  printValue(I->getOperand(0), true, I->getParent());
  OS << ", ";
  printValue(I->getOperand(1));
}

void PointerTypePrinter::printPHINode(PHINode* I) {
  auto res = static_cast<Value *>(I);
  printValue(res);
  OS << " = " << I->getOpcodeName() << " ";
  printType(res, I->getParent());
  for (auto i = 0; i < I->getNumIncomingValues(); i++) {
    if (i > 0) {
      OS << ",";
    }
    OS << " [ ";
    printValue(I->getIncomingValue(i));
    OS << ", ";
    printValue(I->getIncomingBlock(i));
    OS << " ]";
  }
}