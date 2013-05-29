//===-- GenericToNVVM.cpp - Convert generic module to NVVM module - C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Convert generic global variables into either .global or .const access based
// on the variable's "constant" qualifier.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"

#include "llvm/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace llvm {
void initializeGenericToNVVMPass(PassRegistry &);
}

namespace {
class GenericToNVVM : public ModulePass {
public:
  static char ID;

  GenericToNVVM() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
  }

private:
  Value *getOrInsertCVTA(Module *M, Function *F, GlobalVariable *GV,
                         IRBuilder<> &Builder);
  Value *remapConstant(Module *M, Function *F, Constant *C,
                       IRBuilder<> &Builder);
  Value *remapConstantVectorOrConstantAggregate(Module *M, Function *F,
                                                Constant *C,
                                                IRBuilder<> &Builder);
  Value *remapConstantExpr(Module *M, Function *F, ConstantExpr *C,
                           IRBuilder<> &Builder);
  void remapNamedMDNode(Module *M, NamedMDNode *N);
  MDNode *remapMDNode(Module *M, MDNode *N);

  typedef ValueMap<GlobalVariable *, GlobalVariable *> GVMapTy;
  typedef ValueMap<Constant *, Value *> ConstantToValueMapTy;
  GVMapTy GVMap;
  ConstantToValueMapTy ConstantToValueMap;
};
}

char GenericToNVVM::ID = 0;

ModulePass *llvm::createGenericToNVVMPass() { return new GenericToNVVM(); }

INITIALIZE_PASS(
    GenericToNVVM, "generic-to-nvvm",
    "Ensure that the global variables are in the global address space", false,
    false)

bool GenericToNVVM::runOnModule(Module &M) {
  // Create a clone of each global variable that has the default address space.
  // The clone is created with the global address space  specifier, and the pair
  // of original global variable and its clone is placed in the GVMap for later
  // use.

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E;) {
    GlobalVariable *GV = I++;
    if (GV->getType()->getAddressSpace() == llvm::ADDRESS_SPACE_GENERIC &&
        !llvm::isTexture(*GV) && !llvm::isSurface(*GV) &&
        !GV->getName().startswith("llvm.")) {
      GlobalVariable *NewGV = new GlobalVariable(
          M, GV->getType()->getElementType(), GV->isConstant(),
          GV->getLinkage(), GV->hasInitializer() ? GV->getInitializer() : NULL,
          "", GV, GV->getThreadLocalMode(), llvm::ADDRESS_SPACE_GLOBAL);
      NewGV->copyAttributesFrom(GV);
      GVMap[GV] = NewGV;
    }
  }

  // Return immediately, if every global variable has a specific address space
  // specifier.
  if (GVMap.empty()) {
    return false;
  }

  // Walk through the instructions in function defitinions, and replace any use
  // of original global variables in GVMap with a use of the corresponding
  // copies in GVMap.  If necessary, promote constants to instructions.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (I->isDeclaration()) {
      continue;
    }
    IRBuilder<> Builder(I->getEntryBlock().getFirstNonPHIOrDbg());
    for (Function::iterator BBI = I->begin(), BBE = I->end(); BBI != BBE;
         ++BBI) {
      for (BasicBlock::iterator II = BBI->begin(), IE = BBI->end(); II != IE;
           ++II) {
        for (unsigned i = 0, e = II->getNumOperands(); i < e; ++i) {
          Value *Operand = II->getOperand(i);
          if (isa<Constant>(Operand)) {
            II->setOperand(
                i, remapConstant(&M, I, cast<Constant>(Operand), Builder));
          }
        }
      }
    }
    ConstantToValueMap.clear();
  }

  // Walk through the metadata section and update the debug information
  // associated with the global variables in the default address space.
  for (Module::named_metadata_iterator I = M.named_metadata_begin(),
                                       E = M.named_metadata_end();
       I != E; I++) {
    remapNamedMDNode(&M, I);
  }

  // Walk through the global variable  initializers, and replace any use of
  // original global variables in GVMap with a use of the corresponding copies
  // in GVMap.  The copies need to be bitcast to the original global variable
  // types, as we cannot use cvta in global variable initializers.
  for (GVMapTy::iterator I = GVMap.begin(), E = GVMap.end(); I != E;) {
    GlobalVariable *GV = I->first;
    GlobalVariable *NewGV = I->second;
    ++I;
    Constant *BitCastNewGV = ConstantExpr::getBitCast(NewGV, GV->getType());
    // At this point, the remaining uses of GV should be found only in global
    // variable initializers, as other uses have been already been removed
    // while walking through the instructions in function definitions.
    for (Value::use_iterator UI = GV->use_begin(), UE = GV->use_end();
         UI != UE;) {
      Use &U = (UI++).getUse();
      U.set(BitCastNewGV);
    }
    std::string Name = GV->getName();
    GV->removeDeadConstantUsers();
    GV->eraseFromParent();
    NewGV->setName(Name);
  }
  GVMap.clear();

  return true;
}

Value *GenericToNVVM::getOrInsertCVTA(Module *M, Function *F,
                                      GlobalVariable *GV,
                                      IRBuilder<> &Builder) {
  PointerType *GVType = GV->getType();
  Value *CVTA = NULL;

  // See if the address space conversion requires the operand to be bitcast
  // to i8 addrspace(n)* first.
  EVT ExtendedGVType = EVT::getEVT(GVType->getElementType(), true);
  if (!ExtendedGVType.isInteger() && !ExtendedGVType.isFloatingPoint()) {
    // A bitcast to i8 addrspace(n)* on the operand is needed.
    LLVMContext &Context = M->getContext();
    unsigned int AddrSpace = GVType->getAddressSpace();
    Type *DestTy = PointerType::get(Type::getInt8Ty(Context), AddrSpace);
    CVTA = Builder.CreateBitCast(GV, DestTy, "cvta");
    // Insert the address space conversion.
    Type *ResultType =
        PointerType::get(Type::getInt8Ty(Context), llvm::ADDRESS_SPACE_GENERIC);
    SmallVector<Type *, 2> ParamTypes;
    ParamTypes.push_back(ResultType);
    ParamTypes.push_back(DestTy);
    Function *CVTAFunction = Intrinsic::getDeclaration(
        M, Intrinsic::nvvm_ptr_global_to_gen, ParamTypes);
    CVTA = Builder.CreateCall(CVTAFunction, CVTA, "cvta");
    // Another bitcast from i8 * to <the element type of GVType> * is
    // required.
    DestTy =
        PointerType::get(GVType->getElementType(), llvm::ADDRESS_SPACE_GENERIC);
    CVTA = Builder.CreateBitCast(CVTA, DestTy, "cvta");
  } else {
    // A simple CVTA is enough.
    SmallVector<Type *, 2> ParamTypes;
    ParamTypes.push_back(PointerType::get(GVType->getElementType(),
                                          llvm::ADDRESS_SPACE_GENERIC));
    ParamTypes.push_back(GVType);
    Function *CVTAFunction = Intrinsic::getDeclaration(
        M, Intrinsic::nvvm_ptr_global_to_gen, ParamTypes);
    CVTA = Builder.CreateCall(CVTAFunction, GV, "cvta");
  }

  return CVTA;
}

Value *GenericToNVVM::remapConstant(Module *M, Function *F, Constant *C,
                                    IRBuilder<> &Builder) {
  // If the constant C has been converted already in the given function  F, just
  // return the converted value.
  ConstantToValueMapTy::iterator CTII = ConstantToValueMap.find(C);
  if (CTII != ConstantToValueMap.end()) {
    return CTII->second;
  }

  Value *NewValue = C;
  if (isa<GlobalVariable>(C)) {
    // If the constant C is a global variable and is found in  GVMap, generate a
    // set set of instructions that convert the clone of C with the global
    // address space specifier to a generic pointer.
    // The constant C cannot be used here, as it will be erased from the
    // module eventually.  And the clone of C with the global address space
    // specifier cannot be used here either, as it will affect the types of
    // other instructions in the function.  Hence, this address space conversion
    // is required.
    GVMapTy::iterator I = GVMap.find(cast<GlobalVariable>(C));
    if (I != GVMap.end()) {
      NewValue = getOrInsertCVTA(M, F, I->second, Builder);
    }
  } else if (isa<ConstantVector>(C) || isa<ConstantArray>(C) ||
             isa<ConstantStruct>(C)) {
    // If any element in the constant vector or aggregate C is or uses a global
    // variable in GVMap, the constant C needs to be reconstructed, using a set
    // of instructions.
    NewValue = remapConstantVectorOrConstantAggregate(M, F, C, Builder);
  } else if (isa<ConstantExpr>(C)) {
    // If any operand in the constant expression C is or uses a global variable
    // in GVMap, the constant expression C needs to be reconstructed, using a
    // set of instructions.
    NewValue = remapConstantExpr(M, F, cast<ConstantExpr>(C), Builder);
  }

  ConstantToValueMap[C] = NewValue;
  return NewValue;
}

Value *GenericToNVVM::remapConstantVectorOrConstantAggregate(
    Module *M, Function *F, Constant *C, IRBuilder<> &Builder) {
  bool OperandChanged = false;
  SmallVector<Value *, 4> NewOperands;
  unsigned NumOperands = C->getNumOperands();

  // Check if any element is or uses a global variable in  GVMap, and thus
  // converted to another value.
  for (unsigned i = 0; i < NumOperands; ++i) {
    Value *Operand = C->getOperand(i);
    Value *NewOperand = remapConstant(M, F, cast<Constant>(Operand), Builder);
    OperandChanged |= Operand != NewOperand;
    NewOperands.push_back(NewOperand);
  }

  // If none of the elements has been modified, return C as it is.
  if (!OperandChanged) {
    return C;
  }

  // If any of the elements has been  modified, construct the equivalent
  // vector or aggregate value with a set instructions and the converted
  // elements.
  Value *NewValue = UndefValue::get(C->getType());
  if (isa<ConstantVector>(C)) {
    for (unsigned i = 0; i < NumOperands; ++i) {
      Value *Idx = ConstantInt::get(Type::getInt32Ty(M->getContext()), i);
      NewValue = Builder.CreateInsertElement(NewValue, NewOperands[i], Idx);
    }
  } else {
    for (unsigned i = 0; i < NumOperands; ++i) {
      NewValue =
          Builder.CreateInsertValue(NewValue, NewOperands[i], makeArrayRef(i));
    }
  }

  return NewValue;
}

Value *GenericToNVVM::remapConstantExpr(Module *M, Function *F, ConstantExpr *C,
                                        IRBuilder<> &Builder) {
  bool OperandChanged = false;
  SmallVector<Value *, 4> NewOperands;
  unsigned NumOperands = C->getNumOperands();

  // Check if any operand is or uses a global variable in  GVMap, and thus
  // converted to another value.
  for (unsigned i = 0; i < NumOperands; ++i) {
    Value *Operand = C->getOperand(i);
    Value *NewOperand = remapConstant(M, F, cast<Constant>(Operand), Builder);
    OperandChanged |= Operand != NewOperand;
    NewOperands.push_back(NewOperand);
  }

  // If none of the operands has been modified, return C as it is.
  if (!OperandChanged) {
    return C;
  }

  // If any of the operands has been modified, construct the instruction with
  // the converted operands.
  unsigned Opcode = C->getOpcode();
  switch (Opcode) {
  case Instruction::ICmp:
    // CompareConstantExpr (icmp)
    return Builder.CreateICmp(CmpInst::Predicate(C->getPredicate()),
                              NewOperands[0], NewOperands[1]);
  case Instruction::FCmp:
    // CompareConstantExpr (fcmp)
    assert(false && "Address space conversion should have no effect "
                    "on float point CompareConstantExpr (fcmp)!");
    return C;
  case Instruction::ExtractElement:
    // ExtractElementConstantExpr
    return Builder.CreateExtractElement(NewOperands[0], NewOperands[1]);
  case Instruction::InsertElement:
    // InsertElementConstantExpr
    return Builder.CreateInsertElement(NewOperands[0], NewOperands[1],
                                       NewOperands[2]);
  case Instruction::ShuffleVector:
    // ShuffleVector
    return Builder.CreateShuffleVector(NewOperands[0], NewOperands[1],
                                       NewOperands[2]);
  case Instruction::ExtractValue:
    // ExtractValueConstantExpr
    return Builder.CreateExtractValue(NewOperands[0], C->getIndices());
  case Instruction::InsertValue:
    // InsertValueConstantExpr
    return Builder.CreateInsertValue(NewOperands[0], NewOperands[1],
                                     C->getIndices());
  case Instruction::GetElementPtr:
    // GetElementPtrConstantExpr
    return cast<GEPOperator>(C)->isInBounds()
               ? Builder.CreateGEP(
                     NewOperands[0],
                     makeArrayRef(&NewOperands[1], NumOperands - 1))
               : Builder.CreateInBoundsGEP(
                     NewOperands[0],
                     makeArrayRef(&NewOperands[1], NumOperands - 1));
  case Instruction::Select:
    // SelectConstantExpr
    return Builder.CreateSelect(NewOperands[0], NewOperands[1], NewOperands[2]);
  default:
    // BinaryConstantExpr
    if (Instruction::isBinaryOp(Opcode)) {
      return Builder.CreateBinOp(Instruction::BinaryOps(C->getOpcode()),
                                 NewOperands[0], NewOperands[1]);
    }
    // UnaryConstantExpr
    if (Instruction::isCast(Opcode)) {
      return Builder.CreateCast(Instruction::CastOps(C->getOpcode()),
                                NewOperands[0], C->getType());
    }
    assert(false && "GenericToNVVM encountered an unsupported ConstantExpr");
    return C;
  }
}

void GenericToNVVM::remapNamedMDNode(Module *M, NamedMDNode *N) {

  bool OperandChanged = false;
  SmallVector<MDNode *, 16> NewOperands;
  unsigned NumOperands = N->getNumOperands();

  // Check if any operand is or contains a global variable in  GVMap, and thus
  // converted to another value.
  for (unsigned i = 0; i < NumOperands; ++i) {
    MDNode *Operand = N->getOperand(i);
    MDNode *NewOperand = remapMDNode(M, Operand);
    OperandChanged |= Operand != NewOperand;
    NewOperands.push_back(NewOperand);
  }

  // If none of the operands has been modified, return immediately.
  if (!OperandChanged) {
    return;
  }

  // Replace the old operands with the new operands.
  N->dropAllReferences();
  for (SmallVector<MDNode *, 16>::iterator I = NewOperands.begin(),
                                           E = NewOperands.end();
       I != E; ++I) {
    N->addOperand(*I);
  }
}

MDNode *GenericToNVVM::remapMDNode(Module *M, MDNode *N) {

  bool OperandChanged = false;
  SmallVector<Value *, 8> NewOperands;
  unsigned NumOperands = N->getNumOperands();

  // Check if any operand is or contains a global variable in  GVMap, and thus
  // converted to another value.
  for (unsigned i = 0; i < NumOperands; ++i) {
    Value *Operand = N->getOperand(i);
    Value *NewOperand = Operand;
    if (Operand) {
      if (isa<GlobalVariable>(Operand)) {
        GVMapTy::iterator I = GVMap.find(cast<GlobalVariable>(Operand));
        if (I != GVMap.end()) {
          NewOperand = I->second;
          if (++i < NumOperands) {
            NewOperands.push_back(NewOperand);
            // Address space of the global variable follows the global variable
            // in the global variable debug info (see createGlobalVariable in
            // lib/Analysis/DIBuilder.cpp).
            NewOperand =
                ConstantInt::get(Type::getInt32Ty(M->getContext()),
                                 I->second->getType()->getAddressSpace());
          }
        }
      } else if (isa<MDNode>(Operand)) {
        NewOperand = remapMDNode(M, cast<MDNode>(Operand));
      }
    }
    OperandChanged |= Operand != NewOperand;
    NewOperands.push_back(NewOperand);
  }

  // If none of the operands has been modified, return N as it is.
  if (!OperandChanged) {
    return N;
  }

  // If any of the operands has been modified, create a new MDNode with the new
  // operands.
  return MDNode::get(M->getContext(), makeArrayRef(NewOperands));
}
