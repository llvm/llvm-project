//===-- SITypeRewriter.cpp - Remove unwanted types ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass removes performs the following type substitution on all
/// non-compute shaders:
///
/// v16i8 => i128
///   - v16i8 is used for constant memory resource descriptors.  This type is
///      legal for some compute APIs, and we don't want to declare it as legal
///      in the backend, because we want the legalizer to expand all v16i8
///      operations.
/// v1* => *
///   - Having v1* types complicates the legalizer and we can easily replace
///   - them with the element type.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"

using namespace llvm;

namespace {

class SITypeRewriter : public FunctionPass,
                       public InstVisitor<SITypeRewriter> {

  static char ID;
  Module *Mod;
  Type *v16i8;
  Type *v4i32;

public:
  SITypeRewriter() : FunctionPass(ID) { }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return "SI Type Rewriter"; }
  void visitLoadInst(LoadInst &I);
  void visitCallInst(CallInst &I);
  void visitBitCast(BitCastInst &I);
};

} // End anonymous namespace

char SITypeRewriter::ID = 0;

bool SITypeRewriter::doInitialization(Module &M) {
  Mod = &M;
  v16i8 = VectorType::get(Type::getInt8Ty(M.getContext()), 16);
  v4i32 = VectorType::get(Type::getInt32Ty(M.getContext()), 4);
  return false;
}

bool SITypeRewriter::runOnFunction(Function &F) {
  if (!AMDGPU::isShader(F.getCallingConv()))
    return false;

  visit(F);
  visit(F);

  return false;
}

void SITypeRewriter::visitLoadInst(LoadInst &I) {
  Value *Ptr = I.getPointerOperand();
  Type *PtrTy = Ptr->getType();
  Type *ElemTy = PtrTy->getPointerElementType();
  IRBuilder<> Builder(&I);
  if (ElemTy == v16i8)  {
    Value *BitCast = Builder.CreateBitCast(Ptr,
        PointerType::get(v4i32,PtrTy->getPointerAddressSpace()));
    LoadInst *Load = Builder.CreateLoad(BitCast);
    SmallVector<std::pair<unsigned, MDNode *>, 8> MD;
    I.getAllMetadataOtherThanDebugLoc(MD);
    for (unsigned i = 0, e = MD.size(); i != e; ++i) {
      Load->setMetadata(MD[i].first, MD[i].second);
    }
    Value *BitCastLoad = Builder.CreateBitCast(Load, I.getType());
    I.replaceAllUsesWith(BitCastLoad);
    I.eraseFromParent();
  }
}

void SITypeRewriter::visitCallInst(CallInst &I) {
  IRBuilder<> Builder(&I);

  SmallVector <Value*, 8> Args;
  SmallVector <Type*, 8> Types;
  bool NeedToReplace = false;
  Function *F = I.getCalledFunction();
  if (!F)
    return;

  std::string Name = F->getName();
  for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
    Value *Arg = I.getArgOperand(i);
    if (Arg->getType() == v16i8) {
      Args.push_back(Builder.CreateBitCast(Arg, v4i32));
      Types.push_back(v4i32);
      NeedToReplace = true;
      Name = Name + ".v4i32";
    } else if (Arg->getType()->isVectorTy() &&
               Arg->getType()->getVectorNumElements() == 1 &&
               Arg->getType()->getVectorElementType() ==
                                              Type::getInt32Ty(I.getContext())){
      Type *ElementTy = Arg->getType()->getVectorElementType();
      std::string TypeName = "i32";
      InsertElementInst *Def = cast<InsertElementInst>(Arg);
      Args.push_back(Def->getOperand(1));
      Types.push_back(ElementTy);
      std::string VecTypeName = "v1" + TypeName;
      Name = Name.replace(Name.find(VecTypeName), VecTypeName.length(), TypeName);
      NeedToReplace = true;
    } else {
      Args.push_back(Arg);
      Types.push_back(Arg->getType());
    }
  }

  if (!NeedToReplace) {
    return;
  }
  Function *NewF = Mod->getFunction(Name);
  if (!NewF) {
    NewF = Function::Create(FunctionType::get(F->getReturnType(), Types, false), GlobalValue::ExternalLinkage, Name, Mod);
    NewF->setAttributes(F->getAttributes());
  }
  I.replaceAllUsesWith(Builder.CreateCall(NewF, Args));
  I.eraseFromParent();
}

void SITypeRewriter::visitBitCast(BitCastInst &I) {
  IRBuilder<> Builder(&I);
  if (I.getDestTy() != v4i32) {
    return;
  }

  if (BitCastInst *Op = dyn_cast<BitCastInst>(I.getOperand(0))) {
    if (Op->getSrcTy() == v4i32) {
      I.replaceAllUsesWith(Op->getOperand(0));
      I.eraseFromParent();
    }
  }
}

FunctionPass *llvm::createSITypeRewriter() {
  return new SITypeRewriter();
}
