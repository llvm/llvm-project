//===-- AMDGPUCodeGenPrepare.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. AMDGPU optimizations on IR before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"

#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-codegenprepare"

using namespace llvm;

namespace {

class AMDGPUCodeGenPrepare : public FunctionPass,
                             public InstVisitor<AMDGPUCodeGenPrepare, bool> {
  const GCNTargetMachine *TM;
  const SISubtarget *ST;
  DivergenceAnalysis *DA;
  Module *Mod;
  bool HasUnsafeFPMath;

  /// \brief Copies exact/nsw/nuw flags (if any) from binary operator \p I to
  /// binary operator \p V.
  ///
  /// \returns Binary operator \p V.
  Value *copyFlags(const BinaryOperator &I, Value *V) const;

  /// \returns Equivalent 16 bit integer type for given 32 bit integer type
  /// \p T.
  Type *getI16Ty(IRBuilder<> &B, const Type *T) const;

  /// \returns Equivalent 32 bit integer type for given 16 bit integer type
  /// \p T.
  Type *getI32Ty(IRBuilder<> &B, const Type *T) const;

  /// \returns True if the base element of type \p T is 16 bit integer, false
  /// otherwise.
  bool isI16Ty(const Type *T) const;

  /// \returns True if the base element of type \p T is 32 bit integer, false
  /// otherwise.
  bool isI32Ty(const Type *T) const;

  /// \returns True if binary operation \p I is a signed binary operation, false
  /// otherwise.
  bool isSigned(const BinaryOperator &I) const;

  /// \returns True if the condition of 'select' operation \p I comes from a
  /// signed 'icmp' operation, false otherwise.
  bool isSigned(const SelectInst &I) const;

  /// \brief Promotes uniform 16 bit binary operation \p I to equivalent 32 bit
  /// binary operation by sign or zero extending operands to 32 bits, replacing
  /// 16 bit operation with equivalent 32 bit operation, and truncating the
  /// result of 32 bit operation back to 16 bits. 16 bit division operation is
  /// not promoted.
  ///
  /// \returns True if 16 bit binary operation is promoted to equivalent 32 bit
  /// binary operation, false otherwise.
  bool promoteUniformI16OpToI32(BinaryOperator &I) const;

  /// \brief Promotes uniform 16 bit 'icmp' operation \p I to 32 bit 'icmp'
  /// operation by sign or zero extending operands to 32 bits, and replacing 16
  /// bit operation with 32 bit operation.
  ///
  /// \returns True.
  bool promoteUniformI16OpToI32(ICmpInst &I) const;

  /// \brief Promotes uniform 16 bit 'select' operation \p I to 32 bit 'select'
  /// operation by sign or zero extending operands to 32 bits, replacing 16 bit
  /// operation with 32 bit operation, and truncating the result of 32 bit
  /// operation back to 16 bits.
  ///
  /// \returns True.
  bool promoteUniformI16OpToI32(SelectInst &I) const;

  /// \brief Promotes uniform 16 bit 'bitreverse' intrinsic \p I to 32 bit
  /// 'bitreverse' intrinsic by zero extending operand to 32 bits, replacing 16
  /// bit intrinsic with 32 bit intrinsic, shifting the result of 32 bit
  /// intrinsic 16 bits to the right with zero fill, and truncating the result
  /// of shift operation back to 16 bits.
  ///
  /// \returns True.
  bool promoteUniformI16BitreverseIntrinsicToI32(IntrinsicInst &I) const;

public:
  static char ID;
  AMDGPUCodeGenPrepare(const TargetMachine *TM = nullptr) :
    FunctionPass(ID),
    TM(static_cast<const GCNTargetMachine *>(TM)),
    ST(nullptr),
    DA(nullptr),
    Mod(nullptr),
    HasUnsafeFPMath(false) { }

  bool visitFDiv(BinaryOperator &I);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitBinaryOperator(BinaryOperator &I);
  bool visitICmpInst(ICmpInst &I);
  bool visitSelectInst(SelectInst &I);

  bool visitIntrinsicInst(IntrinsicInst &I);
  bool visitBitreverseIntrinsicInst(IntrinsicInst &I);

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AMDGPU IR optimizations"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DivergenceAnalysis>();
    AU.setPreservesAll();
 }
};

} // End anonymous namespace

Value *AMDGPUCodeGenPrepare::copyFlags(
    const BinaryOperator &I, Value *V) const {
  assert(isa<BinaryOperator>(V) && "V must be binary operator");

  BinaryOperator *BinOp = cast<BinaryOperator>(V);
  if (isa<OverflowingBinaryOperator>(BinOp)) {
    BinOp->setHasNoSignedWrap(I.hasNoSignedWrap());
    BinOp->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
  } else if (isa<PossiblyExactOperator>(BinOp)) {
    BinOp->setIsExact(I.isExact());
  }

  return V;
}

Type *AMDGPUCodeGenPrepare::getI16Ty(IRBuilder<> &B, const Type *T) const {
  assert(isI32Ty(T) && "T must be 32 bits");

  if (T->isIntegerTy())
    return B.getInt16Ty();
  return VectorType::get(B.getInt16Ty(), cast<VectorType>(T)->getNumElements());
}

Type *AMDGPUCodeGenPrepare::getI32Ty(IRBuilder<> &B, const Type *T) const {
  assert(isI16Ty(T) && "T must be 16 bits");

  if (T->isIntegerTy())
    return B.getInt32Ty();
  return VectorType::get(B.getInt32Ty(), cast<VectorType>(T)->getNumElements());
}

bool AMDGPUCodeGenPrepare::isI16Ty(const Type *T) const {
  if (T->isIntegerTy(16))
    return true;
  if (!T->isVectorTy())
    return false;
  return cast<VectorType>(T)->getElementType()->isIntegerTy(16);
}

bool AMDGPUCodeGenPrepare::isI32Ty(const Type *T) const {
  if (T->isIntegerTy(32))
    return true;
  if (!T->isVectorTy())
    return false;
  return cast<VectorType>(T)->getElementType()->isIntegerTy(32);
}

bool AMDGPUCodeGenPrepare::isSigned(const BinaryOperator &I) const {
  return I.getOpcode() == Instruction::AShr ||
      I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::SRem;
}

bool AMDGPUCodeGenPrepare::isSigned(const SelectInst &I) const {
  return isa<ICmpInst>(I.getOperand(0)) ?
      cast<ICmpInst>(I.getOperand(0))->isSigned() : false;
}

bool AMDGPUCodeGenPrepare::promoteUniformI16OpToI32(BinaryOperator &I) const {
  assert(isI16Ty(I.getType()) && "I must be 16 bits");

  if (I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::UDiv)
    return false;

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
  }
  ExtRes = copyFlags(I, Builder.CreateBinOp(I.getOpcode(), ExtOp0, ExtOp1));
  TruncRes = Builder.CreateTrunc(ExtRes, getI16Ty(Builder, ExtRes->getType()));

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformI16OpToI32(ICmpInst &I) const {
  assert(isI16Ty(I.getOperand(0)->getType()) && "Op0 must be 16 bits");
  assert(isI16Ty(I.getOperand(1)->getType()) && "Op1 must be 16 bits");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32TyOp0 = getI32Ty(Builder, I.getOperand(0)->getType());
  Type *I32TyOp1 = getI32Ty(Builder, I.getOperand(1)->getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *NewICmp  = nullptr;

  if (I.isSigned()) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32TyOp0);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32TyOp1);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32TyOp0);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32TyOp1);
  }
  NewICmp = Builder.CreateICmp(I.getPredicate(), ExtOp0, ExtOp1);

  I.replaceAllUsesWith(NewICmp);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformI16OpToI32(SelectInst &I) const {
  assert(isI16Ty(I.getType()) && "I must be 16 bits");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp1 = nullptr;
  Value *ExtOp2 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateSExt(I.getOperand(2), I32Ty);
  } else {
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateZExt(I.getOperand(2), I32Ty);
  }
  ExtRes = Builder.CreateSelect(I.getOperand(0), ExtOp1, ExtOp2);
  TruncRes = Builder.CreateTrunc(ExtRes, getI16Ty(Builder, ExtRes->getType()));

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool AMDGPUCodeGenPrepare::promoteUniformI16BitreverseIntrinsicToI32(
    IntrinsicInst &I) const {
  assert(I.getIntrinsicID() == Intrinsic::bitreverse && "I must be bitreverse");
  assert(isI16Ty(I.getType()) && "I must be 16 bits");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Function *I32 =
      Intrinsic::getDeclaration(Mod, Intrinsic::bitreverse, { I32Ty });;
  Value *ExtOp = Builder.CreateZExt(I.getOperand(0), I32Ty);
  Value *ExtRes = Builder.CreateCall(I32, { ExtOp });
  Value *LShrOp = Builder.CreateLShr(ExtRes, 16);
  Value *TruncRes =
      Builder.CreateTrunc(LShrOp, getI16Ty(Builder, ExtRes->getType()));

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

static bool shouldKeepFDivF32(Value *Num, bool UnsafeDiv) {
  const ConstantFP *CNum = dyn_cast<ConstantFP>(Num);
  if (!CNum)
    return false;

  // Reciprocal f32 is handled separately without denormals.
  return UnsafeDiv || CNum->isExactlyValue(+1.0);
}

// Insert an intrinsic for fast fdiv for safe math situations where we can
// reduce precision. Leave fdiv for situations where the generic node is
// expected to be optimized.
bool AMDGPUCodeGenPrepare::visitFDiv(BinaryOperator &FDiv) {
  Type *Ty = FDiv.getType();

  // TODO: Handle half
  if (!Ty->getScalarType()->isFloatTy())
    return false;

  MDNode *FPMath = FDiv.getMetadata(LLVMContext::MD_fpmath);
  if (!FPMath)
    return false;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&FDiv);
  float ULP = FPOp->getFPAccuracy();
  if (ULP < 2.5f)
    return false;

  FastMathFlags FMF = FPOp->getFastMathFlags();
  bool UnsafeDiv = HasUnsafeFPMath || FMF.unsafeAlgebra() ||
                                      FMF.allowReciprocal();
  if (ST->hasFP32Denormals() && !UnsafeDiv)
    return false;

  IRBuilder<> Builder(FDiv.getParent(), std::next(FDiv.getIterator()), FPMath);
  Builder.setFastMathFlags(FMF);
  Builder.SetCurrentDebugLocation(FDiv.getDebugLoc());

  const AMDGPUIntrinsicInfo *II = TM->getIntrinsicInfo();
  Function *Decl
    = II->getDeclaration(Mod, AMDGPUIntrinsic::amdgcn_fdiv_fast, {});

  Value *Num = FDiv.getOperand(0);
  Value *Den = FDiv.getOperand(1);

  Value *NewFDiv = nullptr;

  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    NewFDiv = UndefValue::get(VT);

    // FIXME: Doesn't do the right thing for cases where the vector is partially
    // constant. This works when the scalarizer pass is run first.
    for (unsigned I = 0, E = VT->getNumElements(); I != E; ++I) {
      Value *NumEltI = Builder.CreateExtractElement(Num, I);
      Value *DenEltI = Builder.CreateExtractElement(Den, I);
      Value *NewElt;

      if (shouldKeepFDivF32(NumEltI, UnsafeDiv)) {
        NewElt = Builder.CreateFDiv(NumEltI, DenEltI);
      } else {
        NewElt = Builder.CreateCall(Decl, { NumEltI, DenEltI });
      }

      NewFDiv = Builder.CreateInsertElement(NewFDiv, NewElt, I);
    }
  } else {
    if (!shouldKeepFDivF32(Num, UnsafeDiv))
      NewFDiv = Builder.CreateCall(Decl, { Num, Den });
  }

  if (NewFDiv) {
    FDiv.replaceAllUsesWith(NewFDiv);
    NewFDiv->takeName(&FDiv);
    FDiv.eraseFromParent();
  }

  return true;
}

static bool hasUnsafeFPMath(const Function &F) {
  Attribute Attr = F.getFnAttribute("unsafe-fp-math");
  return Attr.getValueAsString() == "true";
}

bool AMDGPUCodeGenPrepare::visitBinaryOperator(BinaryOperator &I) {
  bool Changed = false;

  // TODO: Should we promote smaller types that will be legalized to i16?
  if (ST->has16BitInsts() && isI16Ty(I.getType()) && DA->isUniform(&I))
    Changed |= promoteUniformI16OpToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitICmpInst(ICmpInst &I) {
  bool Changed = false;

  // TODO: Should we promote smaller types that will be legalized to i16?
  if (ST->has16BitInsts() && isI16Ty(I.getOperand(0)->getType()) &&
          isI16Ty(I.getOperand(1)->getType()) && DA->isUniform(&I))
    Changed |= promoteUniformI16OpToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitSelectInst(SelectInst &I) {
  bool Changed = false;

  // TODO: Should we promote smaller types that will be legalized to i16?
  if (ST->has16BitInsts() && isI16Ty(I.getType()) && DA->isUniform(&I))
    Changed |= promoteUniformI16OpToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::bitreverse:
    return visitBitreverseIntrinsicInst(I);
  default:
    return false;
  }
}

bool AMDGPUCodeGenPrepare::visitBitreverseIntrinsicInst(IntrinsicInst &I) {
  bool Changed = false;

  // TODO: Should we promote smaller types that will be legalized to i16?
  if (ST->has16BitInsts() && isI16Ty(I.getType()) && DA->isUniform(&I))
    Changed |= promoteUniformI16BitreverseIntrinsicToI32(I);

  return Changed;
}

bool AMDGPUCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  return false;
}

bool AMDGPUCodeGenPrepare::runOnFunction(Function &F) {
  if (!TM || skipFunction(F))
    return false;

  ST = &TM->getSubtarget<SISubtarget>(F);
  DA = &getAnalysis<DivergenceAnalysis>();
  HasUnsafeFPMath = hasUnsafeFPMath(F);

  bool MadeChange = false;

  for (BasicBlock &BB : F) {
    BasicBlock::iterator Next;
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; I = Next) {
      Next = std::next(I);
      MadeChange |= visit(*I);
    }
  }

  return MadeChange;
}

INITIALIZE_TM_PASS_BEGIN(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(DivergenceAnalysis)
INITIALIZE_TM_PASS_END(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                       "AMDGPU IR optimizations", false, false)

char AMDGPUCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPUCodeGenPreparePass(const GCNTargetMachine *TM) {
  return new AMDGPUCodeGenPrepare(TM);
}
