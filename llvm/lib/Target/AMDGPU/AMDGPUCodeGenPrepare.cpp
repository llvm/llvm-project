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

  bool visitInstruction(Instruction &I) {
    return false;
  }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  const char *getPassName() const override {
    return "AMDGPU IR optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DivergenceAnalysis>();
    AU.setPreservesAll();
 }
};

} // End anonymous namespace

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
