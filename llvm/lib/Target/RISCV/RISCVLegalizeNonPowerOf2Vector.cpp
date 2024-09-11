#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/VectorBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

using namespace llvm;

#define DEBUG_TYPE "riscv-legalize-non-power-of-2-vector"
#define PASS_NAME "Legalize non-power-of-2 vector type"

namespace {
class RISCVLegalizeNonPowerOf2Vector : public FunctionPass {
  const RISCVSubtarget *ST;
  unsigned MinVScale;

public:
  static char ID;
  RISCVLegalizeNonPowerOf2Vector() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }

  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return PASS_NAME; }

private:
  FixedVectorType *extracUsedFixedVectorType(const Instruction &I) const;

  bool isTargetType(FixedVectorType *VecTy) const;

  ScalableVectorType *
  getContainerForFixedLengthVector(FixedVectorType *FixedVecTy);
};
} // namespace

FixedVectorType *RISCVLegalizeNonPowerOf2Vector::extracUsedFixedVectorType(
    const Instruction &I) const {
  if (isa<FixedVectorType>(I.getType())) {
    return cast<FixedVectorType>(I.getType());
  } else if (isa<StoreInst>(I) &&
             isa<FixedVectorType>(
                 cast<StoreInst>(&I)->getValueOperand()->getType())) {
    return cast<FixedVectorType>(
        cast<StoreInst>(&I)->getValueOperand()->getType());
  }
  return nullptr;
}

ScalableVectorType *
RISCVLegalizeNonPowerOf2Vector::getContainerForFixedLengthVector(
    FixedVectorType *FixedVecTy) {
  // TODO: Consider vscale_range to pick a better/smaller type.
  //
  uint64_t 	 NumElts =
      std::max<uint64_t>((NextPowerOf2 (FixedVecTy->getNumElements()) / MinVScale), 1);

  Type *ElementType = FixedVecTy->getElementType();

  if (ElementType->isIntegerTy(1))
      NumElts = std::max(NumElts, 8UL);

  return ScalableVectorType::get(ElementType, NumElts);
}

bool RISCVLegalizeNonPowerOf2Vector::isTargetType(
    FixedVectorType *VecTy) const {
  if (isPowerOf2_32(VecTy->getNumElements()))
    return false;

  Type *EltTy = VecTy->getElementType();

  if (EltTy->isIntegerTy(1))
    return false;

  if (EltTy->isIntegerTy(64))
    return ST->hasVInstructionsI64();
  else if (EltTy->isFloatTy())
    return ST->hasVInstructionsF32();
  else if (EltTy->isDoubleTy())
    return ST->hasVInstructionsF64();
  else if (EltTy->isHalfTy())
    return ST->hasVInstructionsF16Minimal();
  else if (EltTy->isBFloatTy())
    return ST->hasVInstructionsBF16Minimal();

  return (EltTy->isIntegerTy(1) || EltTy->isIntegerTy(8) ||
          EltTy->isIntegerTy(16) || EltTy->isIntegerTy(32));
}

bool RISCVLegalizeNonPowerOf2Vector::runOnFunction(Function &F) {

  if (skipFunction(F))
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);

  if (!ST->hasVInstructions())
    return false;

  auto Attr = F.getFnAttribute(Attribute::VScaleRange);
  if (Attr.isValid()) {
      MinVScale = Attr.getVScaleRangeMin ();
  } else {
    unsigned MinVLen = ST->getRealMinVLen();
    if (MinVLen < RISCV::RVVBitsPerBlock)
      return false;
    MinVScale = MinVLen / RISCV::RVVBitsPerBlock;
    AttrBuilder AB(F.getContext());
    AB.addVScaleRangeAttr(MinVScale,
                         std::optional<unsigned>());

    F.addFnAttr (AB.getAttribute(Attribute::VScaleRange));
  }

  bool Modified = false;
  std::vector<Instruction *> ToBeRemoved;
  for (auto &BB : F) {
    for (auto &I : make_range(BB.rbegin(), BB.rend())) {
      if (auto VecTy = extracUsedFixedVectorType(I)) {
        if (!isTargetType(VecTy)) {
          continue;
        }

        Value *I64Zero = ConstantInt::get(Type::getInt64Ty(F.getContext()), 0);

        // Replace fixed length vector with scalable vector
        IRBuilder<> Builder(&I);
        VectorBuilder VecBuilder(Builder);
        VecBuilder.setStaticVL(VecTy->getNumElements());
        VectorType *NewVecTy = getContainerForFixedLengthVector(VecTy);
        VecBuilder.setMask(Builder.CreateVectorSplat(
            NewVecTy->getElementCount(), Builder.getTrue()));

        if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
          Value *Op1 = BinOp->getOperand(0);
          Value *Op2 = BinOp->getOperand(1);
          Value *NewOp1 = Builder.CreateInsertVector(
              NewVecTy, PoisonValue::get(NewVecTy), Op1, I64Zero);
          Value *NewOp2 = Builder.CreateInsertVector(
              NewVecTy, PoisonValue::get(NewVecTy), Op2, I64Zero);
          Value *NewBinOp = VecBuilder.createVectorInstruction(
              BinOp->getOpcode(), NewVecTy, {NewOp1, NewOp2});
          Value *FinalResult =
              Builder.CreateExtractVector(VecTy, NewBinOp, I64Zero);
          BinOp->replaceAllUsesWith(FinalResult);
          ToBeRemoved.push_back(BinOp);
          Modified = true;
        } else if (auto *StoreOp = dyn_cast<StoreInst>(&I)) {
          Value *Val = StoreOp->getOperand(0);
          Value *Addr = StoreOp->getOperand(1);
          Value *NewVal = Builder.CreateInsertVector(
              NewVecTy, PoisonValue::get(NewVecTy), Val, I64Zero);
          Value *NewStoreOp = VecBuilder.createVectorInstruction(
              StoreOp->getOpcode(), NewVecTy, {NewVal, Addr});
          StoreOp->replaceAllUsesWith(NewStoreOp);
          ToBeRemoved.push_back(StoreOp);
        } else if (auto *LoadOp = dyn_cast<LoadInst>(&I)) {
          Value *Addr = LoadOp->getOperand(0);
          Value *NewLoadOp = VecBuilder.createVectorInstruction(
              LoadOp->getOpcode(), NewVecTy, {Addr});
          Value *FinalResult =
              Builder.CreateExtractVector(VecTy, NewLoadOp, I64Zero);
          LoadOp->replaceAllUsesWith(FinalResult);
          ToBeRemoved.push_back(LoadOp);
        }
      }
    }
  }
  for_each(ToBeRemoved.begin(), ToBeRemoved.end(),
           [](Instruction *I) { I->eraseFromParent(); });
  return Modified;
}

char RISCVLegalizeNonPowerOf2Vector::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVLegalizeNonPowerOf2Vector, DEBUG_TYPE, PASS_NAME,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVLegalizeNonPowerOf2Vector, DEBUG_TYPE, PASS_NAME,
                    false, false)

FunctionPass *llvm::createRISCVLegalizeNonPowerOf2Vector() {
  return new RISCVLegalizeNonPowerOf2Vector();
}
