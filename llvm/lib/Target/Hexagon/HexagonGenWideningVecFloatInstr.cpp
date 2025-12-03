//===------------------- HexagonGenWideningVecFloatInstr.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace widening vector float operations with hexagon intrinsics.
//
//===----------------------------------------------------------------------===//
//
// Brief overview of working of GenWideningVecFloatInstr pass.
// This version of pass is replica of already existing pass(which will replace
// widen vector integer operations with it's respective intrinsics). In this
// pass we will generate hexagon intrinsics for widen vector float instructions.
//
// Example1(64 vector-width widening):
// %wide.load = load <64 x half>, <64 x half>* %0, align 2
// %wide.load53 = load <64 x half>, <64 x half>* %2, align 2
// %1 = fpext <64 x half> %wide.load to <64 x float>
// %3 = fpext <64 x half> %wide.load53 to <64 x float>
// %4 = fmul <64 x float> %1, %3
//
// If we run this pass on the above example, it will first find fmul
// instruction, and then it will check whether the operands of fmul instruction
// (%1 and %3) belongs to either of these categories [%1 ->fpext, %3 ->fpext]
// or [%1 ->fpext, %3 ->constant_vector] or [%1 ->constant_vector, %3 ->fpext].
// If it sees such pattern, then this pass will replace such pattern with
// appropriate hexagon intrinsics.
//
// After replacement:
// %wide.load = load <64 x half>, <64 x half>* %0, align 2
// %wide.load53 = load <64 x half>, <64 x half>* %2, align 2
// %3 = bitcast <64 x half> %wide.load to <32 x i32>
// %4 = bitcast <64 x half> %wide.load53 to <32 x i32>
// %5 = call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(%3, %4)
// %6 = shufflevector <64 x i32> %5, <64 x i32> poison, <64 x i32> ShuffMask1
// %7 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %6)
// %8 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %6)
// %9 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %7)
// %10 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %8)
// %11 = bitcast <32 x i32> %9 to <32 x float>
// %12 = bitcast <32 x i32> %10 to <32 x float>
// %13 = shufflevector <32 x float> %12, <32 x float> %11, <64 x i32> ShuffMask2
//
//
//
// Example2(128 vector-width widening):
// %0 = bitcast half* %a to <128 x half>*
// %wide.load = load <128 x half>, <128 x half>* %0, align 2
// %1 = fpext <128 x half> %wide.load to <128 x float>
// %2 = bitcast half* %b to <128 x half>*
// %wide.load2 = load <128 x half>, <128 x half>* %2, align 2
// %3 = fpext <128 x half> %wide.load2 to <128 x float>
// %4 = fmul <128 x float> %1, %3
//
// After replacement:
// %0 = bitcast half* %a to <128 x half>*
// %wide.load = load <128 x half>, <128 x half>* %0, align 2
// %1 = bitcast half* %b to <128 x half>*
// %wide.load2 = load <128 x half>, <128 x half>* %1, align 2
// %2 = bitcast <128 x half> %wide.load to <64 x i32>
// %3 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %2)
// %4 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %2)
// %5 = bitcast <128 x half> %wide.load2 to <64 x i32>
// %6 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %5)
// %7 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %5)
// %8 = call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(%3, %6)
// %9 = shufflevector <64 x i32> %8, <64 x i32> poison, <64 x i32> Mask1
// %10 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %9)
// %11 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %9)
// %12 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %10)
// %13 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %11)
// %14 = bitcast <32 x i32> %12 to <32 x float>
// %15 = bitcast <32 x i32> %13 to <32 x float>
// %16 = shufflevector <32 x float> %15, <32 x float> %14, <64 x i32> Mask2
// %17 = call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(%4, %7)
// %18 = shufflevector <64 x i32> %17, <64 x i32> poison, <64 x i32> Mask1
// %19 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %18)
// %20 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %18)
// %21 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %19)
// %22 = call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %20)
// %23 = bitcast <32 x i32> %21 to <32 x float>
// %24 = bitcast <32 x i32> %22 to <32 x float>
// %25 = shufflevector <32 x float> %24, <32 x float> %23, <64 x i32> Mask2
// %26 = shufflevector <64 x float> %25, <64 x float> %16, <128 x i32> Mask3
//
//
//===----------------------------------------------------------------------===//
#include "HexagonTargetMachine.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <utility>

using namespace llvm;

namespace llvm {
void initializeHexagonGenWideningVecFloatInstrPass(PassRegistry &);
FunctionPass *
createHexagonGenWideningVecFloatInstr(const HexagonTargetMachine &);
} // end namespace llvm

namespace {

class HexagonGenWideningVecFloatInstr : public FunctionPass {
public:
  static char ID;

  HexagonGenWideningVecFloatInstr() : FunctionPass(ID) {
    initializeHexagonGenWideningVecFloatInstrPass(
        *PassRegistry::getPassRegistry());
  }

  HexagonGenWideningVecFloatInstr(const HexagonTargetMachine *TM)
      : FunctionPass(ID), TM(TM) {
    initializeHexagonGenWideningVecFloatInstrPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Hexagon generate widening vector float instructions";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
  }

private:
  Module *M = nullptr;
  const HexagonTargetMachine *TM = nullptr;
  const HexagonSubtarget *HST = nullptr;
  unsigned HwVLen;
  unsigned NumHalfEltsInFullVec;

  struct OPInfo {
    Value *OP;
    Value *ExtInOP;
    unsigned ExtInSize;
  };

  bool visitBlock(BasicBlock *B);
  bool processInstruction(Instruction *Inst);
  bool replaceWithIntrinsic(Instruction *Inst, OPInfo &OP1Info,
                            OPInfo &OP2Info);

  bool getOperandInfo(Value *V, OPInfo &OPI);
  bool isExtendedConstant(Constant *C);
  unsigned getElementSizeInBits(Value *V);
  Type *getElementTy(unsigned size, IRBuilder<> &IRB);

  Value *adjustExtensionForOp(OPInfo &OPI, IRBuilder<> &IRB,
                              unsigned NewEltsize, unsigned NumElts);

  std::pair<Value *, Value *> opSplit(Value *OP, Instruction *Inst);

  Value *createIntrinsic(Intrinsic::ID IntId, Instruction *Inst, Value *NewOP1,
                         Value *NewOP2, FixedVectorType *ResType,
                         unsigned NumElts, bool BitCastOp);
};

} // end anonymous namespace

char HexagonGenWideningVecFloatInstr::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonGenWideningVecFloatInstr, "widening-vec-float",
                      "Hexagon generate "
                      "widening vector float instructions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(HexagonGenWideningVecFloatInstr, "widening-vec-float",
                    "Hexagon generate "
                    "widening vector float instructions",
                    false, false)

bool HexagonGenWideningVecFloatInstr::isExtendedConstant(Constant *C) {
  if (Value *SplatV = C->getSplatValue()) {
    if (auto *CFP = dyn_cast<ConstantFP>(SplatV)) {
      bool Ignored;
      APFloat APF = CFP->getValueAPF();
      APFloat::opStatus sts = APF.convert(
          APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
      if (sts == APFloat::opStatus::opOK || sts == APFloat::opStatus::opInexact)
        return true;
    }
    return false;
  }
  unsigned NumElts = cast<FixedVectorType>(C->getType())->getNumElements();
  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    if (auto *CFP = dyn_cast<ConstantFP>(C->getAggregateElement(i))) {
      bool Ignored;
      APFloat APF = CFP->getValueAPF();
      APFloat::opStatus sts = APF.convert(
          APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
      if (sts != APFloat::opStatus::opOK && sts != APFloat::opStatus::opInexact)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

unsigned HexagonGenWideningVecFloatInstr::getElementSizeInBits(Value *V) {
  Type *ValTy = V->getType();
  Type *EltTy = ValTy;
  if (dyn_cast<Constant>(V)) {
    unsigned EltSize =
        cast<VectorType>(EltTy)->getElementType()->getPrimitiveSizeInBits();
    unsigned ReducedSize = EltSize / 2;

    return ReducedSize;
  }

  if (ValTy->isVectorTy())
    EltTy = cast<VectorType>(ValTy)->getElementType();
  return EltTy->getPrimitiveSizeInBits();
}

bool HexagonGenWideningVecFloatInstr::getOperandInfo(Value *V, OPInfo &OPI) {
  using namespace PatternMatch;
  OPI.OP = V;
  Value *ExtV = nullptr;
  Constant *C = nullptr;

  if (match(V, (m_FPExt(m_Value(ExtV)))) ||
      match(V,
            m_Shuffle(m_InsertElt(m_Poison(), m_FPExt(m_Value(ExtV)), m_Zero()),
                      m_Poison(), m_ZeroMask()))) {

    if (auto *ExtVType = dyn_cast<VectorType>(ExtV->getType())) {
      // Matches the first branch.
      if (ExtVType->getElementType()->isBFloatTy())
        // do not confuse bf16 with ieee-fp16.
        return false;
    } else {
      // Matches the second branch (insert element branch)
      if (ExtV->getType()->isBFloatTy())
        return false;
    }

    OPI.ExtInOP = ExtV;
    OPI.ExtInSize = getElementSizeInBits(OPI.ExtInOP);
    return true;
  }

  if (match(V, m_Constant(C))) {
    if (!isExtendedConstant(C))
      return false;
    OPI.ExtInOP = C;
    OPI.ExtInSize = getElementSizeInBits(OPI.ExtInOP);
    return true;
  }

  return false;
}

Type *HexagonGenWideningVecFloatInstr::getElementTy(unsigned size,
                                                    IRBuilder<> &IRB) {
  switch (size) {
  case 16:
    return IRB.getHalfTy();
  case 32:
    return IRB.getFloatTy();
  default:
    llvm_unreachable("Unhandled Element size");
  }
}

Value *HexagonGenWideningVecFloatInstr::adjustExtensionForOp(
    OPInfo &OPI, IRBuilder<> &IRB, unsigned NewExtSize, unsigned NumElts) {
  Value *V = OPI.ExtInOP;
  unsigned EltSize = getElementSizeInBits(OPI.ExtInOP);
  assert(NewExtSize >= EltSize);
  Type *EltType = getElementTy(NewExtSize, IRB);
  auto *NewOpTy = FixedVectorType::get(EltType, NumElts);

  if (auto *C = dyn_cast<Constant>(V))
    return IRB.CreateFPTrunc(C, NewOpTy);

  if (V->getType()->isVectorTy())
    if (NewExtSize == EltSize)
      return V;

  return nullptr;
}

std::pair<Value *, Value *>
HexagonGenWideningVecFloatInstr::opSplit(Value *OP, Instruction *Inst) {
  Type *InstTy = Inst->getType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  IRBuilder<> IRB(Inst);
  Intrinsic::ID IntHi = Intrinsic::hexagon_V6_hi_128B;
  Intrinsic::ID IntLo = Intrinsic::hexagon_V6_lo_128B;
  Function *ExtFHi = Intrinsic::getOrInsertDeclaration(M, IntHi);
  Function *ExtFLo = Intrinsic::getOrInsertDeclaration(M, IntLo);
  if (NumElts == 128) {
    auto *InType = FixedVectorType::get(IRB.getInt32Ty(), 64);
    OP = IRB.CreateBitCast(OP, InType);
  }
  Value *OP1Hi = IRB.CreateCall(ExtFHi, {OP});
  Value *OP1Lo = IRB.CreateCall(ExtFLo, {OP});
  return std::pair<Value *, Value *>(OP1Hi, OP1Lo);
}

Value *HexagonGenWideningVecFloatInstr::createIntrinsic(
    Intrinsic::ID IntId, Instruction *Inst, Value *NewOP1, Value *NewOP2,
    FixedVectorType *ResType, unsigned NumElts, bool BitCastOp) {

  IRBuilder<> IRB(Inst);
  Function *ExtF = Intrinsic::getOrInsertDeclaration(M, IntId);
  Function *ConvF = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::hexagon_V6_vconv_sf_qf32_128B);
  auto *InType = FixedVectorType::get(IRB.getInt32Ty(), 32);
  auto *RType = FixedVectorType::get(IRB.getFloatTy(), 32);

  // Make sure inputs to vmpy instrinsic are full vectors
  if (NumElts == NumHalfEltsInFullVec / 2) {
    SmallVector<Constant *, 16> ConcatMask1;
    for (unsigned i = 0; i < NumHalfEltsInFullVec; ++i)
      ConcatMask1.push_back(IRB.getInt32(i));
    NewOP1 =
        IRB.CreateShuffleVector(NewOP1, PoisonValue::get(NewOP1->getType()),
                                ConstantVector::get(ConcatMask1));
    NewOP2 =
        IRB.CreateShuffleVector(NewOP2, PoisonValue::get(NewOP2->getType()),
                                ConstantVector::get(ConcatMask1));
  }

  if (BitCastOp) {
    NewOP1 = IRB.CreateBitCast(NewOP1, InType);
    NewOP2 = IRB.CreateBitCast(NewOP2, InType);
  }

  Value *NewIn = IRB.CreateCall(ExtF, {NewOP1, NewOP2});
  // Interleave the output elements to ensure correct order in Hi and Lo vectors
  // Shuffled Mask: [0, 32, 1, 33, ..., 31, 63]
  // Hi: [0, 1, ..., 31] and Lo: [32, 33, ..., 63]
  SmallVector<Constant *, 16> Mask;
  unsigned HalfVecPoint = NumHalfEltsInFullVec / 2;
  for (unsigned i = 0; i < HalfVecPoint; ++i) {
    Mask.push_back(IRB.getInt32(i));
    Mask.push_back(IRB.getInt32(HalfVecPoint + i));
  }
  NewIn = IRB.CreateShuffleVector(NewIn, PoisonValue::get(NewIn->getType()),
                                  ConstantVector::get(Mask));

  std::pair<Value *, Value *> SplitOP = opSplit(NewIn, Inst);
  Value *ConvHi = IRB.CreateCall(ConvF, {SplitOP.first});
  ConvHi = IRB.CreateBitCast(ConvHi, RType);

  if (ResType->getNumElements() == NumHalfEltsInFullVec / 2) {
    return ConvHi;
  }

  Value *ConvLo = IRB.CreateCall(ConvF, {SplitOP.second});
  ConvLo = IRB.CreateBitCast(ConvLo, RType);

  SmallVector<Constant *, 16> ShuffleMask;
  for (unsigned i = 0; i < NumElts; ++i)
    ShuffleMask.push_back(IRB.getInt32(i));
  // Concat Hi and Lo.
  NewIn =
      IRB.CreateShuffleVector(ConvLo, ConvHi, ConstantVector::get(ShuffleMask));
  return NewIn;
}

bool HexagonGenWideningVecFloatInstr::replaceWithIntrinsic(Instruction *Inst,
                                                           OPInfo &OP1Info,
                                                           OPInfo &OP2Info) {
  Type *InstTy = Inst->getType();
  Type *EltTy = cast<FixedVectorType>(InstTy)->getElementType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  [[maybe_unused]] unsigned InstEltSize = EltTy->getPrimitiveSizeInBits();

  unsigned MaxEltSize = OP1Info.ExtInSize;
  unsigned NewOpEltSize = MaxEltSize;
  unsigned NewResEltSize = 2 * MaxEltSize;

  unsigned ResVLen = NewResEltSize * NumElts;
  if (NewOpEltSize > 16 || ((ResVLen > HwVLen) && (ResVLen % HwVLen) != 0))
    return false;

  Intrinsic::ID IntId = Intrinsic::hexagon_V6_vmpy_qf32_hf_128B;
  IRBuilder<> IRB(Inst);
  Value *NewOP1 = adjustExtensionForOp(OP1Info, IRB, NewOpEltSize, NumElts);
  Value *NewOP2 = adjustExtensionForOp(OP2Info, IRB, NewOpEltSize, NumElts);

  if (NewOP1 == nullptr || NewOP2 == nullptr)
    return false;

  if (ResVLen > 2 * HwVLen) {
    // The code written in this if block generates the widening code when
    // vector-width is 128:
    //
    // Step 1: Bitcast <128 x half> type to <64 x i32>
    // %wide.load = load <128 x half>, <128 x half>* %0 is bitcasted to,
    // bitcast <128 x half> %wide.load to <64 x i32>
    //
    // Step 2: Generate Hi and Lo vectors
    // call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %4)
    // call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %4)
    //
    // Perform above 2 steps for both the operands of fmul instruction
    //
    // Step 3: Generate vmpy_qf32_hf multiply instruction to multiply two Hi
    // vectors from both operands.
    // call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(%5, %8)
    //
    // Step 4: Convert the resultant 'qf32' output to 'sf' format
    // %11 = shufflevector <64 x i32> %10, <64 x i32> poison, <64 x i32> Mask1
    // %12 = call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %11)
    // %13 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %11)
    // call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %12)
    // call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %13)
    //
    // Repeat steps 3 and 4 for mutiplication and conversion of Lo vectors.
    // Finally merge the output values in correct sequence using shuffle
    // vectors.

    assert(ResVLen == 4 * HwVLen);
    // Split the operands
    unsigned HalfElts = NumElts / 2;
    std::pair<Value *, Value *> SplitOP1 = opSplit(NewOP1, Inst);
    std::pair<Value *, Value *> SplitOP2 = opSplit(NewOP2, Inst);
    auto *castResType = FixedVectorType::get(IRB.getInt32Ty(), HalfElts);
    Value *NewInHi =
        createIntrinsic(IntId, Inst, SplitOP1.first, SplitOP2.first,
                        castResType, HalfElts, false);
    Value *NewInLo =
        createIntrinsic(IntId, Inst, SplitOP1.second, SplitOP2.second,
                        castResType, HalfElts, false);
    assert(InstEltSize == NewResEltSize);
    SmallVector<Constant *, 8> ShuffleMask;
    for (unsigned i = 0; i < NumElts; ++i)
      ShuffleMask.push_back(IRB.getInt32(i));
    // Concat Hi and Lo.
    Value *NewIn = IRB.CreateShuffleVector(NewInLo, NewInHi,
                                           ConstantVector::get(ShuffleMask));

    Inst->replaceAllUsesWith(NewIn);
    return true;
  }

  auto *ResType =
      FixedVectorType::get(getElementTy(NewResEltSize, IRB), NumElts);

  // The following widening code can only be generated in cases where
  // input vectors are 64xhalf/32xhalf and the results are 64xfloat/32xfloat
  // respectively.
  if (!(NumElts == NumHalfEltsInFullVec &&
        ResType->getNumElements() == NumHalfEltsInFullVec) &&
      !(NumElts == NumHalfEltsInFullVec / 2 &&
        ResType->getNumElements() == NumHalfEltsInFullVec / 2))
    return false;
  Value *NewIn =
      createIntrinsic(IntId, Inst, NewOP1, NewOP2, ResType, NumElts, true);

  Inst->replaceAllUsesWith(NewIn);
  return true;
}

// Process instruction and replace them with widening vector
// intrinsics if possible.
bool HexagonGenWideningVecFloatInstr::processInstruction(Instruction *Inst) {
  Type *InstTy = Inst->getType();
  if (!InstTy->isVectorTy() ||
      cast<FixedVectorType>(InstTy)->getNumElements() > 128)
    return false;
  unsigned InstLen = InstTy->getPrimitiveSizeInBits();
  if (!HST->isTypeForHVX(cast<VectorType>(InstTy)) && InstLen != 4 * HwVLen)
    return false;
  if (InstLen < HwVLen)
    return false;

  using namespace PatternMatch;

  Value *OP1 = nullptr, *OP2 = nullptr;
  OPInfo OP1Info, OP2Info;

  // Handle the case when Inst = fpext(fmul<64xhalf>(op1, op2)). The Inst can
  // be replaced with widening multiply.
  if (match(Inst, (m_FPExt((m_FMul(m_Value(OP1), m_Value(OP2))))))) {
    OP1Info.ExtInOP = OP1;
    OP1Info.ExtInSize = getElementSizeInBits(OP1);
    OP2Info.ExtInOP = OP2;
    OP2Info.ExtInSize = getElementSizeInBits(OP2);

    if (auto *Op1Vtype = dyn_cast<VectorType>(OP1->getType())) {
      if (!Op1Vtype->getElementType()->isHalfTy()) {
        return false;
      }
    } else {
      return false;
    }

    if (OP1Info.ExtInSize == OP2Info.ExtInSize && OP1Info.ExtInSize == 16 &&
        getElementSizeInBits(Inst) == 32) {
      return replaceWithIntrinsic(Inst, OP1Info, OP2Info);
    }
  }

  if (!match(Inst, (m_FMul(m_Value(OP1), m_Value(OP2)))))
    return false;

  if (!getOperandInfo(OP1, OP1Info) || !getOperandInfo(OP2, OP2Info))
    return false;

  if (!OP1Info.ExtInOP || !OP2Info.ExtInOP)
    return false;

  if (OP1Info.ExtInSize == OP2Info.ExtInSize && OP1Info.ExtInSize == 16) {
    return replaceWithIntrinsic(Inst, OP1Info, OP2Info);
  }

  return false;
}

bool HexagonGenWideningVecFloatInstr::visitBlock(BasicBlock *B) {
  bool Changed = false;
  for (auto &I : *B)
    Changed |= processInstruction(&I);
  return Changed;
}

bool HexagonGenWideningVecFloatInstr::runOnFunction(Function &F) {
  M = F.getParent();
  HST = TM->getSubtargetImpl(F);

  // Return if useHVX128BOps is not set. It can be enabled for 64B mode
  // but wil require some changes. For example, bitcast for intrinsics
  // assumes 128B mode.
  if (skipFunction(F) || !HST->useHVX128BOps())
    return false;

  unsigned VecLength = HST->getVectorLength(); // Vector Length in Bytes
  HwVLen = HST->getVectorLength() * 8;         // Vector Length in bits
  NumHalfEltsInFullVec =
      VecLength /
      2; // Number of half (2B) elements that fit into a full HVX vector
  bool Changed = false;
  for (auto &B : F)
    Changed |= visitBlock(&B);

  return Changed;
}

FunctionPass *
llvm::createHexagonGenWideningVecFloatInstr(const HexagonTargetMachine &TM) {
  return new HexagonGenWideningVecFloatInstr(&TM);
}
