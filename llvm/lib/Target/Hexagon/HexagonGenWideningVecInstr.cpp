//===--------------------- HexagonGenWideningVecInstr.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace widening vector operations with hexagon intrinsics.
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
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <utility>

using namespace llvm;

// A command line argument to enable the generation of widening instructions
// for short-vectors.
static cl::opt<bool> WidenShortVector(
    "hexagon-widen-short-vector",
    cl::desc("Generate widening instructions for short vectors."), cl::Hidden);

namespace llvm {
void initializeHexagonGenWideningVecInstrPass(PassRegistry &);
FunctionPass *createHexagonGenWideningVecInstr(const HexagonTargetMachine &);
} // end namespace llvm

namespace {

class HexagonGenWideningVecInstr : public FunctionPass {
public:
  static char ID;

  HexagonGenWideningVecInstr() : FunctionPass(ID) {
    initializeHexagonGenWideningVecInstrPass(*PassRegistry::getPassRegistry());
  }

  HexagonGenWideningVecInstr(const HexagonTargetMachine *TM)
      : FunctionPass(ID), TM(TM) {
    initializeHexagonGenWideningVecInstrPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Hexagon generate widening vector instructions";
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
  enum OPKind { OP_None = 0, OP_Add, OP_Sub, OP_Mul, OP_Shl };

  struct OPInfo {
    Value *OP = nullptr;
    Value *ExtInOP = nullptr;
    bool IsZExt = false;
    unsigned ExtInSize = 0;
    bool IsScalar = false;
  };

  bool visitBlock(BasicBlock *B);
  bool processInstruction(Instruction *Inst);
  bool replaceWithIntrinsic(Instruction *Inst, OPKind OPK, OPInfo &OP1Info,
                            OPInfo &OP2Info);
  bool getOperandInfo(Value *V, OPInfo &OPI);
  bool isExtendedConstant(Constant *C, bool IsSigned);
  unsigned getElementSizeInBits(Value *V, bool IsZExt);
  Type *getElementTy(unsigned size, IRBuilder<> &IRB);

  Value *adjustExtensionForOp(OPInfo &OPI, IRBuilder<> &IRB,
                              unsigned NewEltsize, unsigned NumElts);

  Intrinsic::ID getIntrinsic(OPKind OPK, bool IsOP1ZExt, bool IsOP2ZExt,
                             unsigned NewOpEltSize, unsigned NewResEltSize,
                             bool IsConstScalar, int ConstOpNum);

  std::pair<Value *, Value *> opSplit(Value *OP, Instruction *Inst,
                                      Type *NewOpType);

  Value *createIntrinsic(Intrinsic::ID IntId, Instruction *Inst, Value *NewOP1,
                         Value *NewOP2, Type *ResType, unsigned NumElts,
                         bool Interleave);
  bool processInstructionForVMPA(Instruction *Inst);
  bool getVmpaOperandInfo(Value *V, OPInfo &OPI);
  void reorderVmpaOperands(OPInfo *OPI);
  bool replaceWithVmpaIntrinsic(Instruction *Inst, OPInfo *OPI);
  bool genSaturatingInst(Instruction *Inst);
  bool getMinMax(Constant *MinC, Constant *MaxC, std::pair<int, int> &MinMax);
  bool isSaturatingVAsr(Instruction *Inst, Value *S, int MinV, int MaxV,
                        bool &IsResSigned);
  Value *extendShiftByVal(Value *ShiftByVal, IRBuilder<> &IRB);
  Intrinsic::ID getVAsrIntrinsic(bool IsInSigned, bool IsResSigned);
  Value *createVAsrIntrinsic(Instruction *Inst, Value *VecOP, Value *ShiftByVal,
                             bool IsResSigned);
  bool genVAvg(Instruction *Inst);
  bool checkConstantVector(Value *OP, int64_t &SplatVal, bool IsOPZExt);
  void updateMPYConst(Intrinsic::ID IntId, int64_t &SplatVal, bool IsOPZExt,
                      Value *&OP, IRBuilder<> &IRB);
  void packConstant(Intrinsic::ID IntId, int64_t &SplatVal, Value *&OP,
                    IRBuilder<> &IRB);
};

} // end anonymous namespace

char HexagonGenWideningVecInstr::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonGenWideningVecInstr, "widening-vec",
                      "Hexagon generate "
                      "widening vector instructions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(HexagonGenWideningVecInstr, "widening-vec",
                    "Hexagon generate "
                    "widening vector instructions",
                    false, false)

static bool hasNegativeValues(Constant *C) {
  if (Value *SplatV = C->getSplatValue()) {
    auto *CI = dyn_cast<ConstantInt>(SplatV);
    assert(CI);
    return CI->getValue().isNegative();
  }
  unsigned NumElts = cast<FixedVectorType>(C->getType())->getNumElements();
  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    auto *CI = dyn_cast<ConstantInt>(C->getAggregateElement(i));
    assert(CI);
    if (CI->getValue().isNegative())
      return true;
    continue;
  }
  return false;
}

bool HexagonGenWideningVecInstr::getOperandInfo(Value *V, OPInfo &OPI) {
  using namespace PatternMatch;
  OPI.OP = V;
  Value *ExtV = nullptr;
  Constant *C = nullptr;

  bool Match = false;
  if ((Match = (match(V, (m_ZExt(m_Value(ExtV)))) ||
                match(V, m_Shuffle(m_InsertElt(m_Poison(),
                                               m_ZExt(m_Value(ExtV)), m_Zero()),
                                   m_Poison(), m_ZeroMask()))))) {
    OPI.ExtInOP = ExtV;
    OPI.IsZExt = true;
  }

  if (!Match &&
      (Match = (match(V, (m_SExt(m_Value(ExtV)))) ||
                match(V, m_Shuffle(m_InsertElt(m_Poison(),
                                               m_SExt(m_Value(ExtV)), m_Zero()),
                                   m_Poison(), m_ZeroMask()))))) {
    OPI.ExtInOP = ExtV;
    OPI.IsZExt = false;
  }
  if (!Match &&
      (Match =
           (match(V, m_Shuffle(m_InsertElt(m_Poison(), m_Value(ExtV), m_Zero()),
                               m_Poison(), m_ZeroMask()))))) {
    if (match(ExtV, m_And(m_Value(), m_SpecificInt(255)))) {
      OPI.ExtInOP = ExtV;
      OPI.IsZExt = true;
      OPI.ExtInSize = 8;
      return true;
    }
    if (match(ExtV, m_And(m_Value(), m_SpecificInt(65535)))) {
      OPI.ExtInOP = ExtV;
      OPI.IsZExt = true;
      OPI.ExtInSize = 16;
      return true;
    }
    return false;
  }

  if (!Match && (Match = match(V, m_Constant(C)))) {
    if (!isExtendedConstant(C, false) && !isExtendedConstant(C, true))
      return false;
    OPI.ExtInOP = C;
    OPI.IsZExt = !hasNegativeValues(C);
  }

  if (!Match)
    return false;

  // If the operand is extended, find the element size of its input.
  if (OPI.ExtInOP)
    OPI.ExtInSize = getElementSizeInBits(OPI.ExtInOP, OPI.IsZExt);
  return true;
}

bool HexagonGenWideningVecInstr::isExtendedConstant(Constant *C,
                                                    bool IsSigned) {
  Type *CTy = cast<FixedVectorType>(C->getType())->getElementType();
  unsigned EltSize = CTy->getPrimitiveSizeInBits();
  unsigned HalfSize = EltSize / 2;
  if (Value *SplatV = C->getSplatValue()) {
    if (auto *CI = dyn_cast<ConstantInt>(SplatV))
      return IsSigned ? isIntN(HalfSize, CI->getSExtValue())
                      : isUIntN(HalfSize, CI->getZExtValue());
    return false;
  }
  unsigned NumElts = cast<FixedVectorType>(C->getType())->getNumElements();
  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    if (auto *CI = dyn_cast<ConstantInt>(C->getAggregateElement(i))) {
      if ((IsSigned && !isIntN(HalfSize, CI->getSExtValue())) ||
          (!IsSigned && !isUIntN(HalfSize, CI->getZExtValue())))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

unsigned HexagonGenWideningVecInstr::getElementSizeInBits(Value *V,
                                                          bool IsZExt = false) {
  using namespace PatternMatch;
  Type *ValTy = V->getType();
  Type *EltTy = ValTy;
  if (auto *C = dyn_cast<Constant>(V)) {
    unsigned NumElts = cast<FixedVectorType>(EltTy)->getNumElements();
    unsigned EltSize = cast<FixedVectorType>(EltTy)
                           ->getElementType()
                           ->getPrimitiveSizeInBits()
                           .getKnownMinValue();
    unsigned ReducedSize = EltSize / 2;

    while (ReducedSize >= 8) {
      for (unsigned i = 0, e = NumElts; i != e; ++i) {
        if (auto *CI = dyn_cast<ConstantInt>(C->getAggregateElement(i))) {
          if (IsZExt) {
            if (!isUIntN(ReducedSize, CI->getZExtValue()))
              return EltSize;
          } else if (!isIntN(ReducedSize, CI->getSExtValue()))
            return EltSize;
        }
      }
      EltSize = ReducedSize;
      ReducedSize = ReducedSize / 2;
    }
    return EltSize;
  }

  if (ValTy->isVectorTy())
    EltTy = cast<FixedVectorType>(ValTy)->getElementType();
  return EltTy->getPrimitiveSizeInBits();
}

Value *HexagonGenWideningVecInstr::adjustExtensionForOp(OPInfo &OPI,
                                                        IRBuilder<> &IRB,
                                                        unsigned NewExtSize,
                                                        unsigned NumElts) {
  Value *V = OPI.ExtInOP;
  bool IsZExt = OPI.IsZExt;
  unsigned EltSize = getElementSizeInBits(OPI.ExtInOP, OPI.IsZExt);
  Type *EltType = getElementTy(NewExtSize, IRB);
  auto *NewOpTy = FixedVectorType::get(EltType, NumElts);

  if (dyn_cast<Constant>(V))
    return IRB.CreateTrunc(V, NewOpTy);

  if (V->getType()->isVectorTy()) {
    if (NewExtSize == EltSize)
      return V;
    assert(NewExtSize == 16);
    auto *NewOpTy = FixedVectorType::get(IRB.getInt16Ty(), NumElts);
    return (IsZExt) ? IRB.CreateZExt(V, NewOpTy) : IRB.CreateSExt(V, NewOpTy);
  }

  // The operand must correspond to a shuffle vector which is used to construct
  // a vector out of a scalar. Since the scalar value (V) is extended,
  // replace it with a new shuffle vector with the smaller element size.
  [[maybe_unused]] auto *I = dyn_cast<Instruction>(OPI.OP);
  assert(I && I->getOpcode() == Instruction::ShuffleVector);

  if (NewExtSize > EltSize)
    V = (IsZExt) ? IRB.CreateZExt(V, EltType) : IRB.CreateSExt(V, EltType);
  else if (NewExtSize < EltSize)
    V = IRB.CreateTrunc(V, EltType);

  Value *IE =
      IRB.CreateInsertElement(PoisonValue::get(NewOpTy), V, IRB.getInt32(0));

  SmallVector<Constant *, 8> ShuffleMask;
  for (unsigned i = 0; i < NumElts; ++i)
    ShuffleMask.push_back(IRB.getInt32(0));

  return IRB.CreateShuffleVector(IE, PoisonValue::get(NewOpTy),
                                 ConstantVector::get(ShuffleMask));
}

Intrinsic::ID HexagonGenWideningVecInstr::getIntrinsic(
    OPKind OPK, bool IsOP1ZExt, bool IsOP2ZExt, unsigned InEltSize,
    unsigned ResEltSize, bool IsConstScalar, int ConstOpNum) {
  // Since the operands have been extended, the ResEltSize must be 16 or more.
  switch (OPK) {
  case OP_Add:
    // Both operands should be either zero extended or sign extended.
    assert(IsOP1ZExt == IsOP2ZExt);
    if (InEltSize == 8 && ResEltSize == 16) {
      // Operands must be zero extended as we don't have a widening vector
      // 'add' that can take signed exteded values.
      assert(IsOP1ZExt && "Operands must be zero-extended");
      return Intrinsic::hexagon_vadd_uu;
    }
    if (InEltSize == 16 && ResEltSize == 32)
      return (IsOP1ZExt) ? Intrinsic::hexagon_vadd_uu
                         : Intrinsic::hexagon_vadd_ss;

    llvm_unreachable("Incorrect input and output operand sizes");

  case OP_Sub:
    // Both operands should be either zero extended or sign extended.
    assert(IsOP1ZExt == IsOP2ZExt);
    if (InEltSize == 8 && ResEltSize == 16) {
      // Operands must be zero extended as we don't have a widening vector
      // 'sub' that can take signed exteded values.
      assert(IsOP1ZExt && "Operands must be zero-extended");
      return Intrinsic::hexagon_vsub_uu;
    }
    if (InEltSize == 16 && ResEltSize == 32)
      return (IsOP1ZExt) ? Intrinsic::hexagon_vsub_uu
                         : Intrinsic::hexagon_vsub_ss;

    llvm_unreachable("Incorrect input and output operand sizes");

  case OP_Mul:
    assert(ResEltSize = 2 * InEltSize);
    // Enter inside 'if' block when one of the operand is constant vector
    if (IsConstScalar) {
      // When inputs are of 8bit type and output is 16bit type, enter 'if' block
      if (InEltSize == 8 && ResEltSize == 16) {
        // Enter the 'if' block, when 2nd operand of the mul instruction is
        // constant vector, otherwise enter 'else' block
        if (ConstOpNum == 2 && IsOP1ZExt) {
          // If the value inside the constant vector is zero-extended, then
          // return hexagon_vmpy_ub_ub, else return hexagon_vmpy_ub_b
          return (IsOP2ZExt) ? Intrinsic::hexagon_vmpy_ub_ub
                             : Intrinsic::hexagon_vmpy_ub_b;
        } else if (ConstOpNum == 1 && IsOP2ZExt) {
          return (IsOP1ZExt) ? Intrinsic::hexagon_vmpy_ub_ub
                             : Intrinsic::hexagon_vmpy_ub_b;
        }
      }
      // When inputs are of 16bit type and output is 32bit type,
      // enter 'if' block
      if (InEltSize == 16 && ResEltSize == 32) {
        if (IsOP1ZExt && IsOP2ZExt) {
          // If the value inside the constant vector and other operand is
          // zero-extended, then return hexagon_vmpy_uh_uh
          return Intrinsic::hexagon_vmpy_uh_uh;
        } else if (!IsOP1ZExt && !IsOP2ZExt) {
          // If the value inside the constant vector and other operand is
          // sign-extended, then return hexagon_vmpy_h_h
          return Intrinsic::hexagon_vmpy_h_h;
        }
      }
    }
    if (IsOP1ZExt)
      return IsOP2ZExt ? Intrinsic::hexagon_vmpy_uu
                       : Intrinsic::hexagon_vmpy_us;
    else
      return IsOP2ZExt ? Intrinsic::hexagon_vmpy_su
                       : Intrinsic::hexagon_vmpy_ss;
  default:
    llvm_unreachable("Instruction not handled!");
  }
}

Type *HexagonGenWideningVecInstr::getElementTy(unsigned size,
                                               IRBuilder<> &IRB) {
  switch (size) {
  case 8:
    return IRB.getInt8Ty();
  case 16:
    return IRB.getInt16Ty();
  case 32:
    return IRB.getInt32Ty();
  default:
    llvm_unreachable("Unhandled Element size");
  }
}

Value *HexagonGenWideningVecInstr::createIntrinsic(
    Intrinsic::ID IntId, Instruction *Inst, Value *NewOP1, Value *NewOP2,
    Type *ResType, unsigned NumElts, bool Interleave = true) {
  IRBuilder<> IRB(Inst);
  Function *ExtF = Intrinsic::getOrInsertDeclaration(M, IntId, ResType);
  Value *NewIn = IRB.CreateCall(ExtF, {NewOP1, NewOP2});
  if (Interleave) {
    // Interleave elements in the output vector.
    SmallVector<Constant *, 16> ShuffleMask;
    unsigned HalfElts = NumElts / 2;
    for (unsigned i = 0; i < HalfElts; ++i) {
      ShuffleMask.push_back(IRB.getInt32(i));
      ShuffleMask.push_back(IRB.getInt32(HalfElts + i));
    }
    NewIn = IRB.CreateShuffleVector(NewIn, PoisonValue::get(ResType),
                                    ConstantVector::get(ShuffleMask));
  }
  return NewIn;
}

std::pair<Value *, Value *>
HexagonGenWideningVecInstr::opSplit(Value *OP, Instruction *Inst,
                                    Type *NewOpType) {
  Type *InstTy = Inst->getType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  IRBuilder<> IRB(Inst);
  if (InstTy->getPrimitiveSizeInBits() < 2 * HwVLen) {
    // The only time we need to split an OP even though it is not a
    // vector-pair is while generating vasr instruction for the short vector.
    // Since hi/lo intrinsics can't be used here as they expect the operands to
    // be of 64xi32 type, the shuffle_vector pair with the appropriate masks is
    // used instead.
    assert(NumElts % 2 == 0 && "Unexpected Vector Type!!");
    unsigned HalfElts = NumElts / 2;
    SmallVector<Constant *, 8> HiM;
    SmallVector<Constant *, 8> LoM;
    for (unsigned i = 0; i < HalfElts; ++i)
      LoM.push_back(IRB.getInt32(i));
    for (unsigned i = 0; i < HalfElts; ++i)
      HiM.push_back(IRB.getInt32(HalfElts + i));

    Value *Hi = IRB.CreateShuffleVector(OP, PoisonValue::get(OP->getType()),
                                        ConstantVector::get(HiM));
    Value *Lo = IRB.CreateShuffleVector(OP, PoisonValue::get(OP->getType()),
                                        ConstantVector::get(LoM));
    return std::pair<Value *, Value *>(Hi, Lo);
  }

  Intrinsic::ID IntHi = Intrinsic::hexagon_V6_hi_128B;
  Intrinsic::ID IntLo = Intrinsic::hexagon_V6_lo_128B;
  Function *ExtFHi = Intrinsic::getOrInsertDeclaration(M, IntHi);
  Function *ExtFLo = Intrinsic::getOrInsertDeclaration(M, IntLo);
  auto *InType = FixedVectorType::get(IRB.getInt32Ty(), 64);
  OP = IRB.CreateBitCast(OP, InType);
  Value *Hi = IRB.CreateCall(ExtFHi, {OP}); // 32xi32
  Value *Lo = IRB.CreateCall(ExtFLo, {OP});
  Hi = IRB.CreateBitCast(Hi, NewOpType);
  Lo = IRB.CreateBitCast(Lo, NewOpType);
  return std::pair<Value *, Value *>(Hi, Lo);
}

bool HexagonGenWideningVecInstr::checkConstantVector(Value *OP,
                                                     int64_t &SplatVal,
                                                     bool IsOPZExt) {
  if (auto *C1 = dyn_cast<Constant>(OP)) {
    if (Value *SplatV = C1->getSplatValue()) {
      auto *CI = dyn_cast<ConstantInt>(SplatV);
      if (IsOPZExt) {
        SplatVal = CI->getZExtValue();
      } else {
        SplatVal = CI->getSExtValue();
      }
      return true;
    }
  }
  return false;
}

void HexagonGenWideningVecInstr::updateMPYConst(Intrinsic::ID IntId,
                                                int64_t &SplatVal,
                                                bool IsOPZExt, Value *&OP,
                                                IRBuilder<> &IRB) {
  if ((IntId == Intrinsic::hexagon_vmpy_uu ||
       IntId == Intrinsic::hexagon_vmpy_us ||
       IntId == Intrinsic::hexagon_vmpy_su ||
       IntId == Intrinsic::hexagon_vmpy_ss) &&
      OP->getType()->isVectorTy()) {
    // Create a vector with all elements equal to SplatVal
    auto *VecTy = cast<VectorType>(OP->getType());
    Value *scalar = IRB.getIntN(VecTy->getScalarSizeInBits(),
                                static_cast<uint32_t>(SplatVal));
    Value *splatVector = ConstantVector::getSplat(VecTy->getElementCount(),
                                                  cast<Constant>(scalar));
    OP = IsOPZExt ? IRB.CreateZExt(splatVector, VecTy)
                  : IRB.CreateSExt(splatVector, VecTy);
  } else {
    packConstant(IntId, SplatVal, OP, IRB);
  }
}

void HexagonGenWideningVecInstr::packConstant(Intrinsic::ID IntId,
                                              int64_t &SplatVal, Value *&OP,
                                              IRBuilder<> &IRB) {
  uint32_t Val32 = static_cast<uint32_t>(SplatVal);
  if (IntId == Intrinsic::hexagon_vmpy_ub_ub) {
    assert(SplatVal >= 0 && SplatVal <= UINT8_MAX);
    uint32_t packed = (Val32 << 24) | (Val32 << 16) | (Val32 << 8) | Val32;
    OP = IRB.getInt32(packed);
  } else if (IntId == Intrinsic::hexagon_vmpy_ub_b) {
    assert(SplatVal >= INT8_MIN && SplatVal <= INT8_MAX);
    uint32_t packed = (Val32 << 24) | ((Val32 << 16) & ((1 << 24) - 1)) |
                      ((Val32 << 8) & ((1 << 16) - 1)) |
                      (Val32 & ((1 << 8) - 1));
    OP = IRB.getInt32(packed);
  } else if (IntId == Intrinsic::hexagon_vmpy_uh_uh) {
    assert(SplatVal >= 0 && SplatVal <= UINT16_MAX);
    uint32_t packed = (Val32 << 16) | Val32;
    OP = IRB.getInt32(packed);
  } else if (IntId == Intrinsic::hexagon_vmpy_h_h) {
    assert(SplatVal >= INT16_MIN && SplatVal <= INT16_MAX);
    uint32_t packed = (Val32 << 16) | (Val32 & ((1 << 16) - 1));
    OP = IRB.getInt32(packed);
  }
}

bool HexagonGenWideningVecInstr::replaceWithIntrinsic(Instruction *Inst,
                                                      OPKind OPK,
                                                      OPInfo &OP1Info,
                                                      OPInfo &OP2Info) {
  Type *InstTy = Inst->getType();
  Type *EltTy = cast<FixedVectorType>(InstTy)->getElementType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  unsigned InstEltSize = EltTy->getPrimitiveSizeInBits();

  bool IsOP1ZExt = OP1Info.IsZExt;
  bool IsOP2ZExt = OP2Info.IsZExt;

  // The resulting values of 'add' and 'sub' are always sign-extended.
  bool IsResZExt = (OPK == OP_Mul || OPK == OP_Shl)
                       ? (OP1Info.IsZExt && OP2Info.IsZExt)
                       : false;

  unsigned MaxEltSize = std::max(OP1Info.ExtInSize, OP2Info.ExtInSize);
  unsigned NewOpEltSize = MaxEltSize;
  unsigned NewResEltSize = 2 * MaxEltSize;

  // For Add and Sub, both the operands should be either zero extended
  // or sign extended. In case of a mismatch, they are extended  to the
  // next size (ex: 8 bits -> 16 bits) so that the sign-extended vadd/vsub
  // instructions can be used. Also, we don't support 8-bits signed vadd/vsub
  // instructions. They are extended to 16-bits and then signed 16-bits
  // non-widening vadd/vsub is used to perform the operation.
  if (OPK != OP_Mul && OPK != OP_Shl &&
      (IsOP1ZExt != IsOP2ZExt || (!IsOP1ZExt && NewOpEltSize == 8)))
    NewOpEltSize = 2 * NewOpEltSize;

  unsigned ResVLen = NewResEltSize * NumElts;
  if (ResVLen < HwVLen && !WidenShortVector)
    return false;
  if (NewOpEltSize > 16 || ((ResVLen > HwVLen) && (ResVLen % HwVLen) != 0))
    return false;

  IRBuilder<> IRB(Inst);
  Value *NewOP1 = adjustExtensionForOp(OP1Info, IRB, NewOpEltSize, NumElts);
  Value *NewOP2 = adjustExtensionForOp(OP2Info, IRB, NewOpEltSize, NumElts);

  if (NewOpEltSize == NewResEltSize) {
    assert(OPK != OP_Mul && OPK != OP_Shl);
    // Instead of intrinsics, use vector add/sub.
    Value *NewIn = IRB.CreateBinOp(cast<BinaryOperator>(Inst)->getOpcode(),
                                   NewOP1, NewOP2);
    if (InstEltSize > NewResEltSize)
      NewIn = IRB.CreateSExt(NewIn, InstTy);
    Inst->replaceAllUsesWith(NewIn);
    return true;
  }

  bool IsConstScalar = false;
  int64_t SplatVal = 0;
  int ConstOpNum = 1;
  if (OPK == OP_Mul || OPK == OP_Shl) {
    IsConstScalar = checkConstantVector(NewOP1, SplatVal, IsOP1ZExt);
    if (!IsConstScalar) {
      IsConstScalar = checkConstantVector(NewOP2, SplatVal, IsOP2ZExt);
      ConstOpNum = 2;
    }
  }

  if (IsConstScalar && OPK == OP_Shl) {
    if (((NewOpEltSize == 8) && (SplatVal > 0) && (SplatVal < 8)) ||
        ((NewOpEltSize == 16) && (SplatVal > 0) && (SplatVal < 16))) {
      SplatVal = 1 << SplatVal;
      OPK = OP_Mul;
    } else {
      return false;
    }
  } else if (!IsConstScalar && OPK == OP_Shl) {
    return false;
  }

  Intrinsic::ID IntId = getIntrinsic(OPK, IsOP1ZExt, IsOP2ZExt, NewOpEltSize,
                                     NewResEltSize, IsConstScalar, ConstOpNum);

  if (IsConstScalar) {
    updateMPYConst(IntId, SplatVal, IsOP2ZExt, NewOP2, IRB);
  }

  // Split the node if it needs more than a vector pair for the result.
  if (ResVLen > 2 * HwVLen) {
    assert(ResVLen == 4 * HwVLen);
    // Split the operands
    unsigned HalfElts = NumElts / 2;
    auto *NewOpType =
        FixedVectorType::get(getElementTy(NewOpEltSize, IRB), HalfElts);
    auto *ResType =
        FixedVectorType::get(getElementTy(NewResEltSize, IRB), HalfElts);
    std::pair<Value *, Value *> SplitOP1 = opSplit(NewOP1, Inst, NewOpType);
    std::pair<Value *, Value *> SplitOP2;
    if (IsConstScalar && (IntId == Intrinsic::hexagon_vmpy_h_h ||
                          IntId == Intrinsic::hexagon_vmpy_uh_uh)) {
      SplitOP2 = std::pair<Value *, Value *>(NewOP2, NewOP2);
    } else {
      SplitOP2 = opSplit(NewOP2, Inst, NewOpType);
    }
    Value *NewInHi = createIntrinsic(IntId, Inst, SplitOP1.first,
                                     SplitOP2.first, ResType, HalfElts, true);
    Value *NewInLo = createIntrinsic(IntId, Inst, SplitOP1.second,
                                     SplitOP2.second, ResType, HalfElts, true);
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
  Value *NewIn =
      createIntrinsic(IntId, Inst, NewOP1, NewOP2, ResType, NumElts, true);
  if (InstEltSize > NewResEltSize)
    NewIn = (IsResZExt) ? IRB.CreateZExt(NewIn, InstTy)
                        : IRB.CreateSExt(NewIn, InstTy);

  Inst->replaceAllUsesWith(NewIn);

  return true;
}

// Process instruction and replace them with widening vector
// intrinsics if possible.
bool HexagonGenWideningVecInstr::processInstruction(Instruction *Inst) {
  Type *InstTy = Inst->getType();
  if (!InstTy->isVectorTy() ||
      cast<FixedVectorType>(InstTy)->getNumElements() > 128)
    return false;
  unsigned InstLen = InstTy->getPrimitiveSizeInBits();
  if (!HST->isTypeForHVX(cast<VectorType>(InstTy)) && InstLen != 4 * HwVLen)
    return false;
  if (InstLen < HwVLen && !WidenShortVector)
    return false;

  using namespace PatternMatch;

  OPKind OPK;
  Value *OP1 = nullptr, *OP2 = nullptr;
  if (match(Inst, (m_Sub(m_Value(OP1), m_Value(OP2)))))
    OPK = OP_Sub;
  else if (match(Inst, (m_Add(m_Value(OP1), m_Value(OP2)))))
    OPK = OP_Add;
  else if (match(Inst, (m_Mul(m_Value(OP1), m_Value(OP2)))))
    OPK = OP_Mul;
  else if (match(Inst, (m_Shl(m_Value(OP1), m_Value(OP2)))))
    OPK = OP_Shl;
  else
    return false;

  OPInfo OP1Info, OP2Info;

  if (!getOperandInfo(OP1, OP1Info) || !getOperandInfo(OP2, OP2Info))
    return false;

  // Proceed only if both input operands are extended.
  if (!OP1Info.ExtInOP || !OP2Info.ExtInOP)
    return false;

  return replaceWithIntrinsic(Inst, OPK, OP1Info, OP2Info);
}

bool HexagonGenWideningVecInstr::getVmpaOperandInfo(Value *V, OPInfo &OPI) {
  using namespace PatternMatch;
  OPI.OP = V;
  Value *ExtV, *OP1 = nullptr;

  if (match(V,
            m_ZExt(m_Shuffle(m_InsertElt(m_Poison(), m_Value(ExtV), m_Zero()),
                             m_Poison(), m_ZeroMask()))) ||
      match(V,
            m_Shuffle(m_InsertElt(m_Poison(), m_ZExt(m_Value(ExtV)), m_Zero()),
                      m_Poison(), m_ZeroMask()))) {
    OPI.ExtInOP = ExtV;
    OPI.IsZExt = true;
    OPI.IsScalar = true;
    OPI.ExtInSize = ExtV->getType()->getPrimitiveSizeInBits();
    return true;
  }

  ConstantInt *I = nullptr;
  if ((match(V, m_Shuffle(m_InsertElt(m_Poison(), m_Value(ExtV), m_Zero()),
                          m_Poison(), m_ZeroMask())))) {
    if (match(ExtV, m_And(m_Value(OP1), m_ConstantInt(I)))) {
      uint32_t IValue = I->getZExtValue();
      if (IValue <= 255) {
        OPI.ExtInOP = ExtV;
        OPI.IsZExt = true;
        OPI.ExtInSize = 8;
        OPI.IsScalar = true;
        return true;
      }
    }
  }

  // Match for non-scalar operands
  return getOperandInfo(V, OPI);
}

// Process instruction and replace with the vmpa intrinsic if possible.
bool HexagonGenWideningVecInstr::processInstructionForVMPA(Instruction *Inst) {
  using namespace PatternMatch;
  Type *InstTy = Inst->getType();
  // TODO: Extend it to handle short vector instructions (< HwVLen).
  // vmpa instructions produce a vector register pair.
  if (!InstTy->isVectorTy() || InstTy->getPrimitiveSizeInBits() != 2 * HwVLen)
    return false;

  Value *OP1 = nullptr, *OP2 = nullptr;
  if (!match(Inst, (m_Add(m_Value(OP1), m_Value(OP2)))))
    return false;

  Value *OP[4] = {nullptr, nullptr, nullptr, nullptr};
  if (!match(OP1, m_Mul(m_Value(OP[0]), m_Value(OP[1]))) ||
      !match(OP2, m_Mul(m_Value(OP[2]), m_Value(OP[3]))))
    return false;

  OPInfo OP_Info[4];
  for (unsigned i = 0; i < 4; i++)
    if (!getVmpaOperandInfo(OP[i], OP_Info[i]) || !OP_Info[i].ExtInOP)
      return false;

  return replaceWithVmpaIntrinsic(Inst, OP_Info);
}

// Reorder operand info in OPI so that the vector operands come before their
// scalar counterparts.
void HexagonGenWideningVecInstr::reorderVmpaOperands(OPInfo *OPI) {
  for (unsigned i = 0; i < 2; i++)
    if (!OPI[2 * i].ExtInOP->getType()->isVectorTy()) {
      OPInfo Temp;
      Temp = OPI[2 * i];
      OPI[2 * i] = OPI[2 * i + 1];
      OPI[2 * i + 1] = Temp;
    }
}

// Only handles the case where one input to vmpa has to be a scalar
// and another is a vector. It can be easily extended to cover
// other types of vmpa instructions.
bool HexagonGenWideningVecInstr::replaceWithVmpaIntrinsic(Instruction *Inst,
                                                          OPInfo *OPI) {
  reorderVmpaOperands(OPI);

  // After reordering of the operands in OPI, the odd elements must have
  // IsScalar flag set to true. Also, check the even elements for non-scalars.
  if (!OPI[1].IsScalar || !OPI[3].IsScalar || OPI[0].IsScalar ||
      OPI[2].IsScalar)
    return false;

  OPInfo SOPI1 = OPI[1];
  OPInfo SOPI2 = OPI[3];

  // The scalar operand in the vmpa instructions needs to be an int8.
  if (SOPI1.ExtInSize != SOPI2.ExtInSize || SOPI1.ExtInSize != 8)
    return false;

  Type *InstTy = Inst->getType();
  Type *EltTy = cast<FixedVectorType>(InstTy)->getElementType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  unsigned InstEltSize = EltTy->getPrimitiveSizeInBits();

  unsigned MaxVEltSize = std::max(OPI[0].ExtInSize, OPI[2].ExtInSize);
  unsigned NewVOpEltSize = MaxVEltSize;
  unsigned NewResEltSize = 2 * MaxVEltSize;

  if (NumElts * NewVOpEltSize < HwVLen) {
    // Extend the operand so that we don't end up with an invalid vector size.
    NewVOpEltSize = 2 * NewVOpEltSize;
    NewResEltSize = 2 * NewResEltSize;
  }

  IRBuilder<> IRB(Inst);

  // Construct scalar operand
  Value *NewSOP1 = SOPI1.ExtInOP;
  Value *NewSOP2 = SOPI2.ExtInOP;

  Type *S1Ty = NewSOP1->getType();
  Type *S2Ty = NewSOP2->getType();
  if (S1Ty->getPrimitiveSizeInBits() < 32)
    NewSOP1 = IRB.CreateZExt(NewSOP1, IRB.getInt32Ty());
  if (S2Ty->getPrimitiveSizeInBits() < 32)
    NewSOP2 = IRB.CreateZExt(NewSOP2, IRB.getInt32Ty());

  Value *SHL = IRB.CreateShl(NewSOP1, IRB.getInt32(8));
  Value *OR = IRB.CreateOr(SHL, NewSOP2);
  Intrinsic::ID CombineIntID = Intrinsic::hexagon_A2_combine_ll;
  Function *ExtF = Intrinsic::getOrInsertDeclaration(M, CombineIntID);
  Value *ScalarOP = IRB.CreateCall(ExtF, {OR, OR});

  // Construct vector operand
  Value *NewVOP1 = adjustExtensionForOp(OPI[0], IRB, NewVOpEltSize, NumElts);
  Value *NewVOP2 = adjustExtensionForOp(OPI[2], IRB, NewVOpEltSize, NumElts);

  // Combine both vector operands to form the vector-pair for vmpa
  Intrinsic::ID VCombineIntID = Intrinsic::hexagon_V6_vcombine_128B;
  ExtF = Intrinsic::getOrInsertDeclaration(M, VCombineIntID);
  Type *InType = FixedVectorType::get(IRB.getInt32Ty(), 32);
  NewVOP1 = IRB.CreateBitCast(NewVOP1, InType);
  NewVOP2 = IRB.CreateBitCast(NewVOP2, InType);
  Value *VecOP = IRB.CreateCall(ExtF, {NewVOP1, NewVOP2});

  Intrinsic::ID VmpaIntID =
      (NewResEltSize == 16) ? VmpaIntID = Intrinsic::hexagon_V6_vmpabus_128B
                            : VmpaIntID = Intrinsic::hexagon_V6_vmpauhb_128B;
  ExtF = Intrinsic::getOrInsertDeclaration(M, VmpaIntID);
  auto *ResType =
      FixedVectorType::get(getElementTy(NewResEltSize, IRB), NumElts);
  Value *NewIn = IRB.CreateCall(ExtF, {VecOP, ScalarOP});
  NewIn = IRB.CreateBitCast(NewIn, ResType);

  if (InstEltSize > NewResEltSize)
    // Extend the output to match the original instruction type.
    NewIn = IRB.CreateSExt(NewIn, InstTy);

  // Interleave elements in the output vector.
  SmallVector<Constant *, 16> ShuffleMask;
  unsigned HalfElts = NumElts / 2;
  for (unsigned i = 0; i < HalfElts; ++i) {
    ShuffleMask.push_back(IRB.getInt32(i));
    ShuffleMask.push_back(IRB.getInt32(HalfElts + i));
  }
  NewIn = IRB.CreateShuffleVector(NewIn, PoisonValue::get(ResType),
                                  ConstantVector::get(ShuffleMask));

  Inst->replaceAllUsesWith(NewIn);
  return true;
}

bool HexagonGenWideningVecInstr::genSaturatingInst(Instruction *Inst) {
  Type *InstTy = Inst->getType();
  assert(InstTy->isVectorTy());
  if (InstTy->getPrimitiveSizeInBits() > HwVLen)
    return false;

  using namespace PatternMatch;
  CmpPredicate P1, P2;
  Value *L1 = nullptr, *T1 = nullptr, *L2 = nullptr, *T2 = nullptr,
        *L3 = nullptr;
  Constant *RC1 = nullptr, *FC1 = nullptr, *RC2 = nullptr, *FC2 = nullptr,
           *RC3 = nullptr;

  // Pattern of interest: ashr -> llvm.smin -> llvm.smax -> trunc
  // Match trunc instruction
  if (match(Inst, m_Trunc(m_Intrinsic<Intrinsic::smax>(m_Value(L1),
                                                       m_Constant(RC1))))) {
    // Match llvm.smin instruction
    if (match(L1, m_Intrinsic<Intrinsic::smin>(m_Value(L2), m_Constant(RC2)))) {
      // Match ashr instruction
      if (match(L2, m_AShr(m_Value(L3), m_Constant(RC3)))) {
        std::pair<int, int> MinMax;
        // get min, max values from operatands of smin and smax
        if (getMinMax(RC1, RC2, MinMax)) {
          bool IsResSigned;
          // Validate the saturating vasr pattern
          if (isSaturatingVAsr(Inst, L2, MinMax.first, MinMax.second,
                               IsResSigned)) {
            // Get the shift value from the ashr operand
            ConstantInt *shift_val =
                dyn_cast<ConstantInt>(RC3->getSplatValue());
            if (shift_val) {
              Value *NewIn =
                  createVAsrIntrinsic(Inst, L3, shift_val, IsResSigned);
              Inst->replaceAllUsesWith(NewIn);
              return true;
            }
          }
        }
      }
    }
  }

  if (!match(Inst, (m_Trunc(m_Select(m_ICmp(P1, m_Value(L1), m_Constant(RC1)),
                                     m_Value(T1), m_Constant(FC1))))) ||
      (T1 != L1 || FC1 != RC1))
    return false;

  if (!match(L1, m_Select(m_ICmp(P2, m_Value(L2), m_Constant(RC2)), m_Value(T2),
                          m_Constant(FC2))) ||
      (T2 != L2 || FC2 != RC2))
    return false;

  if (!((P1 == CmpInst::ICMP_SGT && P2 == CmpInst::ICMP_SLT) ||
        (P1 == CmpInst::ICMP_SLT && P2 == CmpInst::ICMP_SGT)))
    return false;

  std::pair<int, int> MinMax;
  if ((P1 == CmpInst::ICMP_SGT) && (P2 == CmpInst::ICMP_SLT)) {
    if (!getMinMax(RC1, RC2, MinMax))
      return false;
  } else if (!getMinMax(RC2, RC1, MinMax))
    return false;

  Value *S = L2; // Value being saturated

  // Only AShr instructions are handled.
  // Also, second operand to AShr must be a scalar.
  Value *OP1 = nullptr, *ShiftByVal = nullptr;
  if (!match(S, m_AShr(m_Value(OP1),
                       m_Shuffle(m_InsertElt(m_Poison(), m_Value(ShiftByVal),
                                             m_Zero()),
                                 m_Poison(), m_ZeroMask()))))
    return false;

  bool IsResSigned;
  if (!isSaturatingVAsr(Inst, S, MinMax.first, MinMax.second, IsResSigned))
    return false;

  Value *NewIn = createVAsrIntrinsic(Inst, OP1, ShiftByVal, IsResSigned);
  Inst->replaceAllUsesWith(NewIn);
  return true;
}

Value *HexagonGenWideningVecInstr::extendShiftByVal(Value *ShiftByVal,
                                                    IRBuilder<> &IRB) {
  using namespace PatternMatch;
  Value *A = nullptr;
  if (match(ShiftByVal, m_Trunc(m_Value(A))))
    return A;
  return IRB.CreateZExt(ShiftByVal, IRB.getInt32Ty());
}

bool HexagonGenWideningVecInstr::getMinMax(Constant *MinC, Constant *MaxC,
                                           std::pair<int, int> &MinMax) {
  Value *SplatV;
  if (!(SplatV = MinC->getSplatValue()) || !(dyn_cast<ConstantInt>(SplatV)))
    return false;
  if (!(SplatV = MaxC->getSplatValue()) || !(dyn_cast<ConstantInt>(SplatV)))
    return false;

  ConstantInt *MinI = dyn_cast<ConstantInt>(MinC->getSplatValue());
  ConstantInt *MaxI = dyn_cast<ConstantInt>(MaxC->getSplatValue());
  MinMax = std::pair<int, int>(MinI->getSExtValue(), MaxI->getSExtValue());
  return true;
}

bool HexagonGenWideningVecInstr::isSaturatingVAsr(Instruction *Inst, Value *S,
                                                  int MinV, int MaxV,
                                                  bool &IsResSigned) {
  if (MinV >= MaxV)
    return false;

  IsResSigned = true;
  Type *InstTy = Inst->getType();
  Type *EltTy = cast<VectorType>(InstTy)->getElementType();
  unsigned TruncSize = EltTy->getPrimitiveSizeInBits();

  int MaxRange, MinRange;
  if (MinV < 0) { // Saturate to a signed value
    MaxRange = (1 << (TruncSize - 1)) - 1;
    MinRange = -(1 << (TruncSize - 1));
  } else if (MinV == 0) { // Saturate to an unsigned value
    MaxRange = (1 << (TruncSize)) - 1;
    MinRange = 0;
    IsResSigned = false;
  } else
    return false;

  if (MinV != MinRange || MaxV != MaxRange)
    return false;

  auto *SInst = dyn_cast<Instruction>(S);
  if (SInst->getOpcode() == Instruction::AShr) {
    Type *SInstTy = SInst->getType();
    Type *SEltTy = cast<VectorType>(SInstTy)->getElementType();
    unsigned SInstEltSize = SEltTy->getPrimitiveSizeInBits();
    if (SInstEltSize != 2 * TruncSize || TruncSize > 16)
      return false;
  }
  return true;
}

Intrinsic::ID HexagonGenWideningVecInstr::getVAsrIntrinsic(bool IsInSigned,
                                                           bool IsResSigned) {
  if (!IsResSigned)
    return (IsInSigned) ? Intrinsic::hexagon_vasrsat_su
                        : Intrinsic::hexagon_vasrsat_uu;
  return Intrinsic::hexagon_vasrsat_ss;
}

Value *HexagonGenWideningVecInstr::createVAsrIntrinsic(Instruction *Inst,
                                                       Value *VecOP,
                                                       Value *ShiftByVal,
                                                       bool IsResSigned) {
  IRBuilder<> IRB(Inst);
  Type *ShiftByTy = ShiftByVal->getType();
  if (ShiftByTy->getPrimitiveSizeInBits() < 32)
    ShiftByVal = extendShiftByVal(ShiftByVal, IRB);

  Type *InstTy = Inst->getType();
  Type *EltTy = cast<FixedVectorType>(InstTy)->getElementType();
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  unsigned InstEltSize = EltTy->getPrimitiveSizeInBits();

  // Replace the instruction with saturating vasr intrinsic.
  // Since vasr with saturation interleaves elements from both input vectors,
  // they must be deinterleaved for output to end up in the right order.
  SmallVector<Constant *, 16> ShuffleMask;
  unsigned HalfElts = NumElts / 2;
  // Even elements
  for (unsigned i = 0; i < HalfElts; ++i)
    ShuffleMask.push_back(IRB.getInt32(i * 2));
  // Odd elements
  for (unsigned i = 0; i < HalfElts; ++i)
    ShuffleMask.push_back(IRB.getInt32(i * 2 + 1));

  VecOP = IRB.CreateShuffleVector(VecOP, PoisonValue::get(VecOP->getType()),
                                  ConstantVector::get(ShuffleMask));

  auto *InVecOPTy =
      FixedVectorType::get(getElementTy(InstEltSize * 2, IRB), HalfElts);
  std::pair<Value *, Value *> HiLo = opSplit(VecOP, Inst, InVecOPTy);
  Intrinsic::ID IntID = getVAsrIntrinsic(true, IsResSigned);
  Function *F = Intrinsic::getOrInsertDeclaration(M, IntID, InVecOPTy);
  Value *NewIn = IRB.CreateCall(F, {HiLo.first, HiLo.second, ShiftByVal});
  return IRB.CreateBitCast(NewIn, InstTy);
}

// Generate vavg instruction.
bool HexagonGenWideningVecInstr::genVAvg(Instruction *Inst) {
  using namespace PatternMatch;
  Type *InstTy = Inst->getType();
  assert(InstTy->isVectorTy());

  bool Match = false;
  Value *OP1 = nullptr, *OP2 = nullptr;
  bool IsSigned;
  if ((Match = (match(Inst, m_Trunc(m_LShr(m_Add(m_ZExt(m_Value(OP1)),
                                                 m_ZExt(m_Value(OP2))),
                                           m_SpecificInt(1)))))))
    IsSigned = false;
  if (!Match &&
      (Match = (match(Inst, m_Trunc(m_LShr(m_Add(m_SExt(m_Value(OP1)),
                                                 m_SExt(m_Value(OP2))),
                                           m_SpecificInt(1))))) ||
               match(Inst, m_LShr(m_Add(m_Value(OP1), m_Value(OP2)),
                                  m_SpecificInt(1)))))
    IsSigned = true;

  if (!Match)
    return false;

  unsigned OP1EltSize = getElementSizeInBits(OP1);
  unsigned OP2EltSize = getElementSizeInBits(OP2);
  unsigned NewEltSize = std::max(OP1EltSize, OP2EltSize);

  Type *EltTy = cast<FixedVectorType>(InstTy)->getElementType();
  unsigned InstEltSize = EltTy->getPrimitiveSizeInBits();
  unsigned InstLen = InstTy->getPrimitiveSizeInBits();

  // Only vectors that are either smaller, same or twice of the hardware
  // vector length are allowed.
  if (InstEltSize < NewEltSize || (InstLen > 2 * HwVLen))
    return false;

  if ((InstLen > HwVLen) && (InstLen % HwVLen != 0))
    return false;

  IRBuilder<> IRB(Inst);
  unsigned NumElts = cast<FixedVectorType>(InstTy)->getNumElements();
  auto *AvgInstTy =
      FixedVectorType::get(getElementTy(NewEltSize, IRB), NumElts);
  if (OP1EltSize < NewEltSize)
    OP1 = (IsSigned) ? IRB.CreateSExt(OP1, AvgInstTy)
                     : IRB.CreateZExt(OP1, AvgInstTy);
  if (OP2EltSize < NewEltSize)
    OP2 = (IsSigned) ? IRB.CreateSExt(OP2, AvgInstTy)
                     : IRB.CreateZExt(OP2, AvgInstTy);

  Intrinsic::ID AvgIntID =
      (IsSigned) ? Intrinsic::hexagon_vavgs : Intrinsic::hexagon_vavgu;
  Value *NewIn = nullptr;

  // Split operands if they need more than a vector length.
  if (NewEltSize * NumElts > HwVLen) {
    unsigned HalfElts = NumElts / 2;
    auto *ResType =
        FixedVectorType::get(getElementTy(NewEltSize, IRB), HalfElts);
    std::pair<Value *, Value *> SplitOP1 = opSplit(OP1, Inst, ResType);
    std::pair<Value *, Value *> SplitOP2 = opSplit(OP2, Inst, ResType);
    Value *NewHi = createIntrinsic(AvgIntID, Inst, SplitOP1.first,
                                   SplitOP2.first, ResType, NumElts, false);
    Value *NewLo = createIntrinsic(AvgIntID, Inst, SplitOP1.second,
                                   SplitOP2.second, ResType, NumElts, false);
    SmallVector<Constant *, 8> ShuffleMask;
    for (unsigned i = 0; i < NumElts; ++i)
      ShuffleMask.push_back(IRB.getInt32(i));
    // Concat Hi and Lo.
    NewIn =
        IRB.CreateShuffleVector(NewLo, NewHi, ConstantVector::get(ShuffleMask));
  } else
    NewIn =
        createIntrinsic(AvgIntID, Inst, OP1, OP2, AvgInstTy, NumElts, false);

  if (InstEltSize > NewEltSize)
    // Extend the output to match the original instruction type.
    NewIn = (IsSigned) ? IRB.CreateSExt(NewIn, InstTy)
                       : IRB.CreateZExt(NewIn, InstTy);
  Inst->replaceAllUsesWith(NewIn);
  return true;
}

bool HexagonGenWideningVecInstr::visitBlock(BasicBlock *B) {
  bool Changed = false;
  for (auto &I : *B) {
    Type *InstTy = I.getType();
    if (!InstTy->isVectorTy() || !HST->isTypeForHVX(cast<VectorType>(InstTy)))
      continue;

    unsigned InstLen = InstTy->getPrimitiveSizeInBits();
    if (InstLen < HwVLen && !WidenShortVector)
      continue;

    Changed |= processInstructionForVMPA(&I);
    Changed |= genSaturatingInst(&I);
    Changed |= genVAvg(&I);
  }
  // Generate widening instructions.
  for (auto &I : *B)
    Changed |= processInstruction(&I);
  return Changed;
}

bool HexagonGenWideningVecInstr::runOnFunction(Function &F) {
  M = F.getParent();
  HST = TM->getSubtargetImpl(F);

  // Return if useHVX128BOps is not set. It can be enabled for 64B mode
  // but wil require some changes. For example, bitcast for intrinsics
  // assumes 128B mode.
  if (skipFunction(F) || !HST->useHVX128BOps())
    return false;

  HwVLen = HST->getVectorLength() * 8; // Vector Length in bits
  bool Changed = false;
  for (auto &B : F)
    Changed |= visitBlock(&B);

  return Changed;
}

FunctionPass *
llvm::createHexagonGenWideningVecInstr(const HexagonTargetMachine &TM) {
  return new HexagonGenWideningVecInstr(&TM);
}
