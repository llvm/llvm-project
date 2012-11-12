// llvm/Target/TargetTransformImpl.cpp - Target Loop Trans Info ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetTransformImpl.h"
#include "llvm/Target/TargetLowering.h"
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
//
// Calls used by scalar transformations.
//
//===----------------------------------------------------------------------===//

bool ScalarTargetTransformImpl::isLegalAddImmediate(int64_t imm) const {
  return TLI->isLegalAddImmediate(imm);
}

bool ScalarTargetTransformImpl::isLegalICmpImmediate(int64_t imm) const {
  return TLI->isLegalICmpImmediate(imm);
}

bool ScalarTargetTransformImpl::isLegalAddressingMode(const AddrMode &AM,
                                                      Type *Ty) const {
  return TLI->isLegalAddressingMode(AM, Ty);
}

bool ScalarTargetTransformImpl::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TLI->isTruncateFree(Ty1, Ty2);
}

bool ScalarTargetTransformImpl::isTypeLegal(Type *Ty) const {
  EVT T = TLI->getValueType(Ty);
  return TLI->isTypeLegal(T);
}

unsigned ScalarTargetTransformImpl::getJumpBufAlignment() const {
  return TLI->getJumpBufAlignment();
}

unsigned ScalarTargetTransformImpl::getJumpBufSize() const {
  return TLI->getJumpBufSize();
}

bool ScalarTargetTransformImpl::shouldBuildLookupTables() const {
  return TLI->supportJumpTables() &&
      (TLI->isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
       TLI->isOperationLegalOrCustom(ISD::BRIND, MVT::Other));
}

//===----------------------------------------------------------------------===//
//
// Calls used by the vectorizers.
//
//===----------------------------------------------------------------------===//
int VectorTargetTransformImpl::InstructionOpcodeToISD(unsigned Opcode) const {
  enum InstructionOpcodes {
#define HANDLE_INST(NUM, OPCODE, CLASS) OPCODE = NUM,
#define LAST_OTHER_INST(NUM) InstructionOpcodesCount = NUM
#include "llvm/Instruction.def"
  };
  switch (static_cast<InstructionOpcodes>(Opcode)) {
  case Ret:            return 0;
  case Br:             return 0;
  case Switch:         return 0;
  case IndirectBr:     return 0;
  case Invoke:         return 0;
  case Resume:         return 0;
  case Unreachable:    return 0;
  case Add:            return ISD::ADD;
  case FAdd:           return ISD::FADD;
  case Sub:            return ISD::SUB;
  case FSub:           return ISD::FSUB;
  case Mul:            return ISD::MUL;
  case FMul:           return ISD::FMUL;
  case UDiv:           return ISD::UDIV;
  case SDiv:           return ISD::UDIV;
  case FDiv:           return ISD::FDIV;
  case URem:           return ISD::UREM;
  case SRem:           return ISD::SREM;
  case FRem:           return ISD::FREM;
  case Shl:            return ISD::SHL;
  case LShr:           return ISD::SRL;
  case AShr:           return ISD::SRA;
  case And:            return ISD::AND;
  case Or:             return ISD::OR;
  case Xor:            return ISD::XOR;
  case Alloca:         return 0;
  case Load:           return ISD::LOAD;
  case Store:          return ISD::STORE;
  case GetElementPtr:  return 0;
  case Fence:          return 0;
  case AtomicCmpXchg:  return 0;
  case AtomicRMW:      return 0;
  case Trunc:          return ISD::TRUNCATE;
  case ZExt:           return ISD::ZERO_EXTEND;
  case SExt:           return ISD::SIGN_EXTEND;
  case FPToUI:         return ISD::FP_TO_UINT;
  case FPToSI:         return ISD::FP_TO_SINT;
  case UIToFP:         return ISD::UINT_TO_FP;
  case SIToFP:         return ISD::SINT_TO_FP;
  case FPTrunc:        return ISD::FP_ROUND;
  case FPExt:          return ISD::FP_EXTEND;
  case PtrToInt:       return ISD::BITCAST;
  case IntToPtr:       return ISD::BITCAST;
  case BitCast:        return ISD::BITCAST;
  case ICmp:           return ISD::SETCC;
  case FCmp:           return ISD::SETCC;
  case PHI:            return 0;
  case Call:           return 0;
  case Select:         return ISD::SELECT;
  case UserOp1:        return 0;
  case UserOp2:        return 0;
  case VAArg:          return 0;
  case ExtractElement: return ISD::EXTRACT_VECTOR_ELT;
  case InsertElement:  return ISD::INSERT_VECTOR_ELT;
  case ShuffleVector:  return ISD::VECTOR_SHUFFLE;
  case ExtractValue:   return ISD::MERGE_VALUES;
  case InsertValue:    return ISD::MERGE_VALUES;
  case LandingPad:     return 0;
  }

  llvm_unreachable("Unknown instruction type encountered!");
}

std::pair<unsigned, MVT>
VectorTargetTransformImpl::getTypeLegalizationCost(Type *Ty) const {

  LLVMContext &C = Ty->getContext();
  EVT MTy = TLI->getValueType(Ty);

  unsigned Cost = 1;
  // We keep legalizing the type until we find a legal kind. We assume that
  // the only operation that costs anything is the split. After splitting
  // we need to handle two types.
  while (true) {
    TargetLowering::LegalizeKind LK = TLI->getTypeConversion(C, MTy);

    if (LK.first == TargetLowering::TypeLegal)
      return std::make_pair(Cost, MTy.getSimpleVT());

    if (LK.first == TargetLowering::TypeSplitVector ||
        LK.first == TargetLowering::TypeExpandInteger)
      Cost *= 2;

    // Keep legalizing the type.
    MTy = LK.second;
  }
}

unsigned
VectorTargetTransformImpl::getScalarizationOverhead(Type *Ty,
                                                    bool Insert,
                                                    bool Extract) const {
  assert (Ty->isVectorTy() && "Can only scalarize vectors");
  unsigned Cost = 0;

  for (int i = 0, e = Ty->getVectorNumElements(); i < e; ++i) {
    if (Insert)
      Cost += getVectorInstrCost(Instruction::InsertElement, Ty, i);
    if (Extract)
      Cost += getVectorInstrCost(Instruction::ExtractElement, Ty, i);
  }

  return Cost;
}

unsigned VectorTargetTransformImpl::getArithmeticInstrCost(unsigned Opcode,
                                                           Type *Ty) const {
  // Check if any of the operands are vector operands.
  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(Ty);

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1. Multiply
    // by the type-legalization overhead.
    return LT.first * 1;
  }

  // Else, assume that we need to scalarize this op.
  if (Ty->isVectorTy()) {
    unsigned Num = Ty->getVectorNumElements();
    unsigned Cost = getArithmeticInstrCost(Opcode, Ty->getScalarType());
    // return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(Ty, true, true) + Num * Cost;
  }

  // We don't know anything about this scalar instruction.
  return 1;
}

unsigned VectorTargetTransformImpl::getBroadcastCost(Type *Tp) const {
  return 1;
}

unsigned VectorTargetTransformImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                  Type *Src) const {
  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> SrcLT = getTypeLegalizationCost(Src);
  std::pair<unsigned, MVT> DstLT = getTypeLegalizationCost(Dst);

  // Handle scalar conversions.
  if (!Src->isVectorTy() && !Dst->isVectorTy()) {

    // Scalar bitcasts are usually free.
    if (Opcode == Instruction::BitCast)
      return 0;

    if (Opcode == Instruction::Trunc &&
        TLI->isTruncateFree(SrcLT.second, DstLT.second))
      return 0;

    if (Opcode == Instruction::ZExt &&
        TLI->isZExtFree(SrcLT.second, DstLT.second))
      return 0;

    // Just check the op cost. If the operation is legal then assume it costs 1.
    if (!TLI->isOperationExpand(ISD, DstLT.second))
      return  1;

    // Assume that illegal scalar instruction are expensive.
    return 4;
  }

  // Check vector-to-vector casts.
  if (Dst->isVectorTy() && Src->isVectorTy()) {

    // If the cast is between same-sized registers, then the check is simple.
    if (SrcLT.first == DstLT.first &&
        SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

      // Bitcast between types that are legalized to the same type are free.
      if (Opcode == Instruction::BitCast || Opcode == Instruction::Trunc)
        return 0;

      // Assume that Zext is done using AND.
      if (Opcode == Instruction::ZExt)
        return 1;

      // Assume that sext is done using SHL and SRA.
      if (Opcode == Instruction::SExt)
        return 2;

      // Just check the op cost. If the operation is legal then assume it costs
      // 1 and multiply by the type-legalization overhead.
      if (!TLI->isOperationExpand(ISD, DstLT.second))
        return SrcLT.first * 1;
    }

    // If we are converting vectors and the operation is illegal, or
    // if the vectors are legalized to different types, estimate the
    // scalarization costs.
    unsigned Num = Dst->getVectorNumElements();
    unsigned Cost = getCastInstrCost(Opcode, Dst->getScalarType(),
                                     Src->getScalarType());

    // Return the cost of multiple scalar invocation plus the cost of
    // inserting and extracting the values.
    return getScalarizationOverhead(Dst, true, true) + Num * Cost;
  }

  // We already handled vector-to-vector and scalar-to-scalar conversions. This 
  // is where we handle bitcast between vectors and scalars. We need to assume
  //  that the conversion is scalarized in one way or another.
  if (Opcode == Instruction::BitCast)
    // Illegal bitcasts are done by storing and loading from a stack slot.
    return (Src->isVectorTy()? getScalarizationOverhead(Src, false, true):0) +
           (Dst->isVectorTy()? getScalarizationOverhead(Dst, true, false):0);

  llvm_unreachable("Unhandled cast");
 }

unsigned VectorTargetTransformImpl::getCFInstrCost(unsigned Opcode) const {
  return 1;
}

unsigned VectorTargetTransformImpl::getCmpSelInstrCost(unsigned Opcode,
                                                       Type *ValTy,
                                                       Type *CondTy) const {
  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // Selects on vectors are actually vector selects.
  if (ISD == ISD::SELECT) {
    assert(CondTy && "CondTy must exist");
    if (CondTy->isVectorTy())
      ISD = ISD::VSELECT;
  }

  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(ValTy);

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1. Multiply
    // by the type-legalization overhead.
    return LT.first * 1;
  }

  // Otherwise, assume that the cast is scalarized.
  if (ValTy->isVectorTy()) {
    unsigned Num = ValTy->getVectorNumElements();
    if (CondTy)
      CondTy = CondTy->getScalarType();
    unsigned Cost = getCmpSelInstrCost(Opcode, ValTy->getScalarType(),
                                       CondTy);

    // Return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(ValTy, true, false) + Num * Cost;
  }

  // Unknown scalar opcode.
  return 1;
}

unsigned VectorTargetTransformImpl::getVectorInstrCost(unsigned Opcode,
                                                       Type *Val,
                                                       unsigned Index) const {
  return 1;
}

unsigned
VectorTargetTransformImpl::getInstrCost(unsigned Opcode, Type *Ty1,
                                        Type *Ty2) const {
  return 1;
}

unsigned
VectorTargetTransformImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                           unsigned Alignment,
                                           unsigned AddressSpace) const {
  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(Src);

  // Assume that all loads of legal types cost 1.
  return LT.first;
}

unsigned
VectorTargetTransformImpl::getNumberOfParts(Type *Tp) const {
  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(Tp);
  return LT.first;
}
