//===- ConstantFold.cpp - LLVM constant folder ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements folding of constants for LLVM.  This implements the
// (internal) ConstantFold.h interface, which is used by the
// ConstantExpr::get* methods to automatically fold constants when possible.
//
// The current constant folding implementation is implemented in two pieces: the
// pieces that don't need DataLayout, and the pieces that do. This is to avoid
// a dependence in IR on Target.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;
using namespace llvm::PatternMatch;

//===----------------------------------------------------------------------===//
//                ConstantFold*Instruction Implementations
//===----------------------------------------------------------------------===//

/// Convert the specified vector Constant node to the specified vector type.
/// At this point, we know that the elements of the input vector constant are
/// all simple integer or FP values.
static Constant *BitCastConstantVector(Constant *CV, VectorType *DstTy) {

  if (CV->isAllOnesValue()) return Constant::getAllOnesValue(DstTy);
  if (CV->isNullValue()) return Constant::getNullValue(DstTy);

  // Do not iterate on scalable vector. The num of elements is unknown at
  // compile-time.
  if (isa<ScalableVectorType>(DstTy))
    return nullptr;

  // If this cast changes element count then we can't handle it here:
  // doing so requires endianness information.  This should be handled by
  // Analysis/ConstantFolding.cpp
  unsigned NumElts = cast<FixedVectorType>(DstTy)->getNumElements();
  if (NumElts != cast<FixedVectorType>(CV->getType())->getNumElements())
    return nullptr;

  Type *DstEltTy = DstTy->getElementType();
  // Fast path for splatted constants.
  if (Constant *Splat = CV->getSplatValue()) {
    return ConstantVector::getSplat(DstTy->getElementCount(),
                                    ConstantExpr::getBitCast(Splat, DstEltTy));
  }

  SmallVector<Constant*, 16> Result;
  Type *Ty = IntegerType::get(CV->getContext(), 32);
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *C =
      ConstantExpr::getExtractElement(CV, ConstantInt::get(Ty, i));
    C = ConstantExpr::getBitCast(C, DstEltTy);
    Result.push_back(C);
  }

  return ConstantVector::get(Result);
}

/// This function determines which opcode to use to fold two constant cast
/// expressions together. It uses CastInst::isEliminableCastPair to determine
/// the opcode. Consequently its just a wrapper around that function.
/// Determine if it is valid to fold a cast of a cast
static unsigned
foldConstantCastPair(
  unsigned opc,          ///< opcode of the second cast constant expression
  ConstantExpr *Op,      ///< the first cast constant expression
  Type *DstTy            ///< destination type of the first cast
) {
  assert(Op && Op->isCast() && "Can't fold cast of cast without a cast!");
  assert(DstTy && DstTy->isFirstClassType() && "Invalid cast destination type");
  assert(CastInst::isCast(opc) && "Invalid cast opcode");

  // The types and opcodes for the two Cast constant expressions
  Type *SrcTy = Op->getOperand(0)->getType();
  Type *MidTy = Op->getType();
  Instruction::CastOps firstOp = Instruction::CastOps(Op->getOpcode());
  Instruction::CastOps secondOp = Instruction::CastOps(opc);

  // Assume that pointers are never more than 64 bits wide, and only use this
  // for the middle type. Otherwise we could end up folding away illegal
  // bitcasts between address spaces with different sizes.
  IntegerType *FakeIntPtrTy = Type::getInt64Ty(DstTy->getContext());

  // Let CastInst::isEliminableCastPair do the heavy lifting.
  return CastInst::isEliminableCastPair(firstOp, secondOp, SrcTy, MidTy, DstTy,
                                        nullptr, FakeIntPtrTy, nullptr);
}

static Constant *FoldBitCast(Constant *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy == DestTy)
    return V; // no-op cast

  // Check to see if we are casting a pointer to an aggregate to a pointer to
  // the first element.  If so, return the appropriate GEP instruction.
  if (PointerType *PTy = dyn_cast<PointerType>(V->getType()))
    if (PointerType *DPTy = dyn_cast<PointerType>(DestTy))
      if (PTy->getAddressSpace() == DPTy->getAddressSpace() &&
          !PTy->isOpaque() && !DPTy->isOpaque() &&
          PTy->getNonOpaquePointerElementType()->isSized()) {
        SmallVector<Value*, 8> IdxList;
        Value *Zero =
          Constant::getNullValue(Type::getInt32Ty(DPTy->getContext()));
        IdxList.push_back(Zero);
        Type *ElTy = PTy->getNonOpaquePointerElementType();
        while (ElTy && ElTy != DPTy->getNonOpaquePointerElementType()) {
          ElTy = GetElementPtrInst::getTypeAtIndex(ElTy, (uint64_t)0);
          IdxList.push_back(Zero);
        }

        if (ElTy == DPTy->getNonOpaquePointerElementType())
          // This GEP is inbounds because all indices are zero.
          return ConstantExpr::getInBoundsGetElementPtr(
              PTy->getNonOpaquePointerElementType(), V, IdxList);
      }

  // Handle casts from one vector constant to another.  We know that the src
  // and dest type have the same size (otherwise its an illegal cast).
  if (VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
    if (VectorType *SrcTy = dyn_cast<VectorType>(V->getType())) {
      assert(DestPTy->getPrimitiveSizeInBits() ==
                 SrcTy->getPrimitiveSizeInBits() &&
             "Not cast between same sized vectors!");
      SrcTy = nullptr;
      // First, check for null.  Undef is already handled.
      if (isa<ConstantAggregateZero>(V))
        return Constant::getNullValue(DestTy);

      // Handle ConstantVector and ConstantAggregateVector.
      return BitCastConstantVector(V, DestPTy);
    }

    // Canonicalize scalar-to-vector bitcasts into vector-to-vector bitcasts
    // This allows for other simplifications (although some of them
    // can only be handled by Analysis/ConstantFolding.cpp).
    if (isa<ConstantInt>(V) || isa<ConstantFP>(V))
      return ConstantExpr::getBitCast(ConstantVector::get(V), DestPTy);
  }

  // Finally, implement bitcast folding now.   The code below doesn't handle
  // bitcast right.
  if (isa<ConstantPointerNull>(V))  // ptr->ptr cast.
    return ConstantPointerNull::get(cast<PointerType>(DestTy));

  // Handle integral constant input.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (DestTy->isIntegerTy())
      // Integral -> Integral. This is a no-op because the bit widths must
      // be the same. Consequently, we just fold to V.
      return V;

    // See note below regarding the PPC_FP128 restriction.
    if (DestTy->isFloatingPointTy() && !DestTy->isPPC_FP128Ty())
      return ConstantFP::get(DestTy->getContext(),
                             APFloat(DestTy->getFltSemantics(),
                                     CI->getValue()));

    // Otherwise, can't fold this (vector?)
    return nullptr;
  }

  // Handle ConstantFP input: FP -> Integral.
  if (ConstantFP *FP = dyn_cast<ConstantFP>(V)) {
    // PPC_FP128 is really the sum of two consecutive doubles, where the first
    // double is always stored first in memory, regardless of the target
    // endianness. The memory layout of i128, however, depends on the target
    // endianness, and so we can't fold this without target endianness
    // information. This should instead be handled by
    // Analysis/ConstantFolding.cpp
    if (FP->getType()->isPPC_FP128Ty())
      return nullptr;

    // Make sure dest type is compatible with the folded integer constant.
    if (!DestTy->isIntegerTy())
      return nullptr;

    return ConstantInt::get(FP->getContext(),
                            FP->getValueAPF().bitcastToAPInt());
  }

  return nullptr;
}


/// V is an integer constant which only has a subset of its bytes used.
/// The bytes used are indicated by ByteStart (which is the first byte used,
/// counting from the least significant byte) and ByteSize, which is the number
/// of bytes used.
///
/// This function analyzes the specified constant to see if the specified byte
/// range can be returned as a simplified constant.  If so, the constant is
/// returned, otherwise null is returned.
static Constant *ExtractConstantBytes(Constant *C, unsigned ByteStart,
                                      unsigned ByteSize) {
  assert(C->getType()->isIntegerTy() &&
         (cast<IntegerType>(C->getType())->getBitWidth() & 7) == 0 &&
         "Non-byte sized integer input");
  unsigned CSize = cast<IntegerType>(C->getType())->getBitWidth()/8;
  assert(ByteSize && "Must be accessing some piece");
  assert(ByteStart+ByteSize <= CSize && "Extracting invalid piece from input");
  assert(ByteSize != CSize && "Should not extract everything");

  // Constant Integers are simple.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
    APInt V = CI->getValue();
    if (ByteStart)
      V.lshrInPlace(ByteStart*8);
    V = V.trunc(ByteSize*8);
    return ConstantInt::get(CI->getContext(), V);
  }

  // In the input is a constant expr, we might be able to recursively simplify.
  // If not, we definitely can't do anything.
  ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE) return nullptr;

  switch (CE->getOpcode()) {
  default: return nullptr;
  case Instruction::Or: {
    Constant *RHS = ExtractConstantBytes(CE->getOperand(1), ByteStart,ByteSize);
    if (!RHS)
      return nullptr;

    // X | -1 -> -1.
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS))
      if (RHSC->isMinusOne())
        return RHSC;

    Constant *LHS = ExtractConstantBytes(CE->getOperand(0), ByteStart,ByteSize);
    if (!LHS)
      return nullptr;
    return ConstantExpr::getOr(LHS, RHS);
  }
  case Instruction::And: {
    Constant *RHS = ExtractConstantBytes(CE->getOperand(1), ByteStart,ByteSize);
    if (!RHS)
      return nullptr;

    // X & 0 -> 0.
    if (RHS->isNullValue())
      return RHS;

    Constant *LHS = ExtractConstantBytes(CE->getOperand(0), ByteStart,ByteSize);
    if (!LHS)
      return nullptr;
    return ConstantExpr::getAnd(LHS, RHS);
  }
  case Instruction::LShr: {
    ConstantInt *Amt = dyn_cast<ConstantInt>(CE->getOperand(1));
    if (!Amt)
      return nullptr;
    APInt ShAmt = Amt->getValue();
    // Cannot analyze non-byte shifts.
    if ((ShAmt & 7) != 0)
      return nullptr;
    ShAmt.lshrInPlace(3);

    // If the extract is known to be all zeros, return zero.
    if (ShAmt.uge(CSize - ByteStart))
      return Constant::getNullValue(
          IntegerType::get(CE->getContext(), ByteSize * 8));
    // If the extract is known to be fully in the input, extract it.
    if (ShAmt.ule(CSize - (ByteStart + ByteSize)))
      return ExtractConstantBytes(CE->getOperand(0),
                                  ByteStart + ShAmt.getZExtValue(), ByteSize);

    // TODO: Handle the 'partially zero' case.
    return nullptr;
  }

  case Instruction::Shl: {
    ConstantInt *Amt = dyn_cast<ConstantInt>(CE->getOperand(1));
    if (!Amt)
      return nullptr;
    APInt ShAmt = Amt->getValue();
    // Cannot analyze non-byte shifts.
    if ((ShAmt & 7) != 0)
      return nullptr;
    ShAmt.lshrInPlace(3);

    // If the extract is known to be all zeros, return zero.
    if (ShAmt.uge(ByteStart + ByteSize))
      return Constant::getNullValue(
          IntegerType::get(CE->getContext(), ByteSize * 8));
    // If the extract is known to be fully in the input, extract it.
    if (ShAmt.ule(ByteStart))
      return ExtractConstantBytes(CE->getOperand(0),
                                  ByteStart - ShAmt.getZExtValue(), ByteSize);

    // TODO: Handle the 'partially zero' case.
    return nullptr;
  }

  case Instruction::ZExt: {
    unsigned SrcBitSize =
      cast<IntegerType>(CE->getOperand(0)->getType())->getBitWidth();

    // If extracting something that is completely zero, return 0.
    if (ByteStart*8 >= SrcBitSize)
      return Constant::getNullValue(IntegerType::get(CE->getContext(),
                                                     ByteSize*8));

    // If exactly extracting the input, return it.
    if (ByteStart == 0 && ByteSize*8 == SrcBitSize)
      return CE->getOperand(0);

    // If extracting something completely in the input, if the input is a
    // multiple of 8 bits, recurse.
    if ((SrcBitSize&7) == 0 && (ByteStart+ByteSize)*8 <= SrcBitSize)
      return ExtractConstantBytes(CE->getOperand(0), ByteStart, ByteSize);

    // Otherwise, if extracting a subset of the input, which is not multiple of
    // 8 bits, do a shift and trunc to get the bits.
    if ((ByteStart+ByteSize)*8 < SrcBitSize) {
      assert((SrcBitSize&7) && "Shouldn't get byte sized case here");
      Constant *Res = CE->getOperand(0);
      if (ByteStart)
        Res = ConstantExpr::getLShr(Res,
                                 ConstantInt::get(Res->getType(), ByteStart*8));
      return ConstantExpr::getTrunc(Res, IntegerType::get(C->getContext(),
                                                          ByteSize*8));
    }

    // TODO: Handle the 'partially zero' case.
    return nullptr;
  }
  }
}

Constant *llvm::ConstantFoldCastInstruction(unsigned opc, Constant *V,
                                            Type *DestTy) {
  if (isa<PoisonValue>(V))
    return PoisonValue::get(DestTy);

  if (isa<UndefValue>(V)) {
    // zext(undef) = 0, because the top bits will be zero.
    // sext(undef) = 0, because the top bits will all be the same.
    // [us]itofp(undef) = 0, because the result value is bounded.
    if (opc == Instruction::ZExt || opc == Instruction::SExt ||
        opc == Instruction::UIToFP || opc == Instruction::SIToFP)
      return Constant::getNullValue(DestTy);
    return UndefValue::get(DestTy);
  }

  if (V->isNullValue() && !DestTy->isX86_MMXTy() && !DestTy->isX86_AMXTy() &&
      opc != Instruction::AddrSpaceCast)
    return Constant::getNullValue(DestTy);

  // If the cast operand is a constant expression, there's a few things we can
  // do to try to simplify it.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->isCast()) {
      // Try hard to fold cast of cast because they are often eliminable.
      if (unsigned newOpc = foldConstantCastPair(opc, CE, DestTy))
        return ConstantExpr::getCast(newOpc, CE->getOperand(0), DestTy);
    } else if (CE->getOpcode() == Instruction::GetElementPtr &&
               // Do not fold addrspacecast (gep 0, .., 0). It might make the
               // addrspacecast uncanonicalized.
               opc != Instruction::AddrSpaceCast &&
               // Do not fold bitcast (gep) with inrange index, as this loses
               // information.
               !cast<GEPOperator>(CE)->getInRangeIndex().hasValue() &&
               // Do not fold if the gep type is a vector, as bitcasting
               // operand 0 of a vector gep will result in a bitcast between
               // different sizes.
               !CE->getType()->isVectorTy()) {
      // If all of the indexes in the GEP are null values, there is no pointer
      // adjustment going on.  We might as well cast the source pointer.
      bool isAllNull = true;
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue()) {
          isAllNull = false;
          break;
        }
      if (isAllNull)
        // This is casting one pointer type to another, always BitCast
        return ConstantExpr::getPointerCast(CE->getOperand(0), DestTy);
    }
  }

  // If the cast operand is a constant vector, perform the cast by
  // operating on each element. In the cast of bitcasts, the element
  // count may be mismatched; don't attempt to handle that here.
  if ((isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) &&
      DestTy->isVectorTy() &&
      cast<FixedVectorType>(DestTy)->getNumElements() ==
          cast<FixedVectorType>(V->getType())->getNumElements()) {
    VectorType *DestVecTy = cast<VectorType>(DestTy);
    Type *DstEltTy = DestVecTy->getElementType();
    // Fast path for splatted constants.
    if (Constant *Splat = V->getSplatValue()) {
      return ConstantVector::getSplat(
          cast<VectorType>(DestTy)->getElementCount(),
          ConstantExpr::getCast(opc, Splat, DstEltTy));
    }
    SmallVector<Constant *, 16> res;
    Type *Ty = IntegerType::get(V->getContext(), 32);
    for (unsigned i = 0,
                  e = cast<FixedVectorType>(V->getType())->getNumElements();
         i != e; ++i) {
      Constant *C =
        ConstantExpr::getExtractElement(V, ConstantInt::get(Ty, i));
      res.push_back(ConstantExpr::getCast(opc, C, DstEltTy));
    }
    return ConstantVector::get(res);
  }

  // We actually have to do a cast now. Perform the cast according to the
  // opcode specified.
  switch (opc) {
  default:
    llvm_unreachable("Failed to cast constant expression");
  case Instruction::FPTrunc:
  case Instruction::FPExt:
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(V)) {
      bool ignored;
      APFloat Val = FPC->getValueAPF();
      Val.convert(DestTy->isHalfTy() ? APFloat::IEEEhalf() :
                  DestTy->isFloatTy() ? APFloat::IEEEsingle() :
                  DestTy->isDoubleTy() ? APFloat::IEEEdouble() :
                  DestTy->isX86_FP80Ty() ? APFloat::x87DoubleExtended() :
                  DestTy->isFP128Ty() ? APFloat::IEEEquad() :
                  DestTy->isPPC_FP128Ty() ? APFloat::PPCDoubleDouble() :
                  APFloat::Bogus(),
                  APFloat::rmNearestTiesToEven, &ignored);
      return ConstantFP::get(V->getContext(), Val);
    }
    return nullptr; // Can't fold.
  case Instruction::FPToUI:
  case Instruction::FPToSI:
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(V)) {
      const APFloat &V = FPC->getValueAPF();
      bool ignored;
      uint32_t DestBitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      APSInt IntVal(DestBitWidth, opc == Instruction::FPToUI);
      if (APFloat::opInvalidOp ==
          V.convertToInteger(IntVal, APFloat::rmTowardZero, &ignored)) {
        // Undefined behavior invoked - the destination type can't represent
        // the input constant.
        return PoisonValue::get(DestTy);
      }
      return ConstantInt::get(FPC->getContext(), IntVal);
    }
    return nullptr; // Can't fold.
  case Instruction::IntToPtr:   //always treated as unsigned
    if (V->isNullValue())       // Is it an integral null value?
      return ConstantPointerNull::get(cast<PointerType>(DestTy));
    return nullptr;                   // Other pointer types cannot be casted
  case Instruction::PtrToInt:   // always treated as unsigned
    // Is it a null pointer value?
    if (V->isNullValue())
      return ConstantInt::get(DestTy, 0);
    // Other pointer types cannot be casted
    return nullptr;
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      const APInt &api = CI->getValue();
      APFloat apf(DestTy->getFltSemantics(),
                  APInt::getZero(DestTy->getPrimitiveSizeInBits()));
      apf.convertFromAPInt(api, opc==Instruction::SIToFP,
                           APFloat::rmNearestTiesToEven);
      return ConstantFP::get(V->getContext(), apf);
    }
    return nullptr;
  case Instruction::ZExt:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      uint32_t BitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      return ConstantInt::get(V->getContext(),
                              CI->getValue().zext(BitWidth));
    }
    return nullptr;
  case Instruction::SExt:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      uint32_t BitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      return ConstantInt::get(V->getContext(),
                              CI->getValue().sext(BitWidth));
    }
    return nullptr;
  case Instruction::Trunc: {
    if (V->getType()->isVectorTy())
      return nullptr;

    uint32_t DestBitWidth = cast<IntegerType>(DestTy)->getBitWidth();
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      return ConstantInt::get(V->getContext(),
                              CI->getValue().trunc(DestBitWidth));
    }

    // The input must be a constantexpr.  See if we can simplify this based on
    // the bytes we are demanding.  Only do this if the source and dest are an
    // even multiple of a byte.
    if ((DestBitWidth & 7) == 0 &&
        (cast<IntegerType>(V->getType())->getBitWidth() & 7) == 0)
      if (Constant *Res = ExtractConstantBytes(V, 0, DestBitWidth / 8))
        return Res;

    return nullptr;
  }
  case Instruction::BitCast:
    return FoldBitCast(V, DestTy);
  case Instruction::AddrSpaceCast:
    return nullptr;
  }
}

Constant *llvm::ConstantFoldSelectInstruction(Constant *Cond,
                                              Constant *V1, Constant *V2) {
  // Check for i1 and vector true/false conditions.
  if (Cond->isNullValue()) return V2;
  if (Cond->isAllOnesValue()) return V1;

  // If the condition is a vector constant, fold the result elementwise.
  if (ConstantVector *CondV = dyn_cast<ConstantVector>(Cond)) {
    auto *V1VTy = CondV->getType();
    SmallVector<Constant*, 16> Result;
    Type *Ty = IntegerType::get(CondV->getContext(), 32);
    for (unsigned i = 0, e = V1VTy->getNumElements(); i != e; ++i) {
      Constant *V;
      Constant *V1Element = ConstantExpr::getExtractElement(V1,
                                                    ConstantInt::get(Ty, i));
      Constant *V2Element = ConstantExpr::getExtractElement(V2,
                                                    ConstantInt::get(Ty, i));
      auto *Cond = cast<Constant>(CondV->getOperand(i));
      if (isa<PoisonValue>(Cond)) {
        V = PoisonValue::get(V1Element->getType());
      } else if (V1Element == V2Element) {
        V = V1Element;
      } else if (isa<UndefValue>(Cond)) {
        V = isa<UndefValue>(V1Element) ? V1Element : V2Element;
      } else {
        if (!isa<ConstantInt>(Cond)) break;
        V = Cond->isNullValue() ? V2Element : V1Element;
      }
      Result.push_back(V);
    }

    // If we were able to build the vector, return it.
    if (Result.size() == V1VTy->getNumElements())
      return ConstantVector::get(Result);
  }

  if (isa<PoisonValue>(Cond))
    return PoisonValue::get(V1->getType());

  if (isa<UndefValue>(Cond)) {
    if (isa<UndefValue>(V1)) return V1;
    return V2;
  }

  if (V1 == V2) return V1;

  if (isa<PoisonValue>(V1))
    return V2;
  if (isa<PoisonValue>(V2))
    return V1;

  // If the true or false value is undef, we can fold to the other value as
  // long as the other value isn't poison.
  auto NotPoison = [](Constant *C) {
    if (isa<PoisonValue>(C))
      return false;

    // TODO: We can analyze ConstExpr by opcode to determine if there is any
    //       possibility of poison.
    if (isa<ConstantExpr>(C))
      return false;

    if (isa<ConstantInt>(C) || isa<GlobalVariable>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C) || isa<Function>(C))
      return true;

    if (C->getType()->isVectorTy())
      return !C->containsPoisonElement() && !C->containsConstantExpression();

    // TODO: Recursively analyze aggregates or other constants.
    return false;
  };
  if (isa<UndefValue>(V1) && NotPoison(V2)) return V2;
  if (isa<UndefValue>(V2) && NotPoison(V1)) return V1;

  if (ConstantExpr *TrueVal = dyn_cast<ConstantExpr>(V1)) {
    if (TrueVal->getOpcode() == Instruction::Select)
      if (TrueVal->getOperand(0) == Cond)
        return ConstantExpr::getSelect(Cond, TrueVal->getOperand(1), V2);
  }
  if (ConstantExpr *FalseVal = dyn_cast<ConstantExpr>(V2)) {
    if (FalseVal->getOpcode() == Instruction::Select)
      if (FalseVal->getOperand(0) == Cond)
        return ConstantExpr::getSelect(Cond, V1, FalseVal->getOperand(2));
  }

  return nullptr;
}

Constant *llvm::ConstantFoldExtractElementInstruction(Constant *Val,
                                                      Constant *Idx) {
  auto *ValVTy = cast<VectorType>(Val->getType());

  // extractelt poison, C -> poison
  // extractelt C, undef -> poison
  if (isa<PoisonValue>(Val) || isa<UndefValue>(Idx))
    return PoisonValue::get(ValVTy->getElementType());

  // extractelt undef, C -> undef
  if (isa<UndefValue>(Val))
    return UndefValue::get(ValVTy->getElementType());

  auto *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx)
    return nullptr;

  if (auto *ValFVTy = dyn_cast<FixedVectorType>(Val->getType())) {
    // ee({w,x,y,z}, wrong_value) -> poison
    if (CIdx->uge(ValFVTy->getNumElements()))
      return PoisonValue::get(ValFVTy->getElementType());
  }

  // ee (gep (ptr, idx0, ...), idx) -> gep (ee (ptr, idx), ee (idx0, idx), ...)
  if (auto *CE = dyn_cast<ConstantExpr>(Val)) {
    if (auto *GEP = dyn_cast<GEPOperator>(CE)) {
      SmallVector<Constant *, 8> Ops;
      Ops.reserve(CE->getNumOperands());
      for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i) {
        Constant *Op = CE->getOperand(i);
        if (Op->getType()->isVectorTy()) {
          Constant *ScalarOp = ConstantExpr::getExtractElement(Op, Idx);
          if (!ScalarOp)
            return nullptr;
          Ops.push_back(ScalarOp);
        } else
          Ops.push_back(Op);
      }
      return CE->getWithOperands(Ops, ValVTy->getElementType(), false,
                                 GEP->getSourceElementType());
    } else if (CE->getOpcode() == Instruction::InsertElement) {
      if (const auto *IEIdx = dyn_cast<ConstantInt>(CE->getOperand(2))) {
        if (APSInt::isSameValue(APSInt(IEIdx->getValue()),
                                APSInt(CIdx->getValue()))) {
          return CE->getOperand(1);
        } else {
          return ConstantExpr::getExtractElement(CE->getOperand(0), CIdx);
        }
      }
    }
  }

  if (Constant *C = Val->getAggregateElement(CIdx))
    return C;

  // Lane < Splat minimum vector width => extractelt Splat(x), Lane -> x
  if (CIdx->getValue().ult(ValVTy->getElementCount().getKnownMinValue())) {
    if (Constant *SplatVal = Val->getSplatValue())
      return SplatVal;
  }

  return nullptr;
}

Constant *llvm::ConstantFoldInsertElementInstruction(Constant *Val,
                                                     Constant *Elt,
                                                     Constant *Idx) {
  if (isa<UndefValue>(Idx))
    return PoisonValue::get(Val->getType());

  // Inserting null into all zeros is still all zeros.
  // TODO: This is true for undef and poison splats too.
  if (isa<ConstantAggregateZero>(Val) && Elt->isNullValue())
    return Val;

  ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx) return nullptr;

  // Do not iterate on scalable vector. The num of elements is unknown at
  // compile-time.
  if (isa<ScalableVectorType>(Val->getType()))
    return nullptr;

  auto *ValTy = cast<FixedVectorType>(Val->getType());

  unsigned NumElts = ValTy->getNumElements();
  if (CIdx->uge(NumElts))
    return PoisonValue::get(Val->getType());

  SmallVector<Constant*, 16> Result;
  Result.reserve(NumElts);
  auto *Ty = Type::getInt32Ty(Val->getContext());
  uint64_t IdxVal = CIdx->getZExtValue();
  for (unsigned i = 0; i != NumElts; ++i) {
    if (i == IdxVal) {
      Result.push_back(Elt);
      continue;
    }

    Constant *C = ConstantExpr::getExtractElement(Val, ConstantInt::get(Ty, i));
    Result.push_back(C);
  }

  return ConstantVector::get(Result);
}

Constant *llvm::ConstantFoldShuffleVectorInstruction(Constant *V1, Constant *V2,
                                                     ArrayRef<int> Mask) {
  auto *V1VTy = cast<VectorType>(V1->getType());
  unsigned MaskNumElts = Mask.size();
  auto MaskEltCount =
      ElementCount::get(MaskNumElts, isa<ScalableVectorType>(V1VTy));
  Type *EltTy = V1VTy->getElementType();

  // Undefined shuffle mask -> undefined value.
  if (all_of(Mask, [](int Elt) { return Elt == UndefMaskElem; })) {
    return UndefValue::get(VectorType::get(EltTy, MaskEltCount));
  }

  // If the mask is all zeros this is a splat, no need to go through all
  // elements.
  if (all_of(Mask, [](int Elt) { return Elt == 0; })) {
    Type *Ty = IntegerType::get(V1->getContext(), 32);
    Constant *Elt =
        ConstantExpr::getExtractElement(V1, ConstantInt::get(Ty, 0));

    if (Elt->isNullValue()) {
      auto *VTy = VectorType::get(EltTy, MaskEltCount);
      return ConstantAggregateZero::get(VTy);
    } else if (!MaskEltCount.isScalable())
      return ConstantVector::getSplat(MaskEltCount, Elt);
  }
  // Do not iterate on scalable vector. The num of elements is unknown at
  // compile-time.
  if (isa<ScalableVectorType>(V1VTy))
    return nullptr;

  unsigned SrcNumElts = V1VTy->getElementCount().getKnownMinValue();

  // Loop over the shuffle mask, evaluating each element.
  SmallVector<Constant*, 32> Result;
  for (unsigned i = 0; i != MaskNumElts; ++i) {
    int Elt = Mask[i];
    if (Elt == -1) {
      Result.push_back(UndefValue::get(EltTy));
      continue;
    }
    Constant *InElt;
    if (unsigned(Elt) >= SrcNumElts*2)
      InElt = UndefValue::get(EltTy);
    else if (unsigned(Elt) >= SrcNumElts) {
      Type *Ty = IntegerType::get(V2->getContext(), 32);
      InElt =
        ConstantExpr::getExtractElement(V2,
                                        ConstantInt::get(Ty, Elt - SrcNumElts));
    } else {
      Type *Ty = IntegerType::get(V1->getContext(), 32);
      InElt = ConstantExpr::getExtractElement(V1, ConstantInt::get(Ty, Elt));
    }
    Result.push_back(InElt);
  }

  return ConstantVector::get(Result);
}

Constant *llvm::ConstantFoldExtractValueInstruction(Constant *Agg,
                                                    ArrayRef<unsigned> Idxs) {
  // Base case: no indices, so return the entire value.
  if (Idxs.empty())
    return Agg;

  if (Constant *C = Agg->getAggregateElement(Idxs[0]))
    return ConstantFoldExtractValueInstruction(C, Idxs.slice(1));

  return nullptr;
}

Constant *llvm::ConstantFoldInsertValueInstruction(Constant *Agg,
                                                   Constant *Val,
                                                   ArrayRef<unsigned> Idxs) {
  // Base case: no indices, so replace the entire value.
  if (Idxs.empty())
    return Val;

  unsigned NumElts;
  if (StructType *ST = dyn_cast<StructType>(Agg->getType()))
    NumElts = ST->getNumElements();
  else
    NumElts = cast<ArrayType>(Agg->getType())->getNumElements();

  SmallVector<Constant*, 32> Result;
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *C = Agg->getAggregateElement(i);
    if (!C) return nullptr;

    if (Idxs[0] == i)
      C = ConstantFoldInsertValueInstruction(C, Val, Idxs.slice(1));

    Result.push_back(C);
  }

  if (StructType *ST = dyn_cast<StructType>(Agg->getType()))
    return ConstantStruct::get(ST, Result);
  return ConstantArray::get(cast<ArrayType>(Agg->getType()), Result);
}

Constant *llvm::ConstantFoldUnaryInstruction(unsigned Opcode, Constant *C) {
  assert(Instruction::isUnaryOp(Opcode) && "Non-unary instruction detected");

  // Handle scalar UndefValue and scalable vector UndefValue. Fixed-length
  // vectors are always evaluated per element.
  bool IsScalableVector = isa<ScalableVectorType>(C->getType());
  bool HasScalarUndefOrScalableVectorUndef =
      (!C->getType()->isVectorTy() || IsScalableVector) && isa<UndefValue>(C);

  if (HasScalarUndefOrScalableVectorUndef) {
    switch (static_cast<Instruction::UnaryOps>(Opcode)) {
    case Instruction::FNeg:
      return C; // -undef -> undef
    case Instruction::UnaryOpsEnd:
      llvm_unreachable("Invalid UnaryOp");
    }
  }

  // Constant should not be UndefValue, unless these are vector constants.
  assert(!HasScalarUndefOrScalableVectorUndef && "Unexpected UndefValue");
  // We only have FP UnaryOps right now.
  assert(!isa<ConstantInt>(C) && "Unexpected Integer UnaryOp");

  if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    const APFloat &CV = CFP->getValueAPF();
    switch (Opcode) {
    default:
      break;
    case Instruction::FNeg:
      return ConstantFP::get(C->getContext(), neg(CV));
    }
  } else if (auto *VTy = dyn_cast<FixedVectorType>(C->getType())) {

    Type *Ty = IntegerType::get(VTy->getContext(), 32);
    // Fast path for splatted constants.
    if (Constant *Splat = C->getSplatValue()) {
      Constant *Elt = ConstantExpr::get(Opcode, Splat);
      return ConstantVector::getSplat(VTy->getElementCount(), Elt);
    }

    // Fold each element and create a vector constant from those constants.
    SmallVector<Constant *, 16> Result;
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
      Constant *ExtractIdx = ConstantInt::get(Ty, i);
      Constant *Elt = ConstantExpr::getExtractElement(C, ExtractIdx);

      Result.push_back(ConstantExpr::get(Opcode, Elt));
    }

    return ConstantVector::get(Result);
  }

  // We don't know how to fold this.
  return nullptr;
}

Constant *llvm::ConstantFoldBinaryInstruction(unsigned Opcode, Constant *C1,
                                              Constant *C2) {
  assert(Instruction::isBinaryOp(Opcode) && "Non-binary instruction detected");

  // Simplify BinOps with their identity values first. They are no-ops and we
  // can always return the other value, including undef or poison values.
  // FIXME: remove unnecessary duplicated identity patterns below.
  // FIXME: Use AllowRHSConstant with getBinOpIdentity to handle additional ops,
  //        like X << 0 = X.
  Constant *Identity = ConstantExpr::getBinOpIdentity(Opcode, C1->getType());
  if (Identity) {
    if (C1 == Identity)
      return C2;
    if (C2 == Identity)
      return C1;
  }

  // Binary operations propagate poison.
  if (isa<PoisonValue>(C1) || isa<PoisonValue>(C2))
    return PoisonValue::get(C1->getType());

  // Handle scalar UndefValue and scalable vector UndefValue. Fixed-length
  // vectors are always evaluated per element.
  bool IsScalableVector = isa<ScalableVectorType>(C1->getType());
  bool HasScalarUndefOrScalableVectorUndef =
      (!C1->getType()->isVectorTy() || IsScalableVector) &&
      (isa<UndefValue>(C1) || isa<UndefValue>(C2));
  if (HasScalarUndefOrScalableVectorUndef) {
    switch (static_cast<Instruction::BinaryOps>(Opcode)) {
    case Instruction::Xor:
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2))
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return Constant::getNullValue(C1->getType());
      LLVM_FALLTHROUGH;
    case Instruction::Add:
    case Instruction::Sub:
      return UndefValue::get(C1->getType());
    case Instruction::And:
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2)) // undef & undef -> undef
        return C1;
      return Constant::getNullValue(C1->getType());   // undef & X -> 0
    case Instruction::Mul: {
      // undef * undef -> undef
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2))
        return C1;
      const APInt *CV;
      // X * undef -> undef   if X is odd
      if (match(C1, m_APInt(CV)) || match(C2, m_APInt(CV)))
        if ((*CV)[0])
          return UndefValue::get(C1->getType());

      // X * undef -> 0       otherwise
      return Constant::getNullValue(C1->getType());
    }
    case Instruction::SDiv:
    case Instruction::UDiv:
      // X / undef -> poison
      // X / 0 -> poison
      if (match(C2, m_CombineOr(m_Undef(), m_Zero())))
        return PoisonValue::get(C2->getType());
      // undef / 1 -> undef
      if (match(C2, m_One()))
        return C1;
      // undef / X -> 0       otherwise
      return Constant::getNullValue(C1->getType());
    case Instruction::URem:
    case Instruction::SRem:
      // X % undef -> poison
      // X % 0 -> poison
      if (match(C2, m_CombineOr(m_Undef(), m_Zero())))
        return PoisonValue::get(C2->getType());
      // undef % X -> 0       otherwise
      return Constant::getNullValue(C1->getType());
    case Instruction::Or:                          // X | undef -> -1
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2)) // undef | undef -> undef
        return C1;
      return Constant::getAllOnesValue(C1->getType()); // undef | X -> ~0
    case Instruction::LShr:
      // X >>l undef -> poison
      if (isa<UndefValue>(C2))
        return PoisonValue::get(C2->getType());
      // undef >>l 0 -> undef
      if (match(C2, m_Zero()))
        return C1;
      // undef >>l X -> 0
      return Constant::getNullValue(C1->getType());
    case Instruction::AShr:
      // X >>a undef -> poison
      if (isa<UndefValue>(C2))
        return PoisonValue::get(C2->getType());
      // undef >>a 0 -> undef
      if (match(C2, m_Zero()))
        return C1;
      // TODO: undef >>a X -> poison if the shift is exact
      // undef >>a X -> 0
      return Constant::getNullValue(C1->getType());
    case Instruction::Shl:
      // X << undef -> undef
      if (isa<UndefValue>(C2))
        return PoisonValue::get(C2->getType());
      // undef << 0 -> undef
      if (match(C2, m_Zero()))
        return C1;
      // undef << X -> 0
      return Constant::getNullValue(C1->getType());
    case Instruction::FSub:
      // -0.0 - undef --> undef (consistent with "fneg undef")
      if (match(C1, m_NegZeroFP()) && isa<UndefValue>(C2))
        return C2;
      LLVM_FALLTHROUGH;
    case Instruction::FAdd:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::FRem:
      // [any flop] undef, undef -> undef
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2))
        return C1;
      // [any flop] C, undef -> NaN
      // [any flop] undef, C -> NaN
      // We could potentially specialize NaN/Inf constants vs. 'normal'
      // constants (possibly differently depending on opcode and operand). This
      // would allow returning undef sometimes. But it is always safe to fold to
      // NaN because we can choose the undef operand as NaN, and any FP opcode
      // with a NaN operand will propagate NaN.
      return ConstantFP::getNaN(C1->getType());
    case Instruction::BinaryOpsEnd:
      llvm_unreachable("Invalid BinaryOp");
    }
  }

  // Neither constant should be UndefValue, unless these are vector constants.
  assert((!HasScalarUndefOrScalableVectorUndef) && "Unexpected UndefValue");

  // Handle simplifications when the RHS is a constant int.
  if (ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
    switch (Opcode) {
    case Instruction::Add:
      if (CI2->isZero()) return C1;                             // X + 0 == X
      break;
    case Instruction::Sub:
      if (CI2->isZero()) return C1;                             // X - 0 == X
      break;
    case Instruction::Mul:
      if (CI2->isZero()) return C2;                             // X * 0 == 0
      if (CI2->isOne())
        return C1;                                              // X * 1 == X
      break;
    case Instruction::UDiv:
    case Instruction::SDiv:
      if (CI2->isOne())
        return C1;                                            // X / 1 == X
      if (CI2->isZero())
        return PoisonValue::get(CI2->getType());              // X / 0 == poison
      break;
    case Instruction::URem:
    case Instruction::SRem:
      if (CI2->isOne())
        return Constant::getNullValue(CI2->getType());        // X % 1 == 0
      if (CI2->isZero())
        return PoisonValue::get(CI2->getType());              // X % 0 == poison
      break;
    case Instruction::And:
      if (CI2->isZero()) return C2;                           // X & 0 == 0
      if (CI2->isMinusOne())
        return C1;                                            // X & -1 == X

      if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
        // (zext i32 to i64) & 4294967295 -> (zext i32 to i64)
        if (CE1->getOpcode() == Instruction::ZExt) {
          unsigned DstWidth = CI2->getType()->getBitWidth();
          unsigned SrcWidth =
            CE1->getOperand(0)->getType()->getPrimitiveSizeInBits();
          APInt PossiblySetBits(APInt::getLowBitsSet(DstWidth, SrcWidth));
          if ((PossiblySetBits & CI2->getValue()) == PossiblySetBits)
            return C1;
        }

        // If and'ing the address of a global with a constant, fold it.
        if (CE1->getOpcode() == Instruction::PtrToInt &&
            isa<GlobalValue>(CE1->getOperand(0))) {
          GlobalValue *GV = cast<GlobalValue>(CE1->getOperand(0));

          MaybeAlign GVAlign;

          if (Module *TheModule = GV->getParent()) {
            const DataLayout &DL = TheModule->getDataLayout();
            GVAlign = GV->getPointerAlignment(DL);

            // If the function alignment is not specified then assume that it
            // is 4.
            // This is dangerous; on x86, the alignment of the pointer
            // corresponds to the alignment of the function, but might be less
            // than 4 if it isn't explicitly specified.
            // However, a fix for this behaviour was reverted because it
            // increased code size (see https://reviews.llvm.org/D55115)
            // FIXME: This code should be deleted once existing targets have
            // appropriate defaults
            if (isa<Function>(GV) && !DL.getFunctionPtrAlign())
              GVAlign = Align(4);
          } else if (isa<Function>(GV)) {
            // Without a datalayout we have to assume the worst case: that the
            // function pointer isn't aligned at all.
            GVAlign = llvm::None;
          } else if (isa<GlobalVariable>(GV)) {
            GVAlign = cast<GlobalVariable>(GV)->getAlign();
          }

          if (GVAlign && *GVAlign > 1) {
            unsigned DstWidth = CI2->getType()->getBitWidth();
            unsigned SrcWidth = std::min(DstWidth, Log2(*GVAlign));
            APInt BitsNotSet(APInt::getLowBitsSet(DstWidth, SrcWidth));

            // If checking bits we know are clear, return zero.
            if ((CI2->getValue() & BitsNotSet) == CI2->getValue())
              return Constant::getNullValue(CI2->getType());
          }
        }
      }
      break;
    case Instruction::Or:
      if (CI2->isZero()) return C1;        // X | 0 == X
      if (CI2->isMinusOne())
        return C2;                         // X | -1 == -1
      break;
    case Instruction::Xor:
      if (CI2->isZero()) return C1;        // X ^ 0 == X

      if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
        switch (CE1->getOpcode()) {
        default: break;
        case Instruction::ICmp:
        case Instruction::FCmp:
          // cmp pred ^ true -> cmp !pred
          assert(CI2->isOne());
          CmpInst::Predicate pred = (CmpInst::Predicate)CE1->getPredicate();
          pred = CmpInst::getInversePredicate(pred);
          return ConstantExpr::getCompare(pred, CE1->getOperand(0),
                                          CE1->getOperand(1));
        }
      }
      break;
    case Instruction::AShr:
      // ashr (zext C to Ty), C2 -> lshr (zext C, CSA), C2
      if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1))
        if (CE1->getOpcode() == Instruction::ZExt)  // Top bits known zero.
          return ConstantExpr::getLShr(C1, C2);
      break;
    }
  } else if (isa<ConstantInt>(C1)) {
    // If C1 is a ConstantInt and C2 is not, swap the operands.
    if (Instruction::isCommutative(Opcode))
      return ConstantExpr::get(Opcode, C2, C1);
  }

  if (ConstantInt *CI1 = dyn_cast<ConstantInt>(C1)) {
    if (ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
      const APInt &C1V = CI1->getValue();
      const APInt &C2V = CI2->getValue();
      switch (Opcode) {
      default:
        break;
      case Instruction::Add:
        return ConstantInt::get(CI1->getContext(), C1V + C2V);
      case Instruction::Sub:
        return ConstantInt::get(CI1->getContext(), C1V - C2V);
      case Instruction::Mul:
        return ConstantInt::get(CI1->getContext(), C1V * C2V);
      case Instruction::UDiv:
        assert(!CI2->isZero() && "Div by zero handled above");
        return ConstantInt::get(CI1->getContext(), C1V.udiv(C2V));
      case Instruction::SDiv:
        assert(!CI2->isZero() && "Div by zero handled above");
        if (C2V.isAllOnes() && C1V.isMinSignedValue())
          return PoisonValue::get(CI1->getType());   // MIN_INT / -1 -> poison
        return ConstantInt::get(CI1->getContext(), C1V.sdiv(C2V));
      case Instruction::URem:
        assert(!CI2->isZero() && "Div by zero handled above");
        return ConstantInt::get(CI1->getContext(), C1V.urem(C2V));
      case Instruction::SRem:
        assert(!CI2->isZero() && "Div by zero handled above");
        if (C2V.isAllOnes() && C1V.isMinSignedValue())
          return PoisonValue::get(CI1->getType());   // MIN_INT % -1 -> poison
        return ConstantInt::get(CI1->getContext(), C1V.srem(C2V));
      case Instruction::And:
        return ConstantInt::get(CI1->getContext(), C1V & C2V);
      case Instruction::Or:
        return ConstantInt::get(CI1->getContext(), C1V | C2V);
      case Instruction::Xor:
        return ConstantInt::get(CI1->getContext(), C1V ^ C2V);
      case Instruction::Shl:
        if (C2V.ult(C1V.getBitWidth()))
          return ConstantInt::get(CI1->getContext(), C1V.shl(C2V));
        return PoisonValue::get(C1->getType()); // too big shift is poison
      case Instruction::LShr:
        if (C2V.ult(C1V.getBitWidth()))
          return ConstantInt::get(CI1->getContext(), C1V.lshr(C2V));
        return PoisonValue::get(C1->getType()); // too big shift is poison
      case Instruction::AShr:
        if (C2V.ult(C1V.getBitWidth()))
          return ConstantInt::get(CI1->getContext(), C1V.ashr(C2V));
        return PoisonValue::get(C1->getType()); // too big shift is poison
      }
    }

    switch (Opcode) {
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Shl:
      if (CI1->isZero()) return C1;
      break;
    default:
      break;
    }
  } else if (ConstantFP *CFP1 = dyn_cast<ConstantFP>(C1)) {
    if (ConstantFP *CFP2 = dyn_cast<ConstantFP>(C2)) {
      const APFloat &C1V = CFP1->getValueAPF();
      const APFloat &C2V = CFP2->getValueAPF();
      APFloat C3V = C1V;  // copy for modification
      switch (Opcode) {
      default:
        break;
      case Instruction::FAdd:
        (void)C3V.add(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(C1->getContext(), C3V);
      case Instruction::FSub:
        (void)C3V.subtract(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(C1->getContext(), C3V);
      case Instruction::FMul:
        (void)C3V.multiply(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(C1->getContext(), C3V);
      case Instruction::FDiv:
        (void)C3V.divide(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(C1->getContext(), C3V);
      case Instruction::FRem:
        (void)C3V.mod(C2V);
        return ConstantFP::get(C1->getContext(), C3V);
      }
    }
  } else if (auto *VTy = dyn_cast<VectorType>(C1->getType())) {
    // Fast path for splatted constants.
    if (Constant *C2Splat = C2->getSplatValue()) {
      if (Instruction::isIntDivRem(Opcode) && C2Splat->isNullValue())
        return PoisonValue::get(VTy);
      if (Constant *C1Splat = C1->getSplatValue()) {
        return ConstantVector::getSplat(
            VTy->getElementCount(),
            ConstantExpr::get(Opcode, C1Splat, C2Splat));
      }
    }

    if (auto *FVTy = dyn_cast<FixedVectorType>(VTy)) {
      // Fold each element and create a vector constant from those constants.
      SmallVector<Constant*, 16> Result;
      Type *Ty = IntegerType::get(FVTy->getContext(), 32);
      for (unsigned i = 0, e = FVTy->getNumElements(); i != e; ++i) {
        Constant *ExtractIdx = ConstantInt::get(Ty, i);
        Constant *LHS = ConstantExpr::getExtractElement(C1, ExtractIdx);
        Constant *RHS = ConstantExpr::getExtractElement(C2, ExtractIdx);

        // If any element of a divisor vector is zero, the whole op is poison.
        if (Instruction::isIntDivRem(Opcode) && RHS->isNullValue())
          return PoisonValue::get(VTy);

        Result.push_back(ConstantExpr::get(Opcode, LHS, RHS));
      }

      return ConstantVector::get(Result);
    }
  }

  if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
    // There are many possible foldings we could do here.  We should probably
    // at least fold add of a pointer with an integer into the appropriate
    // getelementptr.  This will improve alias analysis a bit.

    // Given ((a + b) + c), if (b + c) folds to something interesting, return
    // (a + (b + c)).
    if (Instruction::isAssociative(Opcode) && CE1->getOpcode() == Opcode) {
      Constant *T = ConstantExpr::get(Opcode, CE1->getOperand(1), C2);
      if (!isa<ConstantExpr>(T) || cast<ConstantExpr>(T)->getOpcode() != Opcode)
        return ConstantExpr::get(Opcode, CE1->getOperand(0), T);
    }
  } else if (isa<ConstantExpr>(C2)) {
    // If C2 is a constant expr and C1 isn't, flop them around and fold the
    // other way if possible.
    if (Instruction::isCommutative(Opcode))
      return ConstantFoldBinaryInstruction(Opcode, C2, C1);
  }

  // i1 can be simplified in many cases.
  if (C1->getType()->isIntegerTy(1)) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
      return ConstantExpr::getXor(C1, C2);
    case Instruction::Mul:
      return ConstantExpr::getAnd(C1, C2);
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
      // We can assume that C2 == 0.  If it were one the result would be
      // undefined because the shift value is as large as the bitwidth.
      return C1;
    case Instruction::SDiv:
    case Instruction::UDiv:
      // We can assume that C2 == 1.  If it were zero the result would be
      // undefined through division by zero.
      return C1;
    case Instruction::URem:
    case Instruction::SRem:
      // We can assume that C2 == 1.  If it were zero the result would be
      // undefined through division by zero.
      return ConstantInt::getFalse(C1->getContext());
    default:
      break;
    }
  }

  // We don't know how to fold this.
  return nullptr;
}

/// This function determines if there is anything we can decide about the two
/// constants provided. This doesn't need to handle simple things like
/// ConstantFP comparisons, but should instead handle ConstantExprs.
/// If we can determine that the two constants have a particular relation to
/// each other, we should return the corresponding FCmpInst predicate,
/// otherwise return FCmpInst::BAD_FCMP_PREDICATE. This is used below in
/// ConstantFoldCompareInstruction.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider ConstantFP
/// to be the simplest, and ConstantExprs to be the most complex.
static FCmpInst::Predicate evaluateFCmpRelation(Constant *V1, Constant *V2) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare values of different types!");

  // We do not know if a constant expression will evaluate to a number or NaN.
  // Therefore, we can only say that the relation is unordered or equal.
  if (V1 == V2) return FCmpInst::FCMP_UEQ;

  if (!isa<ConstantExpr>(V1)) {
    if (!isa<ConstantExpr>(V2)) {
      // Simple case, use the standard constant folder.
      ConstantInt *R = nullptr;
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OEQ, V1, V2));
      if (R && !R->isZero())
        return FCmpInst::FCMP_OEQ;
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OLT, V1, V2));
      if (R && !R->isZero())
        return FCmpInst::FCMP_OLT;
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OGT, V1, V2));
      if (R && !R->isZero())
        return FCmpInst::FCMP_OGT;

      // Nothing more we can do
      return FCmpInst::BAD_FCMP_PREDICATE;
    }

    // If the first operand is simple and second is ConstantExpr, swap operands.
    FCmpInst::Predicate SwappedRelation = evaluateFCmpRelation(V2, V1);
    if (SwappedRelation != FCmpInst::BAD_FCMP_PREDICATE)
      return FCmpInst::getSwappedPredicate(SwappedRelation);
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr or a simple constant.
    ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    switch (CE1->getOpcode()) {
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
      // We might be able to do something with these but we don't right now.
      break;
    default:
      break;
    }
  }
  // There are MANY other foldings that we could perform here.  They will
  // probably be added on demand, as they seem needed.
  return FCmpInst::BAD_FCMP_PREDICATE;
}

static ICmpInst::Predicate areGlobalsPotentiallyEqual(const GlobalValue *GV1,
                                                      const GlobalValue *GV2) {
  auto isGlobalUnsafeForEquality = [](const GlobalValue *GV) {
    if (GV->isInterposable() || GV->hasGlobalUnnamedAddr())
      return true;
    if (const auto *GVar = dyn_cast<GlobalVariable>(GV)) {
      Type *Ty = GVar->getValueType();
      // A global with opaque type might end up being zero sized.
      if (!Ty->isSized())
        return true;
      // A global with an empty type might lie at the address of any other
      // global.
      if (Ty->isEmptyTy())
        return true;
    }
    return false;
  };
  // Don't try to decide equality of aliases.
  if (!isa<GlobalAlias>(GV1) && !isa<GlobalAlias>(GV2))
    if (!isGlobalUnsafeForEquality(GV1) && !isGlobalUnsafeForEquality(GV2))
      return ICmpInst::ICMP_NE;
  return ICmpInst::BAD_ICMP_PREDICATE;
}

/// This function determines if there is anything we can decide about the two
/// constants provided. This doesn't need to handle simple things like integer
/// comparisons, but should instead handle ConstantExprs and GlobalValues.
/// If we can determine that the two constants have a particular relation to
/// each other, we should return the corresponding ICmp predicate, otherwise
/// return ICmpInst::BAD_ICMP_PREDICATE.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider simple
/// constants (like ConstantInt) to be the simplest, followed by
/// GlobalValues, followed by ConstantExpr's (the most complex).
///
static ICmpInst::Predicate evaluateICmpRelation(Constant *V1, Constant *V2,
                                                bool isSigned) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare different types of values!");
  if (V1 == V2) return ICmpInst::ICMP_EQ;

  if (!isa<ConstantExpr>(V1) && !isa<GlobalValue>(V1) &&
      !isa<BlockAddress>(V1)) {
    if (!isa<GlobalValue>(V2) && !isa<ConstantExpr>(V2) &&
        !isa<BlockAddress>(V2)) {
      // We distilled this down to a simple case, use the standard constant
      // folder.
      ConstantInt *R = nullptr;
      ICmpInst::Predicate pred = ICmpInst::ICMP_EQ;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, V1, V2));
      if (R && !R->isZero())
        return pred;
      pred = isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, V1, V2));
      if (R && !R->isZero())
        return pred;
      pred = isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, V1, V2));
      if (R && !R->isZero())
        return pred;

      // If we couldn't figure it out, bail.
      return ICmpInst::BAD_ICMP_PREDICATE;
    }

    // If the first operand is simple, swap operands.
    ICmpInst::Predicate SwappedRelation =
      evaluateICmpRelation(V2, V1, isSigned);
    if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
      return ICmpInst::getSwappedPredicate(SwappedRelation);

  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(V1)) {
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
      ICmpInst::Predicate SwappedRelation =
        evaluateICmpRelation(V2, V1, isSigned);
      if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
        return ICmpInst::getSwappedPredicate(SwappedRelation);
      return ICmpInst::BAD_ICMP_PREDICATE;
    }

    // Now we know that the RHS is a GlobalValue, BlockAddress or simple
    // constant (which, since the types must match, means that it's a
    // ConstantPointerNull).
    if (const GlobalValue *GV2 = dyn_cast<GlobalValue>(V2)) {
      return areGlobalsPotentiallyEqual(GV, GV2);
    } else if (isa<BlockAddress>(V2)) {
      return ICmpInst::ICMP_NE; // Globals never equal labels.
    } else {
      assert(isa<ConstantPointerNull>(V2) && "Canonicalization guarantee!");
      // GlobalVals can never be null unless they have external weak linkage.
      // We don't try to evaluate aliases here.
      // NOTE: We should not be doing this constant folding if null pointer
      // is considered valid for the function. But currently there is no way to
      // query it from the Constant type.
      if (!GV->hasExternalWeakLinkage() && !isa<GlobalAlias>(GV) &&
          !NullPointerIsDefined(nullptr /* F */,
                                GV->getType()->getAddressSpace()))
        return ICmpInst::ICMP_UGT;
    }
  } else if (const BlockAddress *BA = dyn_cast<BlockAddress>(V1)) {
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
      ICmpInst::Predicate SwappedRelation =
        evaluateICmpRelation(V2, V1, isSigned);
      if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
        return ICmpInst::getSwappedPredicate(SwappedRelation);
      return ICmpInst::BAD_ICMP_PREDICATE;
    }

    // Now we know that the RHS is a GlobalValue, BlockAddress or simple
    // constant (which, since the types must match, means that it is a
    // ConstantPointerNull).
    if (const BlockAddress *BA2 = dyn_cast<BlockAddress>(V2)) {
      // Block address in another function can't equal this one, but block
      // addresses in the current function might be the same if blocks are
      // empty.
      if (BA2->getFunction() != BA->getFunction())
        return ICmpInst::ICMP_NE;
    } else {
      // Block addresses aren't null, don't equal the address of globals.
      assert((isa<ConstantPointerNull>(V2) || isa<GlobalValue>(V2)) &&
             "Canonicalization guarantee!");
      return ICmpInst::ICMP_NE;
    }
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr, a global, block address, or a simple constant.
    ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    Constant *CE1Op0 = CE1->getOperand(0);

    switch (CE1->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      break; // We can't evaluate floating point casts or truncations.

    case Instruction::BitCast:
      // If this is a global value cast, check to see if the RHS is also a
      // GlobalValue.
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(CE1Op0))
        if (const GlobalValue *GV2 = dyn_cast<GlobalValue>(V2))
          return areGlobalsPotentiallyEqual(GV, GV2);
      LLVM_FALLTHROUGH;
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::ZExt:
    case Instruction::SExt:
      // We can't evaluate floating point casts or truncations.
      if (CE1Op0->getType()->isFPOrFPVectorTy())
        break;

      // If the cast is not actually changing bits, and the second operand is a
      // null pointer, do the comparison with the pre-casted value.
      if (V2->isNullValue() && CE1->getType()->isIntOrPtrTy()) {
        if (CE1->getOpcode() == Instruction::ZExt) isSigned = false;
        if (CE1->getOpcode() == Instruction::SExt) isSigned = true;
        return evaluateICmpRelation(CE1Op0,
                                    Constant::getNullValue(CE1Op0->getType()),
                                    isSigned);
      }
      break;

    case Instruction::GetElementPtr: {
      GEPOperator *CE1GEP = cast<GEPOperator>(CE1);
      // Ok, since this is a getelementptr, we know that the constant has a
      // pointer type.  Check the various cases.
      if (isa<ConstantPointerNull>(V2)) {
        // If we are comparing a GEP to a null pointer, check to see if the base
        // of the GEP equals the null pointer.
        if (const GlobalValue *GV = dyn_cast<GlobalValue>(CE1Op0)) {
          // If its not weak linkage, the GVal must have a non-zero address
          // so the result is greater-than
          if (!GV->hasExternalWeakLinkage() && CE1GEP->isInBounds())
            return ICmpInst::ICMP_UGT;
        }
      } else if (const GlobalValue *GV2 = dyn_cast<GlobalValue>(V2)) {
        if (const GlobalValue *GV = dyn_cast<GlobalValue>(CE1Op0)) {
          if (GV != GV2) {
            if (CE1GEP->hasAllZeroIndices())
              return areGlobalsPotentiallyEqual(GV, GV2);
            return ICmpInst::BAD_ICMP_PREDICATE;
          }
        }
      } else if (const auto *CE2GEP = dyn_cast<GEPOperator>(V2)) {
        // By far the most common case to handle is when the base pointers are
        // obviously to the same global.
        const Constant *CE2Op0 = cast<Constant>(CE2GEP->getPointerOperand());
        if (isa<GlobalValue>(CE1Op0) && isa<GlobalValue>(CE2Op0)) {
          // Don't know relative ordering, but check for inequality.
          if (CE1Op0 != CE2Op0) {
            if (CE1GEP->hasAllZeroIndices() && CE2GEP->hasAllZeroIndices())
              return areGlobalsPotentiallyEqual(cast<GlobalValue>(CE1Op0),
                                                cast<GlobalValue>(CE2Op0));
            return ICmpInst::BAD_ICMP_PREDICATE;
          }
        }
      }
      break;
    }
    default:
      break;
    }
  }

  return ICmpInst::BAD_ICMP_PREDICATE;
}

Constant *llvm::ConstantFoldCompareInstruction(CmpInst::Predicate Predicate,
                                               Constant *C1, Constant *C2) {
  Type *ResultTy;
  if (VectorType *VT = dyn_cast<VectorType>(C1->getType()))
    ResultTy = VectorType::get(Type::getInt1Ty(C1->getContext()),
                               VT->getElementCount());
  else
    ResultTy = Type::getInt1Ty(C1->getContext());

  // Fold FCMP_FALSE/FCMP_TRUE unconditionally.
  if (Predicate == FCmpInst::FCMP_FALSE)
    return Constant::getNullValue(ResultTy);

  if (Predicate == FCmpInst::FCMP_TRUE)
    return Constant::getAllOnesValue(ResultTy);

  // Handle some degenerate cases first
  if (isa<PoisonValue>(C1) || isa<PoisonValue>(C2))
    return PoisonValue::get(ResultTy);

  if (isa<UndefValue>(C1) || isa<UndefValue>(C2)) {
    bool isIntegerPredicate = ICmpInst::isIntPredicate(Predicate);
    // For EQ and NE, we can always pick a value for the undef to make the
    // predicate pass or fail, so we can return undef.
    // Also, if both operands are undef, we can return undef for int comparison.
    if (ICmpInst::isEquality(Predicate) || (isIntegerPredicate && C1 == C2))
      return UndefValue::get(ResultTy);

    // Otherwise, for integer compare, pick the same value as the non-undef
    // operand, and fold it to true or false.
    if (isIntegerPredicate)
      return ConstantInt::get(ResultTy, CmpInst::isTrueWhenEqual(Predicate));

    // Choosing NaN for the undef will always make unordered comparison succeed
    // and ordered comparison fails.
    return ConstantInt::get(ResultTy, CmpInst::isUnordered(Predicate));
  }

  // icmp eq/ne(null,GV) -> false/true
  if (C1->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C2))
      // Don't try to evaluate aliases.  External weak GV can be null.
      if (!isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage() &&
          !NullPointerIsDefined(nullptr /* F */,
                                GV->getType()->getAddressSpace())) {
        if (Predicate == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(C1->getContext());
        else if (Predicate == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(C1->getContext());
      }
  // icmp eq/ne(GV,null) -> false/true
  } else if (C2->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C1)) {
      // Don't try to evaluate aliases.  External weak GV can be null.
      if (!isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage() &&
          !NullPointerIsDefined(nullptr /* F */,
                                GV->getType()->getAddressSpace())) {
        if (Predicate == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(C1->getContext());
        else if (Predicate == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(C1->getContext());
      }
    }

    // The caller is expected to commute the operands if the constant expression
    // is C2.
    // C1 >= 0 --> true
    if (Predicate == ICmpInst::ICMP_UGE)
      return Constant::getAllOnesValue(ResultTy);
    // C1 < 0 --> false
    if (Predicate == ICmpInst::ICMP_ULT)
      return Constant::getNullValue(ResultTy);
  }

  // If the comparison is a comparison between two i1's, simplify it.
  if (C1->getType()->isIntegerTy(1)) {
    switch (Predicate) {
    case ICmpInst::ICMP_EQ:
      if (isa<ConstantInt>(C2))
        return ConstantExpr::getXor(C1, ConstantExpr::getNot(C2));
      return ConstantExpr::getXor(ConstantExpr::getNot(C1), C2);
    case ICmpInst::ICMP_NE:
      return ConstantExpr::getXor(C1, C2);
    default:
      break;
    }
  }

  if (isa<ConstantInt>(C1) && isa<ConstantInt>(C2)) {
    const APInt &V1 = cast<ConstantInt>(C1)->getValue();
    const APInt &V2 = cast<ConstantInt>(C2)->getValue();
    return ConstantInt::get(ResultTy, ICmpInst::compare(V1, V2, Predicate));
  } else if (isa<ConstantFP>(C1) && isa<ConstantFP>(C2)) {
    const APFloat &C1V = cast<ConstantFP>(C1)->getValueAPF();
    const APFloat &C2V = cast<ConstantFP>(C2)->getValueAPF();
    return ConstantInt::get(ResultTy, FCmpInst::compare(C1V, C2V, Predicate));
  } else if (auto *C1VTy = dyn_cast<VectorType>(C1->getType())) {

    // Fast path for splatted constants.
    if (Constant *C1Splat = C1->getSplatValue())
      if (Constant *C2Splat = C2->getSplatValue())
        return ConstantVector::getSplat(
            C1VTy->getElementCount(),
            ConstantExpr::getCompare(Predicate, C1Splat, C2Splat));

    // Do not iterate on scalable vector. The number of elements is unknown at
    // compile-time.
    if (isa<ScalableVectorType>(C1VTy))
      return nullptr;

    // If we can constant fold the comparison of each element, constant fold
    // the whole vector comparison.
    SmallVector<Constant*, 4> ResElts;
    Type *Ty = IntegerType::get(C1->getContext(), 32);
    // Compare the elements, producing an i1 result or constant expr.
    for (unsigned I = 0, E = C1VTy->getElementCount().getKnownMinValue();
         I != E; ++I) {
      Constant *C1E =
          ConstantExpr::getExtractElement(C1, ConstantInt::get(Ty, I));
      Constant *C2E =
          ConstantExpr::getExtractElement(C2, ConstantInt::get(Ty, I));

      ResElts.push_back(ConstantExpr::getCompare(Predicate, C1E, C2E));
    }

    return ConstantVector::get(ResElts);
  }

  if (C1->getType()->isFloatingPointTy() &&
      // Only call evaluateFCmpRelation if we have a constant expr to avoid
      // infinite recursive loop
      (isa<ConstantExpr>(C1) || isa<ConstantExpr>(C2))) {
    int Result = -1;  // -1 = unknown, 0 = known false, 1 = known true.
    switch (evaluateFCmpRelation(C1, C2)) {
    default: llvm_unreachable("Unknown relation!");
    case FCmpInst::FCMP_UNO:
    case FCmpInst::FCMP_ORD:
    case FCmpInst::FCMP_UNE:
    case FCmpInst::FCMP_ULT:
    case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_ULE:
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_TRUE:
    case FCmpInst::FCMP_FALSE:
    case FCmpInst::BAD_FCMP_PREDICATE:
      break; // Couldn't determine anything about these constants.
    case FCmpInst::FCMP_OEQ: // We know that C1 == C2
      Result =
          (Predicate == FCmpInst::FCMP_UEQ || Predicate == FCmpInst::FCMP_OEQ ||
           Predicate == FCmpInst::FCMP_ULE || Predicate == FCmpInst::FCMP_OLE ||
           Predicate == FCmpInst::FCMP_UGE || Predicate == FCmpInst::FCMP_OGE);
      break;
    case FCmpInst::FCMP_OLT: // We know that C1 < C2
      Result =
          (Predicate == FCmpInst::FCMP_UNE || Predicate == FCmpInst::FCMP_ONE ||
           Predicate == FCmpInst::FCMP_ULT || Predicate == FCmpInst::FCMP_OLT ||
           Predicate == FCmpInst::FCMP_ULE || Predicate == FCmpInst::FCMP_OLE);
      break;
    case FCmpInst::FCMP_OGT: // We know that C1 > C2
      Result =
          (Predicate == FCmpInst::FCMP_UNE || Predicate == FCmpInst::FCMP_ONE ||
           Predicate == FCmpInst::FCMP_UGT || Predicate == FCmpInst::FCMP_OGT ||
           Predicate == FCmpInst::FCMP_UGE || Predicate == FCmpInst::FCMP_OGE);
      break;
    case FCmpInst::FCMP_OLE: // We know that C1 <= C2
      // We can only partially decide this relation.
      if (Predicate == FCmpInst::FCMP_UGT || Predicate == FCmpInst::FCMP_OGT)
        Result = 0;
      else if (Predicate == FCmpInst::FCMP_ULT ||
               Predicate == FCmpInst::FCMP_OLT)
        Result = 1;
      break;
    case FCmpInst::FCMP_OGE: // We known that C1 >= C2
      // We can only partially decide this relation.
      if (Predicate == FCmpInst::FCMP_ULT || Predicate == FCmpInst::FCMP_OLT)
        Result = 0;
      else if (Predicate == FCmpInst::FCMP_UGT ||
               Predicate == FCmpInst::FCMP_OGT)
        Result = 1;
      break;
    case FCmpInst::FCMP_ONE: // We know that C1 != C2
      // We can only partially decide this relation.
      if (Predicate == FCmpInst::FCMP_OEQ || Predicate == FCmpInst::FCMP_UEQ)
        Result = 0;
      else if (Predicate == FCmpInst::FCMP_ONE ||
               Predicate == FCmpInst::FCMP_UNE)
        Result = 1;
      break;
    case FCmpInst::FCMP_UEQ: // We know that C1 == C2 || isUnordered(C1, C2).
      // We can only partially decide this relation.
      if (Predicate == FCmpInst::FCMP_ONE)
        Result = 0;
      else if (Predicate == FCmpInst::FCMP_UEQ)
        Result = 1;
      break;
    }

    // If we evaluated the result, return it now.
    if (Result != -1)
      return ConstantInt::get(ResultTy, Result);

  } else {
    // Evaluate the relation between the two constants, per the predicate.
    int Result = -1;  // -1 = unknown, 0 = known false, 1 = known true.
    switch (evaluateICmpRelation(C1, C2, CmpInst::isSigned(Predicate))) {
    default: llvm_unreachable("Unknown relational!");
    case ICmpInst::BAD_ICMP_PREDICATE:
      break;  // Couldn't determine anything about these constants.
    case ICmpInst::ICMP_EQ:   // We know the constants are equal!
      // If we know the constants are equal, we can decide the result of this
      // computation precisely.
      Result = ICmpInst::isTrueWhenEqual(Predicate);
      break;
    case ICmpInst::ICMP_ULT:
      switch (Predicate) {
      case ICmpInst::ICMP_ULT: case ICmpInst::ICMP_NE: case ICmpInst::ICMP_ULE:
        Result = 1; break;
      case ICmpInst::ICMP_UGT: case ICmpInst::ICMP_EQ: case ICmpInst::ICMP_UGE:
        Result = 0; break;
      default:
        break;
      }
      break;
    case ICmpInst::ICMP_SLT:
      switch (Predicate) {
      case ICmpInst::ICMP_SLT: case ICmpInst::ICMP_NE: case ICmpInst::ICMP_SLE:
        Result = 1; break;
      case ICmpInst::ICMP_SGT: case ICmpInst::ICMP_EQ: case ICmpInst::ICMP_SGE:
        Result = 0; break;
      default:
        break;
      }
      break;
    case ICmpInst::ICMP_UGT:
      switch (Predicate) {
      case ICmpInst::ICMP_UGT: case ICmpInst::ICMP_NE: case ICmpInst::ICMP_UGE:
        Result = 1; break;
      case ICmpInst::ICMP_ULT: case ICmpInst::ICMP_EQ: case ICmpInst::ICMP_ULE:
        Result = 0; break;
      default:
        break;
      }
      break;
    case ICmpInst::ICMP_SGT:
      switch (Predicate) {
      case ICmpInst::ICMP_SGT: case ICmpInst::ICMP_NE: case ICmpInst::ICMP_SGE:
        Result = 1; break;
      case ICmpInst::ICMP_SLT: case ICmpInst::ICMP_EQ: case ICmpInst::ICMP_SLE:
        Result = 0; break;
      default:
        break;
      }
      break;
    case ICmpInst::ICMP_ULE:
      if (Predicate == ICmpInst::ICMP_UGT)
        Result = 0;
      if (Predicate == ICmpInst::ICMP_ULT || Predicate == ICmpInst::ICMP_ULE)
        Result = 1;
      break;
    case ICmpInst::ICMP_SLE:
      if (Predicate == ICmpInst::ICMP_SGT)
        Result = 0;
      if (Predicate == ICmpInst::ICMP_SLT || Predicate == ICmpInst::ICMP_SLE)
        Result = 1;
      break;
    case ICmpInst::ICMP_UGE:
      if (Predicate == ICmpInst::ICMP_ULT)
        Result = 0;
      if (Predicate == ICmpInst::ICMP_UGT || Predicate == ICmpInst::ICMP_UGE)
        Result = 1;
      break;
    case ICmpInst::ICMP_SGE:
      if (Predicate == ICmpInst::ICMP_SLT)
        Result = 0;
      if (Predicate == ICmpInst::ICMP_SGT || Predicate == ICmpInst::ICMP_SGE)
        Result = 1;
      break;
    case ICmpInst::ICMP_NE:
      if (Predicate == ICmpInst::ICMP_EQ)
        Result = 0;
      if (Predicate == ICmpInst::ICMP_NE)
        Result = 1;
      break;
    }

    // If we evaluated the result, return it now.
    if (Result != -1)
      return ConstantInt::get(ResultTy, Result);

    // If the right hand side is a bitcast, try using its inverse to simplify
    // it by moving it to the left hand side.  We can't do this if it would turn
    // a vector compare into a scalar compare or visa versa, or if it would turn
    // the operands into FP values.
    if (ConstantExpr *CE2 = dyn_cast<ConstantExpr>(C2)) {
      Constant *CE2Op0 = CE2->getOperand(0);
      if (CE2->getOpcode() == Instruction::BitCast &&
          CE2->getType()->isVectorTy() == CE2Op0->getType()->isVectorTy() &&
          !CE2Op0->getType()->isFPOrFPVectorTy()) {
        Constant *Inverse = ConstantExpr::getBitCast(C1, CE2Op0->getType());
        return ConstantExpr::getICmp(Predicate, Inverse, CE2Op0);
      }
    }

    // If the left hand side is an extension, try eliminating it.
    if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
      if ((CE1->getOpcode() == Instruction::SExt &&
           ICmpInst::isSigned(Predicate)) ||
          (CE1->getOpcode() == Instruction::ZExt &&
           !ICmpInst::isSigned(Predicate))) {
        Constant *CE1Op0 = CE1->getOperand(0);
        Constant *CE1Inverse = ConstantExpr::getTrunc(CE1, CE1Op0->getType());
        if (CE1Inverse == CE1Op0) {
          // Check whether we can safely truncate the right hand side.
          Constant *C2Inverse = ConstantExpr::getTrunc(C2, CE1Op0->getType());
          if (ConstantExpr::getCast(CE1->getOpcode(), C2Inverse,
                                    C2->getType()) == C2)
            return ConstantExpr::getICmp(Predicate, CE1Inverse, C2Inverse);
        }
      }
    }

    if ((!isa<ConstantExpr>(C1) && isa<ConstantExpr>(C2)) ||
        (C1->isNullValue() && !C2->isNullValue())) {
      // If C2 is a constant expr and C1 isn't, flip them around and fold the
      // other way if possible.
      // Also, if C1 is null and C2 isn't, flip them around.
      Predicate = ICmpInst::getSwappedPredicate(Predicate);
      return ConstantExpr::getICmp(Predicate, C2, C1);
    }
  }
  return nullptr;
}

/// Test whether the given sequence of *normalized* indices is "inbounds".
template<typename IndexTy>
static bool isInBoundsIndices(ArrayRef<IndexTy> Idxs) {
  // No indices means nothing that could be out of bounds.
  if (Idxs.empty()) return true;

  // If the first index is zero, it's in bounds.
  if (cast<Constant>(Idxs[0])->isNullValue()) return true;

  // If the first index is one and all the rest are zero, it's in bounds,
  // by the one-past-the-end rule.
  if (auto *CI = dyn_cast<ConstantInt>(Idxs[0])) {
    if (!CI->isOne())
      return false;
  } else {
    auto *CV = cast<ConstantDataVector>(Idxs[0]);
    CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
    if (!CI || !CI->isOne())
      return false;
  }

  for (unsigned i = 1, e = Idxs.size(); i != e; ++i)
    if (!cast<Constant>(Idxs[i])->isNullValue())
      return false;
  return true;
}

/// Test whether a given ConstantInt is in-range for a SequentialType.
static bool isIndexInRangeOfArrayType(uint64_t NumElements,
                                      const ConstantInt *CI) {
  // We cannot bounds check the index if it doesn't fit in an int64_t.
  if (CI->getValue().getMinSignedBits() > 64)
    return false;

  // A negative index or an index past the end of our sequential type is
  // considered out-of-range.
  int64_t IndexVal = CI->getSExtValue();
  if (IndexVal < 0 || (NumElements > 0 && (uint64_t)IndexVal >= NumElements))
    return false;

  // Otherwise, it is in-range.
  return true;
}

// Combine Indices - If the source pointer to this getelementptr instruction
// is a getelementptr instruction, combine the indices of the two
// getelementptr instructions into a single instruction.
static Constant *foldGEPOfGEP(GEPOperator *GEP, Type *PointeeTy, bool InBounds,
                              ArrayRef<Value *> Idxs) {
  if (PointeeTy != GEP->getResultElementType())
    return nullptr;

  Constant *Idx0 = cast<Constant>(Idxs[0]);
  if (Idx0->isNullValue()) {
    // Handle the simple case of a zero index.
    SmallVector<Value*, 16> NewIndices;
    NewIndices.reserve(Idxs.size() + GEP->getNumIndices());
    NewIndices.append(GEP->idx_begin(), GEP->idx_end());
    NewIndices.append(Idxs.begin() + 1, Idxs.end());
    return ConstantExpr::getGetElementPtr(
        GEP->getSourceElementType(), cast<Constant>(GEP->getPointerOperand()),
        NewIndices, InBounds && GEP->isInBounds(), GEP->getInRangeIndex());
  }

  gep_type_iterator LastI = gep_type_end(GEP);
  for (gep_type_iterator I = gep_type_begin(GEP), E = gep_type_end(GEP);
       I != E; ++I)
    LastI = I;

  // We can't combine GEPs if the last index is a struct type.
  if (!LastI.isSequential())
    return nullptr;
  // We could perform the transform with non-constant index, but prefer leaving
  // it as GEP of GEP rather than GEP of add for now.
  ConstantInt *CI = dyn_cast<ConstantInt>(Idx0);
  if (!CI)
    return nullptr;

  // TODO: This code may be extended to handle vectors as well.
  auto *LastIdx = cast<Constant>(GEP->getOperand(GEP->getNumOperands()-1));
  Type *LastIdxTy = LastIdx->getType();
  if (LastIdxTy->isVectorTy())
    return nullptr;

  SmallVector<Value*, 16> NewIndices;
  NewIndices.reserve(Idxs.size() + GEP->getNumIndices());
  NewIndices.append(GEP->idx_begin(), GEP->idx_end() - 1);

  // Add the last index of the source with the first index of the new GEP.
  // Make sure to handle the case when they are actually different types.
  if (LastIdxTy != Idx0->getType()) {
    unsigned CommonExtendedWidth =
        std::max(LastIdxTy->getIntegerBitWidth(),
                 Idx0->getType()->getIntegerBitWidth());
    CommonExtendedWidth = std::max(CommonExtendedWidth, 64U);

    Type *CommonTy =
        Type::getIntNTy(LastIdxTy->getContext(), CommonExtendedWidth);
    Idx0 = ConstantExpr::getSExtOrBitCast(Idx0, CommonTy);
    LastIdx = ConstantExpr::getSExtOrBitCast(LastIdx, CommonTy);
  }

  NewIndices.push_back(ConstantExpr::get(Instruction::Add, Idx0, LastIdx));
  NewIndices.append(Idxs.begin() + 1, Idxs.end());

  // The combined GEP normally inherits its index inrange attribute from
  // the inner GEP, but if the inner GEP's last index was adjusted by the
  // outer GEP, any inbounds attribute on that index is invalidated.
  Optional<unsigned> IRIndex = GEP->getInRangeIndex();
  if (IRIndex && *IRIndex == GEP->getNumIndices() - 1)
    IRIndex = None;

  return ConstantExpr::getGetElementPtr(
      GEP->getSourceElementType(), cast<Constant>(GEP->getPointerOperand()),
      NewIndices, InBounds && GEP->isInBounds(), IRIndex);
}

Constant *llvm::ConstantFoldGetElementPtr(Type *PointeeTy, Constant *C,
                                          bool InBounds,
                                          Optional<unsigned> InRangeIndex,
                                          ArrayRef<Value *> Idxs) {
  if (Idxs.empty()) return C;

  Type *GEPTy = GetElementPtrInst::getGEPReturnType(
      PointeeTy, C, makeArrayRef((Value *const *)Idxs.data(), Idxs.size()));

  if (isa<PoisonValue>(C))
    return PoisonValue::get(GEPTy);

  if (isa<UndefValue>(C))
    // If inbounds, we can choose an out-of-bounds pointer as a base pointer.
    return InBounds ? PoisonValue::get(GEPTy) : UndefValue::get(GEPTy);

  auto IsNoOp = [&]() {
    // For non-opaque pointers having multiple indices will change the result
    // type of the GEP.
    if (!C->getType()->getScalarType()->isOpaquePointerTy() && Idxs.size() != 1)
      return false;

    return all_of(Idxs, [](Value *Idx) {
      Constant *IdxC = cast<Constant>(Idx);
      return IdxC->isNullValue() || isa<UndefValue>(IdxC);
    });
  };
  if (IsNoOp())
    return GEPTy->isVectorTy() && !C->getType()->isVectorTy()
               ? ConstantVector::getSplat(
                     cast<VectorType>(GEPTy)->getElementCount(), C)
               : C;

  if (C->isNullValue()) {
    bool isNull = true;
    for (Value *Idx : Idxs)
      if (!isa<UndefValue>(Idx) && !cast<Constant>(Idx)->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) {
      PointerType *PtrTy = cast<PointerType>(C->getType()->getScalarType());
      Type *Ty = GetElementPtrInst::getIndexedType(PointeeTy, Idxs);

      assert(Ty && "Invalid indices for GEP!");
      Type *OrigGEPTy = PointerType::get(Ty, PtrTy->getAddressSpace());
      Type *GEPTy = PointerType::get(Ty, PtrTy->getAddressSpace());
      if (VectorType *VT = dyn_cast<VectorType>(C->getType()))
        GEPTy = VectorType::get(OrigGEPTy, VT->getElementCount());

      // The GEP returns a vector of pointers when one of more of
      // its arguments is a vector.
      for (Value *Idx : Idxs) {
        if (auto *VT = dyn_cast<VectorType>(Idx->getType())) {
          assert((!isa<VectorType>(GEPTy) || isa<ScalableVectorType>(GEPTy) ==
                                                 isa<ScalableVectorType>(VT)) &&
                 "Mismatched GEPTy vector types");
          GEPTy = VectorType::get(OrigGEPTy, VT->getElementCount());
          break;
        }
      }

      return Constant::getNullValue(GEPTy);
    }
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    if (auto *GEP = dyn_cast<GEPOperator>(CE))
      if (Constant *C = foldGEPOfGEP(GEP, PointeeTy, InBounds, Idxs))
        return C;

    // Attempt to fold casts to the same type away.  For example, folding:
    //
    //   i32* getelementptr ([2 x i32]* bitcast ([3 x i32]* %X to [2 x i32]*),
    //                       i64 0, i64 0)
    // into:
    //
    //   i32* getelementptr ([3 x i32]* %X, i64 0, i64 0)
    //
    // Don't fold if the cast is changing address spaces.
    Constant *Idx0 = cast<Constant>(Idxs[0]);
    if (CE->isCast() && Idxs.size() > 1 && Idx0->isNullValue()) {
      PointerType *SrcPtrTy =
        dyn_cast<PointerType>(CE->getOperand(0)->getType());
      PointerType *DstPtrTy = dyn_cast<PointerType>(CE->getType());
      if (SrcPtrTy && DstPtrTy && !SrcPtrTy->isOpaque() &&
          !DstPtrTy->isOpaque()) {
        ArrayType *SrcArrayTy =
          dyn_cast<ArrayType>(SrcPtrTy->getNonOpaquePointerElementType());
        ArrayType *DstArrayTy =
          dyn_cast<ArrayType>(DstPtrTy->getNonOpaquePointerElementType());
        if (SrcArrayTy && DstArrayTy
            && SrcArrayTy->getElementType() == DstArrayTy->getElementType()
            && SrcPtrTy->getAddressSpace() == DstPtrTy->getAddressSpace())
          return ConstantExpr::getGetElementPtr(SrcArrayTy,
                                                (Constant *)CE->getOperand(0),
                                                Idxs, InBounds, InRangeIndex);
      }
    }
  }

  // Check to see if any array indices are not within the corresponding
  // notional array or vector bounds. If so, try to determine if they can be
  // factored out into preceding dimensions.
  SmallVector<Constant *, 8> NewIdxs;
  Type *Ty = PointeeTy;
  Type *Prev = C->getType();
  auto GEPIter = gep_type_begin(PointeeTy, Idxs);
  bool Unknown =
      !isa<ConstantInt>(Idxs[0]) && !isa<ConstantDataVector>(Idxs[0]);
  for (unsigned i = 1, e = Idxs.size(); i != e;
       Prev = Ty, Ty = (++GEPIter).getIndexedType(), ++i) {
    if (!isa<ConstantInt>(Idxs[i]) && !isa<ConstantDataVector>(Idxs[i])) {
      // We don't know if it's in range or not.
      Unknown = true;
      continue;
    }
    if (!isa<ConstantInt>(Idxs[i - 1]) && !isa<ConstantDataVector>(Idxs[i - 1]))
      // Skip if the type of the previous index is not supported.
      continue;
    if (InRangeIndex && i == *InRangeIndex + 1) {
      // If an index is marked inrange, we cannot apply this canonicalization to
      // the following index, as that will cause the inrange index to point to
      // the wrong element.
      continue;
    }
    if (isa<StructType>(Ty)) {
      // The verify makes sure that GEPs into a struct are in range.
      continue;
    }
    if (isa<VectorType>(Ty)) {
      // There can be awkward padding in after a non-power of two vector.
      Unknown = true;
      continue;
    }
    auto *STy = cast<ArrayType>(Ty);
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Idxs[i])) {
      if (isIndexInRangeOfArrayType(STy->getNumElements(), CI))
        // It's in range, skip to the next index.
        continue;
      if (CI->isNegative()) {
        // It's out of range and negative, don't try to factor it.
        Unknown = true;
        continue;
      }
    } else {
      auto *CV = cast<ConstantDataVector>(Idxs[i]);
      bool InRange = true;
      for (unsigned I = 0, E = CV->getNumElements(); I != E; ++I) {
        auto *CI = cast<ConstantInt>(CV->getElementAsConstant(I));
        InRange &= isIndexInRangeOfArrayType(STy->getNumElements(), CI);
        if (CI->isNegative()) {
          Unknown = true;
          break;
        }
      }
      if (InRange || Unknown)
        // It's in range, skip to the next index.
        // It's out of range and negative, don't try to factor it.
        continue;
    }
    if (isa<StructType>(Prev)) {
      // It's out of range, but the prior dimension is a struct
      // so we can't do anything about it.
      Unknown = true;
      continue;
    }
    // It's out of range, but we can factor it into the prior
    // dimension.
    NewIdxs.resize(Idxs.size());
    // Determine the number of elements in our sequential type.
    uint64_t NumElements = STy->getArrayNumElements();

    // Expand the current index or the previous index to a vector from a scalar
    // if necessary.
    Constant *CurrIdx = cast<Constant>(Idxs[i]);
    auto *PrevIdx =
        NewIdxs[i - 1] ? NewIdxs[i - 1] : cast<Constant>(Idxs[i - 1]);
    bool IsCurrIdxVector = CurrIdx->getType()->isVectorTy();
    bool IsPrevIdxVector = PrevIdx->getType()->isVectorTy();
    bool UseVector = IsCurrIdxVector || IsPrevIdxVector;

    if (!IsCurrIdxVector && IsPrevIdxVector)
      CurrIdx = ConstantDataVector::getSplat(
          cast<FixedVectorType>(PrevIdx->getType())->getNumElements(), CurrIdx);

    if (!IsPrevIdxVector && IsCurrIdxVector)
      PrevIdx = ConstantDataVector::getSplat(
          cast<FixedVectorType>(CurrIdx->getType())->getNumElements(), PrevIdx);

    Constant *Factor =
        ConstantInt::get(CurrIdx->getType()->getScalarType(), NumElements);
    if (UseVector)
      Factor = ConstantDataVector::getSplat(
          IsPrevIdxVector
              ? cast<FixedVectorType>(PrevIdx->getType())->getNumElements()
              : cast<FixedVectorType>(CurrIdx->getType())->getNumElements(),
          Factor);

    NewIdxs[i] = ConstantExpr::getSRem(CurrIdx, Factor);

    Constant *Div = ConstantExpr::getSDiv(CurrIdx, Factor);

    unsigned CommonExtendedWidth =
        std::max(PrevIdx->getType()->getScalarSizeInBits(),
                 Div->getType()->getScalarSizeInBits());
    CommonExtendedWidth = std::max(CommonExtendedWidth, 64U);

    // Before adding, extend both operands to i64 to avoid
    // overflow trouble.
    Type *ExtendedTy = Type::getIntNTy(Div->getContext(), CommonExtendedWidth);
    if (UseVector)
      ExtendedTy = FixedVectorType::get(
          ExtendedTy,
          IsPrevIdxVector
              ? cast<FixedVectorType>(PrevIdx->getType())->getNumElements()
              : cast<FixedVectorType>(CurrIdx->getType())->getNumElements());

    if (!PrevIdx->getType()->isIntOrIntVectorTy(CommonExtendedWidth))
      PrevIdx = ConstantExpr::getSExt(PrevIdx, ExtendedTy);

    if (!Div->getType()->isIntOrIntVectorTy(CommonExtendedWidth))
      Div = ConstantExpr::getSExt(Div, ExtendedTy);

    NewIdxs[i - 1] = ConstantExpr::getAdd(PrevIdx, Div);
  }

  // If we did any factoring, start over with the adjusted indices.
  if (!NewIdxs.empty()) {
    for (unsigned i = 0, e = Idxs.size(); i != e; ++i)
      if (!NewIdxs[i]) NewIdxs[i] = cast<Constant>(Idxs[i]);
    return ConstantExpr::getGetElementPtr(PointeeTy, C, NewIdxs, InBounds,
                                          InRangeIndex);
  }

  // If all indices are known integers and normalized, we can do a simple
  // check for the "inbounds" property.
  if (!Unknown && !InBounds)
    if (auto *GV = dyn_cast<GlobalVariable>(C))
      if (!GV->hasExternalWeakLinkage() && isInBoundsIndices(Idxs))
        return ConstantExpr::getGetElementPtr(PointeeTy, C, Idxs,
                                              /*InBounds=*/true, InRangeIndex);

  return nullptr;
}
