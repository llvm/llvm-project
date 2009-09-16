//===- ConstantFold.cpp - LLVM constant folder ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements folding of constants for LLVM.  This implements the
// (internal) ConstantFold.h interface, which is used by the
// ConstantExpr::get* methods to automatically fold constants when possible.
//
// The current constant folding implementation is implemented in two pieces: the
// template-based folder for simple primitive constants like ConstantInt, and
// the special case hackery that we use to symbolically evaluate expressions
// that use ConstantExprs.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalAlias.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                ConstantFold*Instruction Implementations
//===----------------------------------------------------------------------===//

/// BitCastConstantVector - Convert the specified ConstantVector node to the
/// specified vector type.  At this point, we know that the elements of the
/// input vector constant are all simple integer or FP values.
static Constant *BitCastConstantVector(LLVMContext &Context, ConstantVector *CV,
                                       const VectorType *DstTy) {
  // If this cast changes element count then we can't handle it here:
  // doing so requires endianness information.  This should be handled by
  // Analysis/ConstantFolding.cpp
  unsigned NumElts = DstTy->getNumElements();
  if (NumElts != CV->getNumOperands())
    return 0;
  
  // Check to verify that all elements of the input are simple.
  for (unsigned i = 0; i != NumElts; ++i) {
    if (!isa<ConstantInt>(CV->getOperand(i)) &&
        !isa<ConstantFP>(CV->getOperand(i)))
      return 0;
  }

  // Bitcast each element now.
  std::vector<Constant*> Result;
  const Type *DstEltTy = DstTy->getElementType();
  for (unsigned i = 0; i != NumElts; ++i)
    Result.push_back(ConstantExpr::getBitCast(CV->getOperand(i),
                                                    DstEltTy));
  return ConstantVector::get(Result);
}

/// This function determines which opcode to use to fold two constant cast 
/// expressions together. It uses CastInst::isEliminableCastPair to determine
/// the opcode. Consequently its just a wrapper around that function.
/// @brief Determine if it is valid to fold a cast of a cast
static unsigned
foldConstantCastPair(
  unsigned opc,          ///< opcode of the second cast constant expression
  const ConstantExpr*Op, ///< the first cast constant expression
  const Type *DstTy      ///< desintation type of the first cast
) {
  assert(Op && Op->isCast() && "Can't fold cast of cast without a cast!");
  assert(DstTy && DstTy->isFirstClassType() && "Invalid cast destination type");
  assert(CastInst::isCast(opc) && "Invalid cast opcode");
  
  // The the types and opcodes for the two Cast constant expressions
  const Type *SrcTy = Op->getOperand(0)->getType();
  const Type *MidTy = Op->getType();
  Instruction::CastOps firstOp = Instruction::CastOps(Op->getOpcode());
  Instruction::CastOps secondOp = Instruction::CastOps(opc);

  // Let CastInst::isEliminableCastPair do the heavy lifting.
  return CastInst::isEliminableCastPair(firstOp, secondOp, SrcTy, MidTy, DstTy,
                                        Type::getInt64Ty(DstTy->getContext()));
}

static Constant *FoldBitCast(LLVMContext &Context, 
                             Constant *V, const Type *DestTy) {
  const Type *SrcTy = V->getType();
  if (SrcTy == DestTy)
    return V; // no-op cast
  
  // Check to see if we are casting a pointer to an aggregate to a pointer to
  // the first element.  If so, return the appropriate GEP instruction.
  if (const PointerType *PTy = dyn_cast<PointerType>(V->getType()))
    if (const PointerType *DPTy = dyn_cast<PointerType>(DestTy))
      if (PTy->getAddressSpace() == DPTy->getAddressSpace()) {
        SmallVector<Value*, 8> IdxList;
        Value *Zero = Constant::getNullValue(Type::getInt32Ty(Context));
        IdxList.push_back(Zero);
        const Type *ElTy = PTy->getElementType();
        while (ElTy != DPTy->getElementType()) {
          if (const StructType *STy = dyn_cast<StructType>(ElTy)) {
            if (STy->getNumElements() == 0) break;
            ElTy = STy->getElementType(0);
            IdxList.push_back(Zero);
          } else if (const SequentialType *STy = 
                     dyn_cast<SequentialType>(ElTy)) {
            if (isa<PointerType>(ElTy)) break;  // Can't index into pointers!
            ElTy = STy->getElementType();
            IdxList.push_back(Zero);
          } else {
            break;
          }
        }
        
        if (ElTy == DPTy->getElementType())
          // This GEP is inbounds because all indices are zero.
          return ConstantExpr::getInBoundsGetElementPtr(V, &IdxList[0],
                                                        IdxList.size());
      }
  
  // Handle casts from one vector constant to another.  We know that the src 
  // and dest type have the same size (otherwise its an illegal cast).
  if (const VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
    if (const VectorType *SrcTy = dyn_cast<VectorType>(V->getType())) {
      assert(DestPTy->getBitWidth() == SrcTy->getBitWidth() &&
             "Not cast between same sized vectors!");
      SrcTy = NULL;
      // First, check for null.  Undef is already handled.
      if (isa<ConstantAggregateZero>(V))
        return Constant::getNullValue(DestTy);
      
      if (ConstantVector *CV = dyn_cast<ConstantVector>(V))
        return BitCastConstantVector(Context, CV, DestPTy);
    }

    // Canonicalize scalar-to-vector bitcasts into vector-to-vector bitcasts
    // This allows for other simplifications (although some of them
    // can only be handled by Analysis/ConstantFolding.cpp).
    if (isa<ConstantInt>(V) || isa<ConstantFP>(V))
      return ConstantExpr::getBitCast(
                                     ConstantVector::get(&V, 1), DestPTy);
  }
  
  // Finally, implement bitcast folding now.   The code below doesn't handle
  // bitcast right.
  if (isa<ConstantPointerNull>(V))  // ptr->ptr cast.
    return ConstantPointerNull::get(cast<PointerType>(DestTy));
  
  // Handle integral constant input.
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (DestTy->isInteger())
      // Integral -> Integral. This is a no-op because the bit widths must
      // be the same. Consequently, we just fold to V.
      return V;

    if (DestTy->isFloatingPoint())
      return ConstantFP::get(Context, APFloat(CI->getValue(),
                                     DestTy != Type::getPPC_FP128Ty(Context)));

    // Otherwise, can't fold this (vector?)
    return 0;
  }

  // Handle ConstantFP input.
  if (const ConstantFP *FP = dyn_cast<ConstantFP>(V))
    // FP -> Integral.
    return ConstantInt::get(Context, FP->getValueAPF().bitcastToAPInt());

  return 0;
}


Constant *llvm::ConstantFoldCastInstruction(LLVMContext &Context, 
                                            unsigned opc, const Constant *V,
                                            const Type *DestTy) {
  if (isa<UndefValue>(V)) {
    // zext(undef) = 0, because the top bits will be zero.
    // sext(undef) = 0, because the top bits will all be the same.
    // [us]itofp(undef) = 0, because the result value is bounded.
    if (opc == Instruction::ZExt || opc == Instruction::SExt ||
        opc == Instruction::UIToFP || opc == Instruction::SIToFP)
      return Constant::getNullValue(DestTy);
    return UndefValue::get(DestTy);
  }
  // No compile-time operations on this type yet.
  if (V->getType() == Type::getPPC_FP128Ty(Context) || DestTy == Type::getPPC_FP128Ty(Context))
    return 0;

  // If the cast operand is a constant expression, there's a few things we can
  // do to try to simplify it.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->isCast()) {
      // Try hard to fold cast of cast because they are often eliminable.
      if (unsigned newOpc = foldConstantCastPair(opc, CE, DestTy))
        return ConstantExpr::getCast(newOpc, CE->getOperand(0), DestTy);
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
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
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(V))
    if (isa<VectorType>(DestTy) &&
        cast<VectorType>(DestTy)->getNumElements() ==
        CV->getType()->getNumElements()) {
      std::vector<Constant*> res;
      const VectorType *DestVecTy = cast<VectorType>(DestTy);
      const Type *DstEltTy = DestVecTy->getElementType();
      for (unsigned i = 0, e = CV->getType()->getNumElements(); i != e; ++i)
        res.push_back(ConstantExpr::getCast(opc,
                                            CV->getOperand(i), DstEltTy));
      return ConstantVector::get(DestVecTy, res);
    }

  // We actually have to do a cast now. Perform the cast according to the
  // opcode specified.
  switch (opc) {
  case Instruction::FPTrunc:
  case Instruction::FPExt:
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(V)) {
      bool ignored;
      APFloat Val = FPC->getValueAPF();
      Val.convert(DestTy == Type::getFloatTy(Context) ? APFloat::IEEEsingle :
                  DestTy == Type::getDoubleTy(Context) ? APFloat::IEEEdouble :
                  DestTy == Type::getX86_FP80Ty(Context) ? APFloat::x87DoubleExtended :
                  DestTy == Type::getFP128Ty(Context) ? APFloat::IEEEquad :
                  APFloat::Bogus,
                  APFloat::rmNearestTiesToEven, &ignored);
      return ConstantFP::get(Context, Val);
    }
    return 0; // Can't fold.
  case Instruction::FPToUI: 
  case Instruction::FPToSI:
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(V)) {
      const APFloat &V = FPC->getValueAPF();
      bool ignored;
      uint64_t x[2]; 
      uint32_t DestBitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      (void) V.convertToInteger(x, DestBitWidth, opc==Instruction::FPToSI,
                                APFloat::rmTowardZero, &ignored);
      APInt Val(DestBitWidth, 2, x);
      return ConstantInt::get(Context, Val);
    }
    return 0; // Can't fold.
  case Instruction::IntToPtr:   //always treated as unsigned
    if (V->isNullValue())       // Is it an integral null value?
      return ConstantPointerNull::get(cast<PointerType>(DestTy));
    return 0;                   // Other pointer types cannot be casted
  case Instruction::PtrToInt:   // always treated as unsigned
    if (V->isNullValue())       // is it a null pointer value?
      return ConstantInt::get(DestTy, 0);
    return 0;                   // Other pointer types cannot be casted
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      APInt api = CI->getValue();
      const uint64_t zero[] = {0, 0};
      APFloat apf = APFloat(APInt(DestTy->getPrimitiveSizeInBits(),
                                  2, zero));
      (void)apf.convertFromAPInt(api, 
                                 opc==Instruction::SIToFP,
                                 APFloat::rmNearestTiesToEven);
      return ConstantFP::get(Context, apf);
    }
    return 0;
  case Instruction::ZExt:
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      uint32_t BitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      APInt Result(CI->getValue());
      Result.zext(BitWidth);
      return ConstantInt::get(Context, Result);
    }
    return 0;
  case Instruction::SExt:
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      uint32_t BitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      APInt Result(CI->getValue());
      Result.sext(BitWidth);
      return ConstantInt::get(Context, Result);
    }
    return 0;
  case Instruction::Trunc:
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      uint32_t BitWidth = cast<IntegerType>(DestTy)->getBitWidth();
      APInt Result(CI->getValue());
      Result.trunc(BitWidth);
      return ConstantInt::get(Context, Result);
    }
    return 0;
  case Instruction::BitCast:
    return FoldBitCast(Context, const_cast<Constant*>(V), DestTy);
  default:
    assert(!"Invalid CE CastInst opcode");
    break;
  }

  llvm_unreachable("Failed to cast constant expression");
  return 0;
}

Constant *llvm::ConstantFoldSelectInstruction(LLVMContext&,
                                              const Constant *Cond,
                                              const Constant *V1,
                                              const Constant *V2) {
  if (const ConstantInt *CB = dyn_cast<ConstantInt>(Cond))
    return const_cast<Constant*>(CB->getZExtValue() ? V1 : V2);

  if (isa<UndefValue>(V1)) return const_cast<Constant*>(V2);
  if (isa<UndefValue>(V2)) return const_cast<Constant*>(V1);
  if (isa<UndefValue>(Cond)) return const_cast<Constant*>(V1);
  if (V1 == V2) return const_cast<Constant*>(V1);
  return 0;
}

Constant *llvm::ConstantFoldExtractElementInstruction(LLVMContext &Context,
                                                      const Constant *Val,
                                                      const Constant *Idx) {
  if (isa<UndefValue>(Val))  // ee(undef, x) -> undef
    return UndefValue::get(cast<VectorType>(Val->getType())->getElementType());
  if (Val->isNullValue())  // ee(zero, x) -> zero
    return Constant::getNullValue(
                          cast<VectorType>(Val->getType())->getElementType());
  
  if (const ConstantVector *CVal = dyn_cast<ConstantVector>(Val)) {
    if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx)) {
      return CVal->getOperand(CIdx->getZExtValue());
    } else if (isa<UndefValue>(Idx)) {
      // ee({w,x,y,z}, undef) -> w (an arbitrary value).
      return CVal->getOperand(0);
    }
  }
  return 0;
}

Constant *llvm::ConstantFoldInsertElementInstruction(LLVMContext &Context,
                                                     const Constant *Val,
                                                     const Constant *Elt,
                                                     const Constant *Idx) {
  const ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx) return 0;
  APInt idxVal = CIdx->getValue();
  if (isa<UndefValue>(Val)) { 
    // Insertion of scalar constant into vector undef
    // Optimize away insertion of undef
    if (isa<UndefValue>(Elt))
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate undef into multiple undefs and do
    // the insertion
    unsigned numOps = 
      cast<VectorType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (idxVal == i) ? Elt : UndefValue::get(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantVector::get(Ops);
  }
  if (isa<ConstantAggregateZero>(Val)) {
    // Insertion of scalar constant into vector aggregate zero
    // Optimize away insertion of zero
    if (Elt->isNullValue())
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate zero into multiple zeros and do
    // the insertion
    unsigned numOps = 
      cast<VectorType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (idxVal == i) ? Elt : Constant::getNullValue(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantVector::get(Ops);
  }
  if (const ConstantVector *CVal = dyn_cast<ConstantVector>(Val)) {
    // Insertion of scalar constant into vector constant
    std::vector<Constant*> Ops; 
    Ops.reserve(CVal->getNumOperands());
    for (unsigned i = 0; i < CVal->getNumOperands(); ++i) {
      const Constant *Op =
        (idxVal == i) ? Elt : cast<Constant>(CVal->getOperand(i));
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantVector::get(Ops);
  }

  return 0;
}

/// GetVectorElement - If C is a ConstantVector, ConstantAggregateZero or Undef
/// return the specified element value.  Otherwise return null.
static Constant *GetVectorElement(LLVMContext &Context, const Constant *C,
                                  unsigned EltNo) {
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(C))
    return CV->getOperand(EltNo);
  
  const Type *EltTy = cast<VectorType>(C->getType())->getElementType();
  if (isa<ConstantAggregateZero>(C))
    return Constant::getNullValue(EltTy);
  if (isa<UndefValue>(C))
    return UndefValue::get(EltTy);
  return 0;
}

Constant *llvm::ConstantFoldShuffleVectorInstruction(LLVMContext &Context,
                                                     const Constant *V1,
                                                     const Constant *V2,
                                                     const Constant *Mask) {
  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(Mask)) return UndefValue::get(V1->getType());

  unsigned MaskNumElts = cast<VectorType>(Mask->getType())->getNumElements();
  unsigned SrcNumElts = cast<VectorType>(V1->getType())->getNumElements();
  const Type *EltTy = cast<VectorType>(V1->getType())->getElementType();

  // Loop over the shuffle mask, evaluating each element.
  SmallVector<Constant*, 32> Result;
  for (unsigned i = 0; i != MaskNumElts; ++i) {
    Constant *InElt = GetVectorElement(Context, Mask, i);
    if (InElt == 0) return 0;

    if (isa<UndefValue>(InElt))
      InElt = UndefValue::get(EltTy);
    else if (ConstantInt *CI = dyn_cast<ConstantInt>(InElt)) {
      unsigned Elt = CI->getZExtValue();
      if (Elt >= SrcNumElts*2)
        InElt = UndefValue::get(EltTy);
      else if (Elt >= SrcNumElts)
        InElt = GetVectorElement(Context, V2, Elt - SrcNumElts);
      else
        InElt = GetVectorElement(Context, V1, Elt);
      if (InElt == 0) return 0;
    } else {
      // Unknown value.
      return 0;
    }
    Result.push_back(InElt);
  }

  return ConstantVector::get(&Result[0], Result.size());
}

Constant *llvm::ConstantFoldExtractValueInstruction(LLVMContext &Context,
                                                    const Constant *Agg,
                                                    const unsigned *Idxs,
                                                    unsigned NumIdx) {
  // Base case: no indices, so return the entire value.
  if (NumIdx == 0)
    return const_cast<Constant *>(Agg);

  if (isa<UndefValue>(Agg))  // ev(undef, x) -> undef
    return UndefValue::get(ExtractValueInst::getIndexedType(Agg->getType(),
                                                            Idxs,
                                                            Idxs + NumIdx));

  if (isa<ConstantAggregateZero>(Agg))  // ev(0, x) -> 0
    return
      Constant::getNullValue(ExtractValueInst::getIndexedType(Agg->getType(),
                                                              Idxs,
                                                              Idxs + NumIdx));

  // Otherwise recurse.
  return ConstantFoldExtractValueInstruction(Context, Agg->getOperand(*Idxs),
                                             Idxs+1, NumIdx-1);
}

Constant *llvm::ConstantFoldInsertValueInstruction(LLVMContext &Context,
                                                   const Constant *Agg,
                                                   const Constant *Val,
                                                   const unsigned *Idxs,
                                                   unsigned NumIdx) {
  // Base case: no indices, so replace the entire value.
  if (NumIdx == 0)
    return const_cast<Constant *>(Val);

  if (isa<UndefValue>(Agg)) {
    // Insertion of constant into aggregate undef
    // Optimize away insertion of undef.
    if (isa<UndefValue>(Val))
      return const_cast<Constant*>(Agg);
    
    // Otherwise break the aggregate undef into multiple undefs and do
    // the insertion.
    const CompositeType *AggTy = cast<CompositeType>(Agg->getType());
    unsigned numOps;
    if (const ArrayType *AR = dyn_cast<ArrayType>(AggTy))
      numOps = AR->getNumElements();
    else
      numOps = cast<StructType>(AggTy)->getNumElements();
    
    std::vector<Constant*> Ops(numOps); 
    for (unsigned i = 0; i < numOps; ++i) {
      const Type *MemberTy = AggTy->getTypeAtIndex(i);
      const Constant *Op =
        (*Idxs == i) ?
        ConstantFoldInsertValueInstruction(Context, UndefValue::get(MemberTy),
                                           Val, Idxs+1, NumIdx-1) :
        UndefValue::get(MemberTy);
      Ops[i] = const_cast<Constant*>(Op);
    }
    
    if (const StructType* ST = dyn_cast<StructType>(AggTy))
      return ConstantStruct::get(Context, Ops, ST->isPacked());
    return ConstantArray::get(cast<ArrayType>(AggTy), Ops);
  }
  
  if (isa<ConstantAggregateZero>(Agg)) {
    // Insertion of constant into aggregate zero
    // Optimize away insertion of zero.
    if (Val->isNullValue())
      return const_cast<Constant*>(Agg);
    
    // Otherwise break the aggregate zero into multiple zeros and do
    // the insertion.
    const CompositeType *AggTy = cast<CompositeType>(Agg->getType());
    unsigned numOps;
    if (const ArrayType *AR = dyn_cast<ArrayType>(AggTy))
      numOps = AR->getNumElements();
    else
      numOps = cast<StructType>(AggTy)->getNumElements();
    
    std::vector<Constant*> Ops(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Type *MemberTy = AggTy->getTypeAtIndex(i);
      const Constant *Op =
        (*Idxs == i) ?
        ConstantFoldInsertValueInstruction(Context, 
                                           Constant::getNullValue(MemberTy),
                                           Val, Idxs+1, NumIdx-1) :
        Constant::getNullValue(MemberTy);
      Ops[i] = const_cast<Constant*>(Op);
    }
    
    if (const StructType* ST = dyn_cast<StructType>(AggTy))
      return ConstantStruct::get(Context, Ops, ST->isPacked());
    return ConstantArray::get(cast<ArrayType>(AggTy), Ops);
  }
  
  if (isa<ConstantStruct>(Agg) || isa<ConstantArray>(Agg)) {
    // Insertion of constant into aggregate constant.
    std::vector<Constant*> Ops(Agg->getNumOperands());
    for (unsigned i = 0; i < Agg->getNumOperands(); ++i) {
      const Constant *Op =
        (*Idxs == i) ?
        ConstantFoldInsertValueInstruction(Context, Agg->getOperand(i),
                                           Val, Idxs+1, NumIdx-1) :
        Agg->getOperand(i);
      Ops[i] = const_cast<Constant*>(Op);
    }
    
    if (const StructType* ST = dyn_cast<StructType>(Agg->getType()))
      return ConstantStruct::get(Context, Ops, ST->isPacked());
    return ConstantArray::get(cast<ArrayType>(Agg->getType()), Ops);
  }

  return 0;
}


Constant *llvm::ConstantFoldBinaryInstruction(LLVMContext &Context,
                                              unsigned Opcode,
                                              const Constant *C1,
                                              const Constant *C2) {
  // No compile-time operations on this type yet.
  if (C1->getType() == Type::getPPC_FP128Ty(Context))
    return 0;

  // Handle UndefValue up front.
  if (isa<UndefValue>(C1) || isa<UndefValue>(C2)) {
    switch (Opcode) {
    case Instruction::Xor:
      if (isa<UndefValue>(C1) && isa<UndefValue>(C2))
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return Constant::getNullValue(C1->getType());
      // Fallthrough
    case Instruction::Add:
    case Instruction::Sub:
      return UndefValue::get(C1->getType());
    case Instruction::Mul:
    case Instruction::And:
      return Constant::getNullValue(C1->getType());
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      if (!isa<UndefValue>(C2))                    // undef / X -> 0
        return Constant::getNullValue(C1->getType());
      return const_cast<Constant*>(C2);            // X / undef -> undef
    case Instruction::Or:                          // X | undef -> -1
      if (const VectorType *PTy = dyn_cast<VectorType>(C1->getType()))
        return Constant::getAllOnesValue(PTy);
      return Constant::getAllOnesValue(C1->getType());
    case Instruction::LShr:
      if (isa<UndefValue>(C2) && isa<UndefValue>(C1))
        return const_cast<Constant*>(C1);           // undef lshr undef -> undef
      return Constant::getNullValue(C1->getType()); // X lshr undef -> 0
                                                    // undef lshr X -> 0
    case Instruction::AShr:
      if (!isa<UndefValue>(C2))
        return const_cast<Constant*>(C1);           // undef ashr X --> undef
      else if (isa<UndefValue>(C1)) 
        return const_cast<Constant*>(C1);           // undef ashr undef -> undef
      else
        return const_cast<Constant*>(C1);           // X ashr undef --> X
    case Instruction::Shl:
      // undef << X -> 0   or   X << undef -> 0
      return Constant::getNullValue(C1->getType());
    }
  }

  // Handle simplifications when the RHS is a constant int.
  if (const ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
    switch (Opcode) {
    case Instruction::Add:
      if (CI2->equalsInt(0)) return const_cast<Constant*>(C1);  // X + 0 == X
      break;
    case Instruction::Sub:
      if (CI2->equalsInt(0)) return const_cast<Constant*>(C1);  // X - 0 == X
      break;
    case Instruction::Mul:
      if (CI2->equalsInt(0)) return const_cast<Constant*>(C2);  // X * 0 == 0
      if (CI2->equalsInt(1))
        return const_cast<Constant*>(C1);                       // X * 1 == X
      break;
    case Instruction::UDiv:
    case Instruction::SDiv:
      if (CI2->equalsInt(1))
        return const_cast<Constant*>(C1);                     // X / 1 == X
      if (CI2->equalsInt(0))
        return UndefValue::get(CI2->getType());               // X / 0 == undef
      break;
    case Instruction::URem:
    case Instruction::SRem:
      if (CI2->equalsInt(1))
        return Constant::getNullValue(CI2->getType());        // X % 1 == 0
      if (CI2->equalsInt(0))
        return UndefValue::get(CI2->getType());               // X % 0 == undef
      break;
    case Instruction::And:
      if (CI2->isZero()) return const_cast<Constant*>(C2);    // X & 0 == 0
      if (CI2->isAllOnesValue())
        return const_cast<Constant*>(C1);                     // X & -1 == X
      
      if (const ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
        // (zext i32 to i64) & 4294967295 -> (zext i32 to i64)
        if (CE1->getOpcode() == Instruction::ZExt) {
          unsigned DstWidth = CI2->getType()->getBitWidth();
          unsigned SrcWidth =
            CE1->getOperand(0)->getType()->getPrimitiveSizeInBits();
          APInt PossiblySetBits(APInt::getLowBitsSet(DstWidth, SrcWidth));
          if ((PossiblySetBits & CI2->getValue()) == PossiblySetBits)
            return const_cast<Constant*>(C1);
        }
        
        // If and'ing the address of a global with a constant, fold it.
        if (CE1->getOpcode() == Instruction::PtrToInt && 
            isa<GlobalValue>(CE1->getOperand(0))) {
          GlobalValue *GV = cast<GlobalValue>(CE1->getOperand(0));
        
          // Functions are at least 4-byte aligned.
          unsigned GVAlign = GV->getAlignment();
          if (isa<Function>(GV))
            GVAlign = std::max(GVAlign, 4U);
          
          if (GVAlign > 1) {
            unsigned DstWidth = CI2->getType()->getBitWidth();
            unsigned SrcWidth = std::min(DstWidth, Log2_32(GVAlign));
            APInt BitsNotSet(APInt::getLowBitsSet(DstWidth, SrcWidth));

            // If checking bits we know are clear, return zero.
            if ((CI2->getValue() & BitsNotSet) == CI2->getValue())
              return Constant::getNullValue(CI2->getType());
          }
        }
      }
      break;
    case Instruction::Or:
      if (CI2->equalsInt(0)) return const_cast<Constant*>(C1);  // X | 0 == X
      if (CI2->isAllOnesValue())
        return const_cast<Constant*>(C2);  // X | -1 == -1
      break;
    case Instruction::Xor:
      if (CI2->equalsInt(0)) return const_cast<Constant*>(C1);  // X ^ 0 == X
      break;
    case Instruction::AShr:
      // ashr (zext C to Ty), C2 -> lshr (zext C, CSA), C2
      if (const ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1))
        if (CE1->getOpcode() == Instruction::ZExt)  // Top bits known zero.
          return ConstantExpr::getLShr(const_cast<Constant*>(C1),
                                             const_cast<Constant*>(C2));
      break;
    }
  }
  
  // At this point we know neither constant is an UndefValue.
  if (const ConstantInt *CI1 = dyn_cast<ConstantInt>(C1)) {
    if (const ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
      using namespace APIntOps;
      const APInt &C1V = CI1->getValue();
      const APInt &C2V = CI2->getValue();
      switch (Opcode) {
      default:
        break;
      case Instruction::Add:     
        return ConstantInt::get(Context, C1V + C2V);
      case Instruction::Sub:     
        return ConstantInt::get(Context, C1V - C2V);
      case Instruction::Mul:     
        return ConstantInt::get(Context, C1V * C2V);
      case Instruction::UDiv:
        assert(!CI2->isNullValue() && "Div by zero handled above");
        return ConstantInt::get(Context, C1V.udiv(C2V));
      case Instruction::SDiv:
        assert(!CI2->isNullValue() && "Div by zero handled above");
        if (C2V.isAllOnesValue() && C1V.isMinSignedValue())
          return UndefValue::get(CI1->getType());   // MIN_INT / -1 -> undef
        return ConstantInt::get(Context, C1V.sdiv(C2V));
      case Instruction::URem:
        assert(!CI2->isNullValue() && "Div by zero handled above");
        return ConstantInt::get(Context, C1V.urem(C2V));
      case Instruction::SRem:
        assert(!CI2->isNullValue() && "Div by zero handled above");
        if (C2V.isAllOnesValue() && C1V.isMinSignedValue())
          return UndefValue::get(CI1->getType());   // MIN_INT % -1 -> undef
        return ConstantInt::get(Context, C1V.srem(C2V));
      case Instruction::And:
        return ConstantInt::get(Context, C1V & C2V);
      case Instruction::Or:
        return ConstantInt::get(Context, C1V | C2V);
      case Instruction::Xor:
        return ConstantInt::get(Context, C1V ^ C2V);
      case Instruction::Shl: {
        uint32_t shiftAmt = C2V.getZExtValue();
        if (shiftAmt < C1V.getBitWidth())
          return ConstantInt::get(Context, C1V.shl(shiftAmt));
        else
          return UndefValue::get(C1->getType()); // too big shift is undef
      }
      case Instruction::LShr: {
        uint32_t shiftAmt = C2V.getZExtValue();
        if (shiftAmt < C1V.getBitWidth())
          return ConstantInt::get(Context, C1V.lshr(shiftAmt));
        else
          return UndefValue::get(C1->getType()); // too big shift is undef
      }
      case Instruction::AShr: {
        uint32_t shiftAmt = C2V.getZExtValue();
        if (shiftAmt < C1V.getBitWidth())
          return ConstantInt::get(Context, C1V.ashr(shiftAmt));
        else
          return UndefValue::get(C1->getType()); // too big shift is undef
      }
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
      if (CI1->equalsInt(0)) return const_cast<Constant*>(C1);
      break;
    default:
      break;
    }
  } else if (const ConstantFP *CFP1 = dyn_cast<ConstantFP>(C1)) {
    if (const ConstantFP *CFP2 = dyn_cast<ConstantFP>(C2)) {
      APFloat C1V = CFP1->getValueAPF();
      APFloat C2V = CFP2->getValueAPF();
      APFloat C3V = C1V;  // copy for modification
      switch (Opcode) {
      default:                   
        break;
      case Instruction::FAdd:
        (void)C3V.add(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(Context, C3V);
      case Instruction::FSub:
        (void)C3V.subtract(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(Context, C3V);
      case Instruction::FMul:
        (void)C3V.multiply(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(Context, C3V);
      case Instruction::FDiv:
        (void)C3V.divide(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(Context, C3V);
      case Instruction::FRem:
        (void)C3V.mod(C2V, APFloat::rmNearestTiesToEven);
        return ConstantFP::get(Context, C3V);
      }
    }
  } else if (const VectorType *VTy = dyn_cast<VectorType>(C1->getType())) {
    const ConstantVector *CP1 = dyn_cast<ConstantVector>(C1);
    const ConstantVector *CP2 = dyn_cast<ConstantVector>(C2);
    if ((CP1 != NULL || isa<ConstantAggregateZero>(C1)) &&
        (CP2 != NULL || isa<ConstantAggregateZero>(C2))) {
      std::vector<Constant*> Res;
      const Type* EltTy = VTy->getElementType();  
      const Constant *C1 = 0;
      const Constant *C2 = 0;
      switch (Opcode) {
      default:
        break;
      case Instruction::Add:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getAdd(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::FAdd:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getFAdd(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::Sub:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getSub(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::FSub:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getFSub(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::Mul:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getMul(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::FMul:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getFMul(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::UDiv:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getUDiv(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::SDiv:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getSDiv(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::FDiv:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getFDiv(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::URem:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getURem(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::SRem:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getSRem(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::FRem:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getFRem(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::And: 
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getAnd(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::Or:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getOr(const_cast<Constant*>(C1),
                                                  const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::Xor:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getXor(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::LShr:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getLShr(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::AShr:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getAShr(const_cast<Constant*>(C1),
                                                    const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      case Instruction::Shl:
        for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
          C1 = CP1 ? CP1->getOperand(i) : Constant::getNullValue(EltTy);
          C2 = CP2 ? CP2->getOperand(i) : Constant::getNullValue(EltTy);
          Res.push_back(ConstantExpr::getShl(const_cast<Constant*>(C1),
                                                   const_cast<Constant*>(C2)));
        }
        return ConstantVector::get(Res);
      }
    }
  }

  if (isa<ConstantExpr>(C1)) {
    // There are many possible foldings we could do here.  We should probably
    // at least fold add of a pointer with an integer into the appropriate
    // getelementptr.  This will improve alias analysis a bit.
  } else if (isa<ConstantExpr>(C2)) {
    // If C2 is a constant expr and C1 isn't, flop them around and fold the
    // other way if possible.
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      // No change of opcode required.
      return ConstantFoldBinaryInstruction(Context, Opcode, C2, C1);
      
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    default:  // These instructions cannot be flopped around.
      break;
    }
  }
  
  // We don't know how to fold this.
  return 0;
}

/// isZeroSizedType - This type is zero sized if its an array or structure of
/// zero sized types.  The only leaf zero sized type is an empty structure.
static bool isMaybeZeroSizedType(const Type *Ty) {
  if (isa<OpaqueType>(Ty)) return true;  // Can't say.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {

    // If all of elements have zero size, this does too.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      if (!isMaybeZeroSizedType(STy->getElementType(i))) return false;
    return true;

  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return isMaybeZeroSizedType(ATy->getElementType());
  }
  return false;
}

/// IdxCompare - Compare the two constants as though they were getelementptr
/// indices.  This allows coersion of the types to be the same thing.
///
/// If the two constants are the "same" (after coersion), return 0.  If the
/// first is less than the second, return -1, if the second is less than the
/// first, return 1.  If the constants are not integral, return -2.
///
static int IdxCompare(LLVMContext &Context, Constant *C1, Constant *C2, 
                      const Type *ElTy) {
  if (C1 == C2) return 0;

  // Ok, we found a different index.  If they are not ConstantInt, we can't do
  // anything with them.
  if (!isa<ConstantInt>(C1) || !isa<ConstantInt>(C2))
    return -2; // don't know!

  // Ok, we have two differing integer indices.  Sign extend them to be the same
  // type.  Long is always big enough, so we use it.
  if (C1->getType() != Type::getInt64Ty(Context))
    C1 = ConstantExpr::getSExt(C1, Type::getInt64Ty(Context));

  if (C2->getType() != Type::getInt64Ty(Context))
    C2 = ConstantExpr::getSExt(C2, Type::getInt64Ty(Context));

  if (C1 == C2) return 0;  // They are equal

  // If the type being indexed over is really just a zero sized type, there is
  // no pointer difference being made here.
  if (isMaybeZeroSizedType(ElTy))
    return -2; // dunno.

  // If they are really different, now that they are the same type, then we
  // found a difference!
  if (cast<ConstantInt>(C1)->getSExtValue() < 
      cast<ConstantInt>(C2)->getSExtValue())
    return -1;
  else
    return 1;
}

/// evaluateFCmpRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like ConstantFP comparisons, but should instead handle ConstantExprs.
/// If we can determine that the two constants have a particular relation to 
/// each other, we should return the corresponding FCmpInst predicate, 
/// otherwise return FCmpInst::BAD_FCMP_PREDICATE. This is used below in
/// ConstantFoldCompareInstruction.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider ConstantFP
/// to be the simplest, and ConstantExprs to be the most complex.
static FCmpInst::Predicate evaluateFCmpRelation(LLVMContext &Context,
                                                const Constant *V1, 
                                                const Constant *V2) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare values of different types!");

  // No compile-time operations on this type yet.
  if (V1->getType() == Type::getPPC_FP128Ty(Context))
    return FCmpInst::BAD_FCMP_PREDICATE;

  // Handle degenerate case quickly
  if (V1 == V2) return FCmpInst::FCMP_OEQ;

  if (!isa<ConstantExpr>(V1)) {
    if (!isa<ConstantExpr>(V2)) {
      // We distilled thisUse the standard constant folder for a few cases
      ConstantInt *R = 0;
      Constant *C1 = const_cast<Constant*>(V1);
      Constant *C2 = const_cast<Constant*>(V2);
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OEQ, C1, C2));
      if (R && !R->isZero()) 
        return FCmpInst::FCMP_OEQ;
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OLT, C1, C2));
      if (R && !R->isZero()) 
        return FCmpInst::FCMP_OLT;
      R = dyn_cast<ConstantInt>(
                      ConstantExpr::getFCmp(FCmpInst::FCMP_OGT, C1, C2));
      if (R && !R->isZero()) 
        return FCmpInst::FCMP_OGT;

      // Nothing more we can do
      return FCmpInst::BAD_FCMP_PREDICATE;
    }
    
    // If the first operand is simple and second is ConstantExpr, swap operands.
    FCmpInst::Predicate SwappedRelation = evaluateFCmpRelation(Context, V2, V1);
    if (SwappedRelation != FCmpInst::BAD_FCMP_PREDICATE)
      return FCmpInst::getSwappedPredicate(SwappedRelation);
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr or a simple constant.
    const ConstantExpr *CE1 = cast<ConstantExpr>(V1);
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

/// evaluateICmpRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like integer comparisons, but should instead handle ConstantExprs
/// and GlobalValues.  If we can determine that the two constants have a
/// particular relation to each other, we should return the corresponding ICmp
/// predicate, otherwise return ICmpInst::BAD_ICMP_PREDICATE.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider simple
/// constants (like ConstantInt) to be the simplest, followed by
/// GlobalValues, followed by ConstantExpr's (the most complex).
///
static ICmpInst::Predicate evaluateICmpRelation(LLVMContext &Context,
                                                const Constant *V1, 
                                                const Constant *V2,
                                                bool isSigned) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare different types of values!");
  if (V1 == V2) return ICmpInst::ICMP_EQ;

  if (!isa<ConstantExpr>(V1) && !isa<GlobalValue>(V1)) {
    if (!isa<GlobalValue>(V2) && !isa<ConstantExpr>(V2)) {
      // We distilled this down to a simple case, use the standard constant
      // folder.
      ConstantInt *R = 0;
      Constant *C1 = const_cast<Constant*>(V1);
      Constant *C2 = const_cast<Constant*>(V2);
      ICmpInst::Predicate pred = ICmpInst::ICMP_EQ;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && !R->isZero()) 
        return pred;
      pred = isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && !R->isZero())
        return pred;
      pred = isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
      R = dyn_cast<ConstantInt>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && !R->isZero())
        return pred;
      
      // If we couldn't figure it out, bail.
      return ICmpInst::BAD_ICMP_PREDICATE;
    }
    
    // If the first operand is simple, swap operands.
    ICmpInst::Predicate SwappedRelation = 
      evaluateICmpRelation(Context, V2, V1, isSigned);
    if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
      return ICmpInst::getSwappedPredicate(SwappedRelation);

  } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(V1)) {
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
      ICmpInst::Predicate SwappedRelation = 
        evaluateICmpRelation(Context, V2, V1, isSigned);
      if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
        return ICmpInst::getSwappedPredicate(SwappedRelation);
      else
        return ICmpInst::BAD_ICMP_PREDICATE;
    }

    // Now we know that the RHS is a GlobalValue or simple constant,
    // which (since the types must match) means that it's a ConstantPointerNull.
    if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
      // Don't try to decide equality of aliases.
      if (!isa<GlobalAlias>(CPR1) && !isa<GlobalAlias>(CPR2))
        if (!CPR1->hasExternalWeakLinkage() || !CPR2->hasExternalWeakLinkage())
          return ICmpInst::ICMP_NE;
    } else {
      assert(isa<ConstantPointerNull>(V2) && "Canonicalization guarantee!");
      // GlobalVals can never be null.  Don't try to evaluate aliases.
      if (!CPR1->hasExternalWeakLinkage() && !isa<GlobalAlias>(CPR1))
        return ICmpInst::ICMP_NE;
    }
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr, a CPR, or a simple constant.
    const ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    const Constant *CE1Op0 = CE1->getOperand(0);

    switch (CE1->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      break; // We can't evaluate floating point casts or truncations.

    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::BitCast:
    case Instruction::ZExt:
    case Instruction::SExt:
      // If the cast is not actually changing bits, and the second operand is a
      // null pointer, do the comparison with the pre-casted value.
      if (V2->isNullValue() &&
          (isa<PointerType>(CE1->getType()) || CE1->getType()->isInteger())) {
        bool sgnd = isSigned;
        if (CE1->getOpcode() == Instruction::ZExt) isSigned = false;
        if (CE1->getOpcode() == Instruction::SExt) isSigned = true;
        return evaluateICmpRelation(Context, CE1Op0,
                                    Constant::getNullValue(CE1Op0->getType()), 
                                    sgnd);
      }

      // If the dest type is a pointer type, and the RHS is a constantexpr cast
      // from the same type as the src of the LHS, evaluate the inputs.  This is
      // important for things like "icmp eq (cast 4 to int*), (cast 5 to int*)",
      // which happens a lot in compilers with tagged integers.
      if (const ConstantExpr *CE2 = dyn_cast<ConstantExpr>(V2))
        if (CE2->isCast() && isa<PointerType>(CE1->getType()) &&
            CE1->getOperand(0)->getType() == CE2->getOperand(0)->getType() &&
            CE1->getOperand(0)->getType()->isInteger()) {
          bool sgnd = isSigned;
          if (CE1->getOpcode() == Instruction::ZExt) isSigned = false;
          if (CE1->getOpcode() == Instruction::SExt) isSigned = true;
          return evaluateICmpRelation(Context, CE1->getOperand(0), 
                                      CE2->getOperand(0), sgnd);
        }
      break;

    case Instruction::GetElementPtr:
      // Ok, since this is a getelementptr, we know that the constant has a
      // pointer type.  Check the various cases.
      if (isa<ConstantPointerNull>(V2)) {
        // If we are comparing a GEP to a null pointer, check to see if the base
        // of the GEP equals the null pointer.
        if (const GlobalValue *GV = dyn_cast<GlobalValue>(CE1Op0)) {
          if (GV->hasExternalWeakLinkage())
            // Weak linkage GVals could be zero or not. We're comparing that
            // to null pointer so its greater-or-equal
            return isSigned ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
          else 
            // If its not weak linkage, the GVal must have a non-zero address
            // so the result is greater-than
            return isSigned ? ICmpInst::ICMP_SGT :  ICmpInst::ICMP_UGT;
        } else if (isa<ConstantPointerNull>(CE1Op0)) {
          // If we are indexing from a null pointer, check to see if we have any
          // non-zero indices.
          for (unsigned i = 1, e = CE1->getNumOperands(); i != e; ++i)
            if (!CE1->getOperand(i)->isNullValue())
              // Offsetting from null, must not be equal.
              return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
          // Only zero indexes from null, must still be zero.
          return ICmpInst::ICMP_EQ;
        }
        // Otherwise, we can't really say if the first operand is null or not.
      } else if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
        if (isa<ConstantPointerNull>(CE1Op0)) {
          if (CPR2->hasExternalWeakLinkage())
            // Weak linkage GVals could be zero or not. We're comparing it to
            // a null pointer, so its less-or-equal
            return isSigned ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE;
          else
            // If its not weak linkage, the GVal must have a non-zero address
            // so the result is less-than
            return isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
        } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(CE1Op0)) {
          if (CPR1 == CPR2) {
            // If this is a getelementptr of the same global, then it must be
            // different.  Because the types must match, the getelementptr could
            // only have at most one index, and because we fold getelementptr's
            // with a single zero index, it must be nonzero.
            assert(CE1->getNumOperands() == 2 &&
                   !CE1->getOperand(1)->isNullValue() &&
                   "Suprising getelementptr!");
            return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
          } else {
            // If they are different globals, we don't know what the value is,
            // but they can't be equal.
            return ICmpInst::ICMP_NE;
          }
        }
      } else {
        const ConstantExpr *CE2 = cast<ConstantExpr>(V2);
        const Constant *CE2Op0 = CE2->getOperand(0);

        // There are MANY other foldings that we could perform here.  They will
        // probably be added on demand, as they seem needed.
        switch (CE2->getOpcode()) {
        default: break;
        case Instruction::GetElementPtr:
          // By far the most common case to handle is when the base pointers are
          // obviously to the same or different globals.
          if (isa<GlobalValue>(CE1Op0) && isa<GlobalValue>(CE2Op0)) {
            if (CE1Op0 != CE2Op0) // Don't know relative ordering, but not equal
              return ICmpInst::ICMP_NE;
            // Ok, we know that both getelementptr instructions are based on the
            // same global.  From this, we can precisely determine the relative
            // ordering of the resultant pointers.
            unsigned i = 1;

            // Compare all of the operands the GEP's have in common.
            gep_type_iterator GTI = gep_type_begin(CE1);
            for (;i != CE1->getNumOperands() && i != CE2->getNumOperands();
                 ++i, ++GTI)
              switch (IdxCompare(Context, CE1->getOperand(i),
                                 CE2->getOperand(i), GTI.getIndexedType())) {
              case -1: return isSigned ? ICmpInst::ICMP_SLT:ICmpInst::ICMP_ULT;
              case 1:  return isSigned ? ICmpInst::ICMP_SGT:ICmpInst::ICMP_UGT;
              case -2: return ICmpInst::BAD_ICMP_PREDICATE;
              }

            // Ok, we ran out of things they have in common.  If any leftovers
            // are non-zero then we have a difference, otherwise we are equal.
            for (; i < CE1->getNumOperands(); ++i)
              if (!CE1->getOperand(i)->isNullValue()) {
                if (isa<ConstantInt>(CE1->getOperand(i)))
                  return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
                else
                  return ICmpInst::BAD_ICMP_PREDICATE; // Might be equal.
              }

            for (; i < CE2->getNumOperands(); ++i)
              if (!CE2->getOperand(i)->isNullValue()) {
                if (isa<ConstantInt>(CE2->getOperand(i)))
                  return isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
                else
                  return ICmpInst::BAD_ICMP_PREDICATE; // Might be equal.
              }
            return ICmpInst::ICMP_EQ;
          }
        }
      }
    default:
      break;
    }
  }

  return ICmpInst::BAD_ICMP_PREDICATE;
}

Constant *llvm::ConstantFoldCompareInstruction(LLVMContext &Context,
                                               unsigned short pred, 
                                               const Constant *C1, 
                                               const Constant *C2) {
  const Type *ResultTy;
  if (const VectorType *VT = dyn_cast<VectorType>(C1->getType()))
    ResultTy = VectorType::get(Type::getInt1Ty(Context), VT->getNumElements());
  else
    ResultTy = Type::getInt1Ty(Context);

  // Fold FCMP_FALSE/FCMP_TRUE unconditionally.
  if (pred == FCmpInst::FCMP_FALSE)
    return Constant::getNullValue(ResultTy);

  if (pred == FCmpInst::FCMP_TRUE)
    return Constant::getAllOnesValue(ResultTy);

  // Handle some degenerate cases first
  if (isa<UndefValue>(C1) || isa<UndefValue>(C2))
    return UndefValue::get(ResultTy);

  // No compile-time operations on this type yet.
  if (C1->getType() == Type::getPPC_FP128Ty(Context))
    return 0;

  // icmp eq/ne(null,GV) -> false/true
  if (C1->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C2))
      // Don't try to evaluate aliases.  External weak GV can be null.
      if (!isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage()) {
        if (pred == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(Context);
        else if (pred == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(Context);
      }
  // icmp eq/ne(GV,null) -> false/true
  } else if (C2->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C1))
      // Don't try to evaluate aliases.  External weak GV can be null.
      if (!isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage()) {
        if (pred == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(Context);
        else if (pred == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(Context);
      }
  }

  if (isa<ConstantInt>(C1) && isa<ConstantInt>(C2)) {
    APInt V1 = cast<ConstantInt>(C1)->getValue();
    APInt V2 = cast<ConstantInt>(C2)->getValue();
    switch (pred) {
    default: llvm_unreachable("Invalid ICmp Predicate"); return 0;
    case ICmpInst::ICMP_EQ:
      return ConstantInt::get(Type::getInt1Ty(Context), V1 == V2);
    case ICmpInst::ICMP_NE: 
      return ConstantInt::get(Type::getInt1Ty(Context), V1 != V2);
    case ICmpInst::ICMP_SLT:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.slt(V2));
    case ICmpInst::ICMP_SGT:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.sgt(V2));
    case ICmpInst::ICMP_SLE:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.sle(V2));
    case ICmpInst::ICMP_SGE:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.sge(V2));
    case ICmpInst::ICMP_ULT:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.ult(V2));
    case ICmpInst::ICMP_UGT:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.ugt(V2));
    case ICmpInst::ICMP_ULE:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.ule(V2));
    case ICmpInst::ICMP_UGE:
      return ConstantInt::get(Type::getInt1Ty(Context), V1.uge(V2));
    }
  } else if (isa<ConstantFP>(C1) && isa<ConstantFP>(C2)) {
    APFloat C1V = cast<ConstantFP>(C1)->getValueAPF();
    APFloat C2V = cast<ConstantFP>(C2)->getValueAPF();
    APFloat::cmpResult R = C1V.compare(C2V);
    switch (pred) {
    default: llvm_unreachable("Invalid FCmp Predicate"); return 0;
    case FCmpInst::FCMP_FALSE: return ConstantInt::getFalse(Context);
    case FCmpInst::FCMP_TRUE:  return ConstantInt::getTrue(Context);
    case FCmpInst::FCMP_UNO:
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpUnordered);
    case FCmpInst::FCMP_ORD:
      return ConstantInt::get(Type::getInt1Ty(Context), R!=APFloat::cmpUnordered);
    case FCmpInst::FCMP_UEQ:
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpUnordered ||
                                            R==APFloat::cmpEqual);
    case FCmpInst::FCMP_OEQ:   
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpEqual);
    case FCmpInst::FCMP_UNE:
      return ConstantInt::get(Type::getInt1Ty(Context), R!=APFloat::cmpEqual);
    case FCmpInst::FCMP_ONE:   
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpLessThan ||
                                            R==APFloat::cmpGreaterThan);
    case FCmpInst::FCMP_ULT: 
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpUnordered ||
                                            R==APFloat::cmpLessThan);
    case FCmpInst::FCMP_OLT:   
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpLessThan);
    case FCmpInst::FCMP_UGT:
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpUnordered ||
                                            R==APFloat::cmpGreaterThan);
    case FCmpInst::FCMP_OGT:
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpGreaterThan);
    case FCmpInst::FCMP_ULE:
      return ConstantInt::get(Type::getInt1Ty(Context), R!=APFloat::cmpGreaterThan);
    case FCmpInst::FCMP_OLE: 
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpLessThan ||
                                            R==APFloat::cmpEqual);
    case FCmpInst::FCMP_UGE:
      return ConstantInt::get(Type::getInt1Ty(Context), R!=APFloat::cmpLessThan);
    case FCmpInst::FCMP_OGE: 
      return ConstantInt::get(Type::getInt1Ty(Context), R==APFloat::cmpGreaterThan ||
                                            R==APFloat::cmpEqual);
    }
  } else if (isa<VectorType>(C1->getType())) {
    SmallVector<Constant*, 16> C1Elts, C2Elts;
    C1->getVectorElements(Context, C1Elts);
    C2->getVectorElements(Context, C2Elts);
    
    // If we can constant fold the comparison of each element, constant fold
    // the whole vector comparison.
    SmallVector<Constant*, 4> ResElts;
    for (unsigned i = 0, e = C1Elts.size(); i != e; ++i) {
      // Compare the elements, producing an i1 result or constant expr.
      ResElts.push_back(
                    ConstantExpr::getCompare(pred, C1Elts[i], C2Elts[i]));
    }
    return ConstantVector::get(&ResElts[0], ResElts.size());
  }

  if (C1->getType()->isFloatingPoint()) {
    int Result = -1;  // -1 = unknown, 0 = known false, 1 = known true.
    switch (evaluateFCmpRelation(Context, C1, C2)) {
    default: llvm_unreachable("Unknown relation!");
    case FCmpInst::FCMP_UNO:
    case FCmpInst::FCMP_ORD:
    case FCmpInst::FCMP_UEQ:
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
      Result = (pred == FCmpInst::FCMP_UEQ || pred == FCmpInst::FCMP_OEQ ||
                pred == FCmpInst::FCMP_ULE || pred == FCmpInst::FCMP_OLE ||
                pred == FCmpInst::FCMP_UGE || pred == FCmpInst::FCMP_OGE);
      break;
    case FCmpInst::FCMP_OLT: // We know that C1 < C2
      Result = (pred == FCmpInst::FCMP_UNE || pred == FCmpInst::FCMP_ONE ||
                pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT ||
                pred == FCmpInst::FCMP_ULE || pred == FCmpInst::FCMP_OLE);
      break;
    case FCmpInst::FCMP_OGT: // We know that C1 > C2
      Result = (pred == FCmpInst::FCMP_UNE || pred == FCmpInst::FCMP_ONE ||
                pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT ||
                pred == FCmpInst::FCMP_UGE || pred == FCmpInst::FCMP_OGE);
      break;
    case FCmpInst::FCMP_OLE: // We know that C1 <= C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT) 
        Result = 0;
      else if (pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT) 
        Result = 1;
      break;
    case FCmpInst::FCMP_OGE: // We known that C1 >= C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT) 
        Result = 0;
      else if (pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT) 
        Result = 1;
      break;
    case ICmpInst::ICMP_NE: // We know that C1 != C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_OEQ || pred == FCmpInst::FCMP_UEQ) 
        Result = 0;
      else if (pred == FCmpInst::FCMP_ONE || pred == FCmpInst::FCMP_UNE) 
        Result = 1;
      break;
    }
    
    // If we evaluated the result, return it now.
    if (Result != -1)
      return ConstantInt::get(Type::getInt1Ty(Context), Result);

  } else {
    // Evaluate the relation between the two constants, per the predicate.
    int Result = -1;  // -1 = unknown, 0 = known false, 1 = known true.
    switch (evaluateICmpRelation(Context, C1, C2, CmpInst::isSigned(pred))) {
    default: llvm_unreachable("Unknown relational!");
    case ICmpInst::BAD_ICMP_PREDICATE:
      break;  // Couldn't determine anything about these constants.
    case ICmpInst::ICMP_EQ:   // We know the constants are equal!
      // If we know the constants are equal, we can decide the result of this
      // computation precisely.
      Result = (pred == ICmpInst::ICMP_EQ  ||
                pred == ICmpInst::ICMP_ULE ||
                pred == ICmpInst::ICMP_SLE ||
                pred == ICmpInst::ICMP_UGE ||
                pred == ICmpInst::ICMP_SGE);
      break;
    case ICmpInst::ICMP_ULT:
      // If we know that C1 < C2, we can decide the result of this computation
      // precisely.
      Result = (pred == ICmpInst::ICMP_ULT ||
                pred == ICmpInst::ICMP_NE  ||
                pred == ICmpInst::ICMP_ULE);
      break;
    case ICmpInst::ICMP_SLT:
      // If we know that C1 < C2, we can decide the result of this computation
      // precisely.
      Result = (pred == ICmpInst::ICMP_SLT ||
                pred == ICmpInst::ICMP_NE  ||
                pred == ICmpInst::ICMP_SLE);
      break;
    case ICmpInst::ICMP_UGT:
      // If we know that C1 > C2, we can decide the result of this computation
      // precisely.
      Result = (pred == ICmpInst::ICMP_UGT ||
                pred == ICmpInst::ICMP_NE  ||
                pred == ICmpInst::ICMP_UGE);
      break;
    case ICmpInst::ICMP_SGT:
      // If we know that C1 > C2, we can decide the result of this computation
      // precisely.
      Result = (pred == ICmpInst::ICMP_SGT ||
                pred == ICmpInst::ICMP_NE  ||
                pred == ICmpInst::ICMP_SGE);
      break;
    case ICmpInst::ICMP_ULE:
      // If we know that C1 <= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_UGT) Result = 0;
      if (pred == ICmpInst::ICMP_ULT) Result = 1;
      break;
    case ICmpInst::ICMP_SLE:
      // If we know that C1 <= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_SGT) Result = 0;
      if (pred == ICmpInst::ICMP_SLT) Result = 1;
      break;

    case ICmpInst::ICMP_UGE:
      // If we know that C1 >= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_ULT) Result = 0;
      if (pred == ICmpInst::ICMP_UGT) Result = 1;
      break;
    case ICmpInst::ICMP_SGE:
      // If we know that C1 >= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_SLT) Result = 0;
      if (pred == ICmpInst::ICMP_SGT) Result = 1;
      break;

    case ICmpInst::ICMP_NE:
      // If we know that C1 != C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_EQ) Result = 0;
      if (pred == ICmpInst::ICMP_NE) Result = 1;
      break;
    }
    
    // If we evaluated the result, return it now.
    if (Result != -1)
      return ConstantInt::get(Type::getInt1Ty(Context), Result);
    
    if (!isa<ConstantExpr>(C1) && isa<ConstantExpr>(C2)) {
      // If C2 is a constant expr and C1 isn't, flip them around and fold the
      // other way if possible.
      switch (pred) {
      case ICmpInst::ICMP_EQ:
      case ICmpInst::ICMP_NE:
        // No change of predicate required.
        return ConstantFoldCompareInstruction(Context, pred, C2, C1);

      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_ULE:
      case ICmpInst::ICMP_SLE:
      case ICmpInst::ICMP_UGE:
      case ICmpInst::ICMP_SGE:
        // Change the predicate as necessary to swap the operands.
        pred = ICmpInst::getSwappedPredicate((ICmpInst::Predicate)pred);
        return ConstantFoldCompareInstruction(Context, pred, C2, C1);

      default:  // These predicates cannot be flopped around.
        break;
      }
    }
  }
  return 0;
  }

Constant *llvm::ConstantFoldGetElementPtr(LLVMContext &Context, 
                                          const Constant *C,
                                          Constant* const *Idxs,
                                          unsigned NumIdx) {
  if (NumIdx == 0 ||
      (NumIdx == 1 && Idxs[0]->isNullValue()))
    return const_cast<Constant*>(C);

  if (isa<UndefValue>(C)) {
    const PointerType *Ptr = cast<PointerType>(C->getType());
    const Type *Ty = GetElementPtrInst::getIndexedType(Ptr,
                                                       (Value **)Idxs,
                                                       (Value **)Idxs+NumIdx);
    assert(Ty != 0 && "Invalid indices for GEP!");
    return UndefValue::get(PointerType::get(Ty, Ptr->getAddressSpace()));
  }

  Constant *Idx0 = Idxs[0];
  if (C->isNullValue()) {
    bool isNull = true;
    for (unsigned i = 0, e = NumIdx; i != e; ++i)
      if (!Idxs[i]->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) {
      const PointerType *Ptr = cast<PointerType>(C->getType());
      const Type *Ty = GetElementPtrInst::getIndexedType(Ptr,
                                                         (Value**)Idxs,
                                                         (Value**)Idxs+NumIdx);
      assert(Ty != 0 && "Invalid indices for GEP!");
      return  ConstantPointerNull::get(
                            PointerType::get(Ty,Ptr->getAddressSpace()));
    }
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(const_cast<Constant*>(C))) {
    // Combine Indices - If the source pointer to this getelementptr instruction
    // is a getelementptr instruction, combine the indices of the two
    // getelementptr instructions into a single instruction.
    //
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      const Type *LastTy = 0;
      for (gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
           I != E; ++I)
        LastTy = *I;

      if ((LastTy && isa<ArrayType>(LastTy)) || Idx0->isNullValue()) {
        SmallVector<Value*, 16> NewIndices;
        NewIndices.reserve(NumIdx + CE->getNumOperands());
        for (unsigned i = 1, e = CE->getNumOperands()-1; i != e; ++i)
          NewIndices.push_back(CE->getOperand(i));

        // Add the last index of the source with the first index of the new GEP.
        // Make sure to handle the case when they are actually different types.
        Constant *Combined = CE->getOperand(CE->getNumOperands()-1);
        // Otherwise it must be an array.
        if (!Idx0->isNullValue()) {
          const Type *IdxTy = Combined->getType();
          if (IdxTy != Idx0->getType()) {
            Constant *C1 =
              ConstantExpr::getSExtOrBitCast(Idx0, Type::getInt64Ty(Context));
            Constant *C2 = ConstantExpr::getSExtOrBitCast(Combined, 
                                                          Type::getInt64Ty(Context));
            Combined = ConstantExpr::get(Instruction::Add, C1, C2);
          } else {
            Combined =
              ConstantExpr::get(Instruction::Add, Idx0, Combined);
          }
        }

        NewIndices.push_back(Combined);
        NewIndices.insert(NewIndices.end(), Idxs+1, Idxs+NumIdx);
        return ConstantExpr::getGetElementPtr(CE->getOperand(0),
                                              &NewIndices[0],
                                              NewIndices.size());
      }
    }

    // Implement folding of:
    //    int* getelementptr ([2 x int]* cast ([3 x int]* %X to [2 x int]*),
    //                        long 0, long 0)
    // To: int* getelementptr ([3 x int]* %X, long 0, long 0)
    //
    if (CE->isCast() && NumIdx > 1 && Idx0->isNullValue()) {
      if (const PointerType *SPT =
          dyn_cast<PointerType>(CE->getOperand(0)->getType()))
        if (const ArrayType *SAT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (const ArrayType *CAT =
        dyn_cast<ArrayType>(cast<PointerType>(C->getType())->getElementType()))
            if (CAT->getElementType() == SAT->getElementType())
              return ConstantExpr::getGetElementPtr(
                      (Constant*)CE->getOperand(0), Idxs, NumIdx);
    }
    
    // Fold: getelementptr (i8* inttoptr (i64 1 to i8*), i32 -1)
    // Into: inttoptr (i64 0 to i8*)
    // This happens with pointers to member functions in C++.
    if (CE->getOpcode() == Instruction::IntToPtr && NumIdx == 1 &&
        isa<ConstantInt>(CE->getOperand(0)) && isa<ConstantInt>(Idxs[0]) &&
        cast<PointerType>(CE->getType())->getElementType() == Type::getInt8Ty(Context)) {
      Constant *Base = CE->getOperand(0);
      Constant *Offset = Idxs[0];
      
      // Convert the smaller integer to the larger type.
      if (Offset->getType()->getPrimitiveSizeInBits() < 
          Base->getType()->getPrimitiveSizeInBits())
        Offset = ConstantExpr::getSExt(Offset, Base->getType());
      else if (Base->getType()->getPrimitiveSizeInBits() <
               Offset->getType()->getPrimitiveSizeInBits())
        Base = ConstantExpr::getZExt(Base, Offset->getType());
      
      Base = ConstantExpr::getAdd(Base, Offset);
      return ConstantExpr::getIntToPtr(Base, CE->getType());
    }
  }
  return 0;
}

