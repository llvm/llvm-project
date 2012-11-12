//===-- ConstantFolding.cpp - Fold instructions into constants ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for folding instructions into constants.
//
// Also, to supplement the basic VMCore ConstantExpr simplifications,
// this file defines some additional folding routines that can make use of
// DataLayout information. These functions cannot go in VMCore due to library
// dependency issues.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Operator.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FEnv.h"
#include <cerrno>
#include <cmath>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Constant Folding internal helper functions
//===----------------------------------------------------------------------===//

/// FoldBitCast - Constant fold bitcast, symbolically evaluating it with
/// DataLayout.  This always returns a non-null constant, but it may be a
/// ConstantExpr if unfoldable.
static Constant *FoldBitCast(Constant *C, Type *DestTy,
                             const DataLayout &TD) {
  // Catch the obvious splat cases.
  if (C->isNullValue() && !DestTy->isX86_MMXTy())
    return Constant::getNullValue(DestTy);
  if (C->isAllOnesValue() && !DestTy->isX86_MMXTy())
    return Constant::getAllOnesValue(DestTy);

  // Handle a vector->integer cast.
  if (IntegerType *IT = dyn_cast<IntegerType>(DestTy)) {
    ConstantDataVector *CDV = dyn_cast<ConstantDataVector>(C);
    if (CDV == 0)
      return ConstantExpr::getBitCast(C, DestTy);

    unsigned NumSrcElts = CDV->getType()->getNumElements();

    Type *SrcEltTy = CDV->getType()->getElementType();

    // If the vector is a vector of floating point, convert it to vector of int
    // to simplify things.
    if (SrcEltTy->isFloatingPointTy()) {
      unsigned FPWidth = SrcEltTy->getPrimitiveSizeInBits();
      Type *SrcIVTy =
        VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumSrcElts);
      // Ask VMCore to do the conversion now that #elts line up.
      C = ConstantExpr::getBitCast(C, SrcIVTy);
      CDV = cast<ConstantDataVector>(C);
    }

    // Now that we know that the input value is a vector of integers, just shift
    // and insert them into our result.
    unsigned BitShift = TD.getTypeAllocSizeInBits(SrcEltTy);
    APInt Result(IT->getBitWidth(), 0);
    for (unsigned i = 0; i != NumSrcElts; ++i) {
      Result <<= BitShift;
      if (TD.isLittleEndian())
        Result |= CDV->getElementAsInteger(NumSrcElts-i-1);
      else
        Result |= CDV->getElementAsInteger(i);
    }

    return ConstantInt::get(IT, Result);
  }

  // The code below only handles casts to vectors currently.
  VectorType *DestVTy = dyn_cast<VectorType>(DestTy);
  if (DestVTy == 0)
    return ConstantExpr::getBitCast(C, DestTy);

  // If this is a scalar -> vector cast, convert the input into a <1 x scalar>
  // vector so the code below can handle it uniformly.
  if (isa<ConstantFP>(C) || isa<ConstantInt>(C)) {
    Constant *Ops = C; // don't take the address of C!
    return FoldBitCast(ConstantVector::get(Ops), DestTy, TD);
  }

  // If this is a bitcast from constant vector -> vector, fold it.
  if (!isa<ConstantDataVector>(C) && !isa<ConstantVector>(C))
    return ConstantExpr::getBitCast(C, DestTy);

  // If the element types match, VMCore can fold it.
  unsigned NumDstElt = DestVTy->getNumElements();
  unsigned NumSrcElt = C->getType()->getVectorNumElements();
  if (NumDstElt == NumSrcElt)
    return ConstantExpr::getBitCast(C, DestTy);

  Type *SrcEltTy = C->getType()->getVectorElementType();
  Type *DstEltTy = DestVTy->getElementType();

  // Otherwise, we're changing the number of elements in a vector, which
  // requires endianness information to do the right thing.  For example,
  //    bitcast (<2 x i64> <i64 0, i64 1> to <4 x i32>)
  // folds to (little endian):
  //    <4 x i32> <i32 0, i32 0, i32 1, i32 0>
  // and to (big endian):
  //    <4 x i32> <i32 0, i32 0, i32 0, i32 1>

  // First thing is first.  We only want to think about integer here, so if
  // we have something in FP form, recast it as integer.
  if (DstEltTy->isFloatingPointTy()) {
    // Fold to an vector of integers with same size as our FP type.
    unsigned FPWidth = DstEltTy->getPrimitiveSizeInBits();
    Type *DestIVTy =
      VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumDstElt);
    // Recursively handle this integer conversion, if possible.
    C = FoldBitCast(C, DestIVTy, TD);

    // Finally, VMCore can handle this now that #elts line up.
    return ConstantExpr::getBitCast(C, DestTy);
  }

  // Okay, we know the destination is integer, if the input is FP, convert
  // it to integer first.
  if (SrcEltTy->isFloatingPointTy()) {
    unsigned FPWidth = SrcEltTy->getPrimitiveSizeInBits();
    Type *SrcIVTy =
      VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumSrcElt);
    // Ask VMCore to do the conversion now that #elts line up.
    C = ConstantExpr::getBitCast(C, SrcIVTy);
    // If VMCore wasn't able to fold it, bail out.
    if (!isa<ConstantVector>(C) &&  // FIXME: Remove ConstantVector.
        !isa<ConstantDataVector>(C))
      return C;
  }

  // Now we know that the input and output vectors are both integer vectors
  // of the same size, and that their #elements is not the same.  Do the
  // conversion here, which depends on whether the input or output has
  // more elements.
  bool isLittleEndian = TD.isLittleEndian();

  SmallVector<Constant*, 32> Result;
  if (NumDstElt < NumSrcElt) {
    // Handle: bitcast (<4 x i32> <i32 0, i32 1, i32 2, i32 3> to <2 x i64>)
    Constant *Zero = Constant::getNullValue(DstEltTy);
    unsigned Ratio = NumSrcElt/NumDstElt;
    unsigned SrcBitSize = SrcEltTy->getPrimitiveSizeInBits();
    unsigned SrcElt = 0;
    for (unsigned i = 0; i != NumDstElt; ++i) {
      // Build each element of the result.
      Constant *Elt = Zero;
      unsigned ShiftAmt = isLittleEndian ? 0 : SrcBitSize*(Ratio-1);
      for (unsigned j = 0; j != Ratio; ++j) {
        Constant *Src =dyn_cast<ConstantInt>(C->getAggregateElement(SrcElt++));
        if (!Src)  // Reject constantexpr elements.
          return ConstantExpr::getBitCast(C, DestTy);

        // Zero extend the element to the right size.
        Src = ConstantExpr::getZExt(Src, Elt->getType());

        // Shift it to the right place, depending on endianness.
        Src = ConstantExpr::getShl(Src,
                                   ConstantInt::get(Src->getType(), ShiftAmt));
        ShiftAmt += isLittleEndian ? SrcBitSize : -SrcBitSize;

        // Mix it in.
        Elt = ConstantExpr::getOr(Elt, Src);
      }
      Result.push_back(Elt);
    }
    return ConstantVector::get(Result);
  }

  // Handle: bitcast (<2 x i64> <i64 0, i64 1> to <4 x i32>)
  unsigned Ratio = NumDstElt/NumSrcElt;
  unsigned DstBitSize = DstEltTy->getPrimitiveSizeInBits();

  // Loop over each source value, expanding into multiple results.
  for (unsigned i = 0; i != NumSrcElt; ++i) {
    Constant *Src = dyn_cast<ConstantInt>(C->getAggregateElement(i));
    if (!Src)  // Reject constantexpr elements.
      return ConstantExpr::getBitCast(C, DestTy);

    unsigned ShiftAmt = isLittleEndian ? 0 : DstBitSize*(Ratio-1);
    for (unsigned j = 0; j != Ratio; ++j) {
      // Shift the piece of the value into the right place, depending on
      // endianness.
      Constant *Elt = ConstantExpr::getLShr(Src,
                                  ConstantInt::get(Src->getType(), ShiftAmt));
      ShiftAmt += isLittleEndian ? DstBitSize : -DstBitSize;

      // Truncate and remember this piece.
      Result.push_back(ConstantExpr::getTrunc(Elt, DstEltTy));
    }
  }

  return ConstantVector::get(Result);
}


/// IsConstantOffsetFromGlobal - If this constant is actually a constant offset
/// from a global, return the global and the constant.  Because of
/// constantexprs, this function is recursive.
static bool IsConstantOffsetFromGlobal(Constant *C, GlobalValue *&GV,
                                       int64_t &Offset, const DataLayout &TD) {
  // Trivial case, constant is the global.
  if ((GV = dyn_cast<GlobalValue>(C))) {
    Offset = 0;
    return true;
  }

  // Otherwise, if this isn't a constant expr, bail out.
  ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE) return false;

  // Look through ptr->int and ptr->ptr casts.
  if (CE->getOpcode() == Instruction::PtrToInt ||
      CE->getOpcode() == Instruction::BitCast)
    return IsConstantOffsetFromGlobal(CE->getOperand(0), GV, Offset, TD);

  // i32* getelementptr ([5 x i32]* @a, i32 0, i32 5)
  if (CE->getOpcode() == Instruction::GetElementPtr) {
    // Cannot compute this if the element type of the pointer is missing size
    // info.
    if (!cast<PointerType>(CE->getOperand(0)->getType())
                 ->getElementType()->isSized())
      return false;

    // If the base isn't a global+constant, we aren't either.
    if (!IsConstantOffsetFromGlobal(CE->getOperand(0), GV, Offset, TD))
      return false;

    // Otherwise, add any offset that our operands provide.
    gep_type_iterator GTI = gep_type_begin(CE);
    for (User::const_op_iterator i = CE->op_begin() + 1, e = CE->op_end();
         i != e; ++i, ++GTI) {
      ConstantInt *CI = dyn_cast<ConstantInt>(*i);
      if (!CI) return false;  // Index isn't a simple constant?
      if (CI->isZero()) continue;  // Not adding anything.

      if (StructType *ST = dyn_cast<StructType>(*GTI)) {
        // N = N + Offset
        Offset += TD.getStructLayout(ST)->getElementOffset(CI->getZExtValue());
      } else {
        SequentialType *SQT = cast<SequentialType>(*GTI);
        Offset += TD.getTypeAllocSize(SQT->getElementType())*CI->getSExtValue();
      }
    }
    return true;
  }

  return false;
}

/// ReadDataFromGlobal - Recursive helper to read bits out of global.  C is the
/// constant being copied out of. ByteOffset is an offset into C.  CurPtr is the
/// pointer to copy results into and BytesLeft is the number of bytes left in
/// the CurPtr buffer.  TD is the target data.
static bool ReadDataFromGlobal(Constant *C, uint64_t ByteOffset,
                               unsigned char *CurPtr, unsigned BytesLeft,
                               const DataLayout &TD) {
  assert(ByteOffset <= TD.getTypeAllocSize(C->getType()) &&
         "Out of range access");

  // If this element is zero or undefined, we can just return since *CurPtr is
  // zero initialized.
  if (isa<ConstantAggregateZero>(C) || isa<UndefValue>(C))
    return true;

  if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
    if (CI->getBitWidth() > 64 ||
        (CI->getBitWidth() & 7) != 0)
      return false;

    uint64_t Val = CI->getZExtValue();
    unsigned IntBytes = unsigned(CI->getBitWidth()/8);

    for (unsigned i = 0; i != BytesLeft && ByteOffset != IntBytes; ++i) {
      int n = ByteOffset;
      if (!TD.isLittleEndian())
        n = IntBytes - n - 1;
      CurPtr[i] = (unsigned char)(Val >> (n * 8));
      ++ByteOffset;
    }
    return true;
  }

  if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    if (CFP->getType()->isDoubleTy()) {
      C = FoldBitCast(C, Type::getInt64Ty(C->getContext()), TD);
      return ReadDataFromGlobal(C, ByteOffset, CurPtr, BytesLeft, TD);
    }
    if (CFP->getType()->isFloatTy()){
      C = FoldBitCast(C, Type::getInt32Ty(C->getContext()), TD);
      return ReadDataFromGlobal(C, ByteOffset, CurPtr, BytesLeft, TD);
    }
    return false;
  }

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    const StructLayout *SL = TD.getStructLayout(CS->getType());
    unsigned Index = SL->getElementContainingOffset(ByteOffset);
    uint64_t CurEltOffset = SL->getElementOffset(Index);
    ByteOffset -= CurEltOffset;

    while (1) {
      // If the element access is to the element itself and not to tail padding,
      // read the bytes from the element.
      uint64_t EltSize = TD.getTypeAllocSize(CS->getOperand(Index)->getType());

      if (ByteOffset < EltSize &&
          !ReadDataFromGlobal(CS->getOperand(Index), ByteOffset, CurPtr,
                              BytesLeft, TD))
        return false;

      ++Index;

      // Check to see if we read from the last struct element, if so we're done.
      if (Index == CS->getType()->getNumElements())
        return true;

      // If we read all of the bytes we needed from this element we're done.
      uint64_t NextEltOffset = SL->getElementOffset(Index);

      if (BytesLeft <= NextEltOffset-CurEltOffset-ByteOffset)
        return true;

      // Move to the next element of the struct.
      CurPtr += NextEltOffset-CurEltOffset-ByteOffset;
      BytesLeft -= NextEltOffset-CurEltOffset-ByteOffset;
      ByteOffset = 0;
      CurEltOffset = NextEltOffset;
    }
    // not reached.
  }

  if (isa<ConstantArray>(C) || isa<ConstantVector>(C) ||
      isa<ConstantDataSequential>(C)) {
    Type *EltTy = cast<SequentialType>(C->getType())->getElementType();
    uint64_t EltSize = TD.getTypeAllocSize(EltTy);
    uint64_t Index = ByteOffset / EltSize;
    uint64_t Offset = ByteOffset - Index * EltSize;
    uint64_t NumElts;
    if (ArrayType *AT = dyn_cast<ArrayType>(C->getType()))
      NumElts = AT->getNumElements();
    else
      NumElts = cast<VectorType>(C->getType())->getNumElements();

    for (; Index != NumElts; ++Index) {
      if (!ReadDataFromGlobal(C->getAggregateElement(Index), Offset, CurPtr,
                              BytesLeft, TD))
        return false;

      uint64_t BytesWritten = EltSize - Offset;
      assert(BytesWritten <= EltSize && "Not indexing into this element?");
      if (BytesWritten >= BytesLeft)
        return true;

      Offset = 0;
      BytesLeft -= BytesWritten;
      CurPtr += BytesWritten;
    }
    return true;
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::IntToPtr &&
        CE->getOperand(0)->getType() == TD.getIntPtrType(CE->getContext()))
      return ReadDataFromGlobal(CE->getOperand(0), ByteOffset, CurPtr,
                                BytesLeft, TD);
  }

  // Otherwise, unknown initializer type.
  return false;
}

static Constant *FoldReinterpretLoadFromConstPtr(Constant *C,
                                                 const DataLayout &TD) {
  Type *LoadTy = cast<PointerType>(C->getType())->getElementType();
  IntegerType *IntType = dyn_cast<IntegerType>(LoadTy);

  // If this isn't an integer load we can't fold it directly.
  if (!IntType) {
    // If this is a float/double load, we can try folding it as an int32/64 load
    // and then bitcast the result.  This can be useful for union cases.  Note
    // that address spaces don't matter here since we're not going to result in
    // an actual new load.
    Type *MapTy;
    if (LoadTy->isFloatTy())
      MapTy = Type::getInt32PtrTy(C->getContext());
    else if (LoadTy->isDoubleTy())
      MapTy = Type::getInt64PtrTy(C->getContext());
    else if (LoadTy->isVectorTy()) {
      MapTy = IntegerType::get(C->getContext(),
                               TD.getTypeAllocSizeInBits(LoadTy));
      MapTy = PointerType::getUnqual(MapTy);
    } else
      return 0;

    C = FoldBitCast(C, MapTy, TD);
    if (Constant *Res = FoldReinterpretLoadFromConstPtr(C, TD))
      return FoldBitCast(Res, LoadTy, TD);
    return 0;
  }

  unsigned BytesLoaded = (IntType->getBitWidth() + 7) / 8;
  if (BytesLoaded > 32 || BytesLoaded == 0) return 0;

  GlobalValue *GVal;
  int64_t Offset;
  if (!IsConstantOffsetFromGlobal(C, GVal, Offset, TD))
    return 0;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(GVal);
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer() ||
      !GV->getInitializer()->getType()->isSized())
    return 0;

  // If we're loading off the beginning of the global, some bytes may be valid,
  // but we don't try to handle this.
  if (Offset < 0) return 0;

  // If we're not accessing anything in this constant, the result is undefined.
  if (uint64_t(Offset) >= TD.getTypeAllocSize(GV->getInitializer()->getType()))
    return UndefValue::get(IntType);

  unsigned char RawBytes[32] = {0};
  if (!ReadDataFromGlobal(GV->getInitializer(), Offset, RawBytes,
                          BytesLoaded, TD))
    return 0;

  APInt ResultVal = APInt(IntType->getBitWidth(), 0);
  if (TD.isLittleEndian()) {
    ResultVal = RawBytes[BytesLoaded - 1];
    for (unsigned i = 1; i != BytesLoaded; ++i) {
      ResultVal <<= 8;
      ResultVal |= RawBytes[BytesLoaded-1-i];
    }
  } else {
    ResultVal = RawBytes[0];
    for (unsigned i = 1; i != BytesLoaded; ++i) {
      ResultVal <<= 8;
      ResultVal |= RawBytes[i];
    }
  }

  return ConstantInt::get(IntType->getContext(), ResultVal);
}

/// ConstantFoldLoadFromConstPtr - Return the value that a load from C would
/// produce if it is constant and determinable.  If this is not determinable,
/// return null.
Constant *llvm::ConstantFoldLoadFromConstPtr(Constant *C,
                                             const DataLayout *TD) {
  // First, try the easy cases:
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    if (GV->isConstant() && GV->hasDefinitiveInitializer())
      return GV->getInitializer();

  // If the loaded value isn't a constant expr, we can't handle it.
  ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE) return 0;

  if (CE->getOpcode() == Instruction::GetElementPtr) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
      if (GV->isConstant() && GV->hasDefinitiveInitializer())
        if (Constant *V =
             ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE))
          return V;
  }

  // Instead of loading constant c string, use corresponding integer value
  // directly if string length is small enough.
  StringRef Str;
  if (TD && getConstantStringInfo(CE, Str) && !Str.empty()) {
    unsigned StrLen = Str.size();
    Type *Ty = cast<PointerType>(CE->getType())->getElementType();
    unsigned NumBits = Ty->getPrimitiveSizeInBits();
    // Replace load with immediate integer if the result is an integer or fp
    // value.
    if ((NumBits >> 3) == StrLen + 1 && (NumBits & 7) == 0 &&
        (isa<IntegerType>(Ty) || Ty->isFloatingPointTy())) {
      APInt StrVal(NumBits, 0);
      APInt SingleChar(NumBits, 0);
      if (TD->isLittleEndian()) {
        for (signed i = StrLen-1; i >= 0; i--) {
          SingleChar = (uint64_t) Str[i] & UCHAR_MAX;
          StrVal = (StrVal << 8) | SingleChar;
        }
      } else {
        for (unsigned i = 0; i < StrLen; i++) {
          SingleChar = (uint64_t) Str[i] & UCHAR_MAX;
          StrVal = (StrVal << 8) | SingleChar;
        }
        // Append NULL at the end.
        SingleChar = 0;
        StrVal = (StrVal << 8) | SingleChar;
      }

      Constant *Res = ConstantInt::get(CE->getContext(), StrVal);
      if (Ty->isFloatingPointTy())
        Res = ConstantExpr::getBitCast(Res, Ty);
      return Res;
    }
  }

  // If this load comes from anywhere in a constant global, and if the global
  // is all undef or zero, we know what it loads.
  if (GlobalVariable *GV =
        dyn_cast<GlobalVariable>(GetUnderlyingObject(CE, TD))) {
    if (GV->isConstant() && GV->hasDefinitiveInitializer()) {
      Type *ResTy = cast<PointerType>(C->getType())->getElementType();
      if (GV->getInitializer()->isNullValue())
        return Constant::getNullValue(ResTy);
      if (isa<UndefValue>(GV->getInitializer()))
        return UndefValue::get(ResTy);
    }
  }

  // Try hard to fold loads from bitcasted strange and non-type-safe things.
  if (TD)
    return FoldReinterpretLoadFromConstPtr(CE, *TD);
  return 0;
}

static Constant *ConstantFoldLoadInst(const LoadInst *LI, const DataLayout *TD){
  if (LI->isVolatile()) return 0;

  if (Constant *C = dyn_cast<Constant>(LI->getOperand(0)))
    return ConstantFoldLoadFromConstPtr(C, TD);

  return 0;
}

/// SymbolicallyEvaluateBinop - One of Op0/Op1 is a constant expression.
/// Attempt to symbolically evaluate the result of a binary operator merging
/// these together.  If target data info is available, it is provided as TD,
/// otherwise TD is null.
static Constant *SymbolicallyEvaluateBinop(unsigned Opc, Constant *Op0,
                                           Constant *Op1, const DataLayout *TD){
  // SROA

  // Fold (and 0xffffffff00000000, (shl x, 32)) -> shl.
  // Fold (lshr (or X, Y), 32) -> (lshr [X/Y], 32) if one doesn't contribute
  // bits.


  // If the constant expr is something like &A[123] - &A[4].f, fold this into a
  // constant.  This happens frequently when iterating over a global array.
  if (Opc == Instruction::Sub && TD) {
    GlobalValue *GV1, *GV2;
    int64_t Offs1, Offs2;

    if (IsConstantOffsetFromGlobal(Op0, GV1, Offs1, *TD))
      if (IsConstantOffsetFromGlobal(Op1, GV2, Offs2, *TD) &&
          GV1 == GV2) {
        // (&GV+C1) - (&GV+C2) -> C1-C2, pointer arithmetic cannot overflow.
        return ConstantInt::get(Op0->getType(), Offs1-Offs2);
      }
  }

  return 0;
}

/// CastGEPIndices - If array indices are not pointer-sized integers,
/// explicitly cast them so that they aren't implicitly casted by the
/// getelementptr.
static Constant *CastGEPIndices(ArrayRef<Constant *> Ops,
                                Type *ResultTy, const DataLayout *TD,
                                const TargetLibraryInfo *TLI) {
  if (!TD) return 0;
  Type *IntPtrTy = TD->getIntPtrType(ResultTy->getContext());

  bool Any = false;
  SmallVector<Constant*, 32> NewIdxs;
  for (unsigned i = 1, e = Ops.size(); i != e; ++i) {
    if ((i == 1 ||
         !isa<StructType>(GetElementPtrInst::getIndexedType(Ops[0]->getType(),
                                                        Ops.slice(1, i-1)))) &&
        Ops[i]->getType() != IntPtrTy) {
      Any = true;
      NewIdxs.push_back(ConstantExpr::getCast(CastInst::getCastOpcode(Ops[i],
                                                                      true,
                                                                      IntPtrTy,
                                                                      true),
                                              Ops[i], IntPtrTy));
    } else
      NewIdxs.push_back(Ops[i]);
  }
  if (!Any) return 0;

  Constant *C =
    ConstantExpr::getGetElementPtr(Ops[0], NewIdxs);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (Constant *Folded = ConstantFoldConstantExpression(CE, TD, TLI))
      C = Folded;
  return C;
}

/// Strip the pointer casts, but preserve the address space information.
static Constant* StripPtrCastKeepAS(Constant* Ptr) {
  assert(Ptr->getType()->isPointerTy() && "Not a pointer type");
  PointerType *OldPtrTy = cast<PointerType>(Ptr->getType());
  Ptr = cast<Constant>(Ptr->stripPointerCasts());
  PointerType *NewPtrTy = cast<PointerType>(Ptr->getType());

  // Preserve the address space number of the pointer.
  if (NewPtrTy->getAddressSpace() != OldPtrTy->getAddressSpace()) {
    NewPtrTy = NewPtrTy->getElementType()->getPointerTo(
      OldPtrTy->getAddressSpace());
    Ptr = ConstantExpr::getBitCast(Ptr, NewPtrTy);
  }
  return Ptr;
}

/// SymbolicallyEvaluateGEP - If we can symbolically evaluate the specified GEP
/// constant expression, do so.
static Constant *SymbolicallyEvaluateGEP(ArrayRef<Constant *> Ops,
                                         Type *ResultTy, const DataLayout *TD,
                                         const TargetLibraryInfo *TLI) {
  Constant *Ptr = Ops[0];
  if (!TD || !cast<PointerType>(Ptr->getType())->getElementType()->isSized() ||
      !Ptr->getType()->isPointerTy())
    return 0;

  Type *IntPtrTy = TD->getIntPtrType(Ptr->getContext());

  // If this is a constant expr gep that is effectively computing an
  // "offsetof", fold it into 'cast int Size to T*' instead of 'gep 0, 0, 12'
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    if (!isa<ConstantInt>(Ops[i])) {

      // If this is "gep i8* Ptr, (sub 0, V)", fold this as:
      // "inttoptr (sub (ptrtoint Ptr), V)"
      if (Ops.size() == 2 &&
          cast<PointerType>(ResultTy)->getElementType()->isIntegerTy(8)) {
        ConstantExpr *CE = dyn_cast<ConstantExpr>(Ops[1]);
        assert((CE == 0 || CE->getType() == IntPtrTy) &&
               "CastGEPIndices didn't canonicalize index types!");
        if (CE && CE->getOpcode() == Instruction::Sub &&
            CE->getOperand(0)->isNullValue()) {
          Constant *Res = ConstantExpr::getPtrToInt(Ptr, CE->getType());
          Res = ConstantExpr::getSub(Res, CE->getOperand(1));
          Res = ConstantExpr::getIntToPtr(Res, ResultTy);
          if (ConstantExpr *ResCE = dyn_cast<ConstantExpr>(Res))
            Res = ConstantFoldConstantExpression(ResCE, TD, TLI);
          return Res;
        }
      }
      return 0;
    }

  unsigned BitWidth = TD->getTypeSizeInBits(IntPtrTy);
  APInt Offset =
    APInt(BitWidth, TD->getIndexedOffset(Ptr->getType(),
                                         makeArrayRef((Value *const*)
                                                        Ops.data() + 1,
                                                      Ops.size() - 1)));
  Ptr = StripPtrCastKeepAS(Ptr);

  // If this is a GEP of a GEP, fold it all into a single GEP.
  while (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr)) {
    SmallVector<Value *, 4> NestedOps(GEP->op_begin()+1, GEP->op_end());

    // Do not try the incorporate the sub-GEP if some index is not a number.
    bool AllConstantInt = true;
    for (unsigned i = 0, e = NestedOps.size(); i != e; ++i)
      if (!isa<ConstantInt>(NestedOps[i])) {
        AllConstantInt = false;
        break;
      }
    if (!AllConstantInt)
      break;

    Ptr = cast<Constant>(GEP->getOperand(0));
    Offset += APInt(BitWidth,
                    TD->getIndexedOffset(Ptr->getType(), NestedOps));
    Ptr = StripPtrCastKeepAS(Ptr);
  }

  // If the base value for this address is a literal integer value, fold the
  // getelementptr to the resulting integer value casted to the pointer type.
  APInt BasePtr(BitWidth, 0);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->getOpcode() == Instruction::IntToPtr)
      if (ConstantInt *Base = dyn_cast<ConstantInt>(CE->getOperand(0)))
        BasePtr = Base->getValue().zextOrTrunc(BitWidth);
  if (Ptr->isNullValue() || BasePtr != 0) {
    Constant *C = ConstantInt::get(Ptr->getContext(), Offset+BasePtr);
    return ConstantExpr::getIntToPtr(C, ResultTy);
  }

  // Otherwise form a regular getelementptr. Recompute the indices so that
  // we eliminate over-indexing of the notional static type array bounds.
  // This makes it easy to determine if the getelementptr is "inbounds".
  // Also, this helps GlobalOpt do SROA on GlobalVariables.
  Type *Ty = Ptr->getType();
  assert(Ty->isPointerTy() && "Forming regular GEP of non-pointer type");
  SmallVector<Constant*, 32> NewIdxs;
  do {
    if (SequentialType *ATy = dyn_cast<SequentialType>(Ty)) {
      if (ATy->isPointerTy()) {
        // The only pointer indexing we'll do is on the first index of the GEP.
        if (!NewIdxs.empty())
          break;

        // Only handle pointers to sized types, not pointers to functions.
        if (!ATy->getElementType()->isSized())
          return 0;
      }

      // Determine which element of the array the offset points into.
      APInt ElemSize(BitWidth, TD->getTypeAllocSize(ATy->getElementType()));
      IntegerType *IntPtrTy = TD->getIntPtrType(Ty->getContext());
      if (ElemSize == 0)
        // The element size is 0. This may be [0 x Ty]*, so just use a zero
        // index for this level and proceed to the next level to see if it can
        // accommodate the offset.
        NewIdxs.push_back(ConstantInt::get(IntPtrTy, 0));
      else {
        // The element size is non-zero divide the offset by the element
        // size (rounding down), to compute the index at this level.
        APInt NewIdx = Offset.udiv(ElemSize);
        Offset -= NewIdx * ElemSize;
        NewIdxs.push_back(ConstantInt::get(IntPtrTy, NewIdx));
      }
      Ty = ATy->getElementType();
    } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
      // If we end up with an offset that isn't valid for this struct type, we
      // can't re-form this GEP in a regular form, so bail out. The pointer
      // operand likely went through casts that are necessary to make the GEP
      // sensible.
      const StructLayout &SL = *TD->getStructLayout(STy);
      if (Offset.uge(SL.getSizeInBytes()))
        break;

      // Determine which field of the struct the offset points into. The
      // getZExtValue is fine as we've already ensured that the offset is
      // within the range representable by the StructLayout API.
      unsigned ElIdx = SL.getElementContainingOffset(Offset.getZExtValue());
      NewIdxs.push_back(ConstantInt::get(Type::getInt32Ty(Ty->getContext()),
                                         ElIdx));
      Offset -= APInt(BitWidth, SL.getElementOffset(ElIdx));
      Ty = STy->getTypeAtIndex(ElIdx);
    } else {
      // We've reached some non-indexable type.
      break;
    }
  } while (Ty != cast<PointerType>(ResultTy)->getElementType());

  // If we haven't used up the entire offset by descending the static
  // type, then the offset is pointing into the middle of an indivisible
  // member, so we can't simplify it.
  if (Offset != 0)
    return 0;

  // Create a GEP.
  Constant *C =
    ConstantExpr::getGetElementPtr(Ptr, NewIdxs);
  assert(cast<PointerType>(C->getType())->getElementType() == Ty &&
         "Computed GetElementPtr has unexpected type!");

  // If we ended up indexing a member with a type that doesn't match
  // the type of what the original indices indexed, add a cast.
  if (Ty != cast<PointerType>(ResultTy)->getElementType())
    C = FoldBitCast(C, ResultTy, *TD);

  return C;
}



//===----------------------------------------------------------------------===//
// Constant Folding public APIs
//===----------------------------------------------------------------------===//

/// ConstantFoldInstruction - Try to constant fold the specified instruction.
/// If successful, the constant result is returned, if not, null is returned.
/// Note that this fails if not all of the operands are constant.  Otherwise,
/// this function can only fail when attempting to fold instructions like loads
/// and stores, which have no constant expression form.
Constant *llvm::ConstantFoldInstruction(Instruction *I,
                                        const DataLayout *TD,
                                        const TargetLibraryInfo *TLI) {
  // Handle PHI nodes quickly here...
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    Constant *CommonValue = 0;

    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *Incoming = PN->getIncomingValue(i);
      // If the incoming value is undef then skip it.  Note that while we could
      // skip the value if it is equal to the phi node itself we choose not to
      // because that would break the rule that constant folding only applies if
      // all operands are constants.
      if (isa<UndefValue>(Incoming))
        continue;
      // If the incoming value is not a constant, then give up.
      Constant *C = dyn_cast<Constant>(Incoming);
      if (!C)
        return 0;
      // Fold the PHI's operands.
      if (ConstantExpr *NewC = dyn_cast<ConstantExpr>(C))
        C = ConstantFoldConstantExpression(NewC, TD, TLI);
      // If the incoming value is a different constant to
      // the one we saw previously, then give up.
      if (CommonValue && C != CommonValue)
        return 0;
      CommonValue = C;
    }


    // If we reach here, all incoming values are the same constant or undef.
    return CommonValue ? CommonValue : UndefValue::get(PN->getType());
  }

  // Scan the operand list, checking to see if they are all constants, if so,
  // hand off to ConstantFoldInstOperands.
  SmallVector<Constant*, 8> Ops;
  for (User::op_iterator i = I->op_begin(), e = I->op_end(); i != e; ++i) {
    Constant *Op = dyn_cast<Constant>(*i);
    if (!Op)
      return 0;  // All operands not constant!

    // Fold the Instruction's operands.
    if (ConstantExpr *NewCE = dyn_cast<ConstantExpr>(Op))
      Op = ConstantFoldConstantExpression(NewCE, TD, TLI);

    Ops.push_back(Op);
  }

  if (const CmpInst *CI = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(CI->getPredicate(), Ops[0], Ops[1],
                                           TD, TLI);

  if (const LoadInst *LI = dyn_cast<LoadInst>(I))
    return ConstantFoldLoadInst(LI, TD);

  if (InsertValueInst *IVI = dyn_cast<InsertValueInst>(I))
    return ConstantExpr::getInsertValue(
                                cast<Constant>(IVI->getAggregateOperand()),
                                cast<Constant>(IVI->getInsertedValueOperand()),
                                IVI->getIndices());

  if (ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I))
    return ConstantExpr::getExtractValue(
                                    cast<Constant>(EVI->getAggregateOperand()),
                                    EVI->getIndices());

  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), Ops, TD, TLI);
}

/// ConstantFoldConstantExpression - Attempt to fold the constant expression
/// using the specified DataLayout.  If successful, the constant result is
/// result is returned, if not, null is returned.
Constant *llvm::ConstantFoldConstantExpression(const ConstantExpr *CE,
                                               const DataLayout *TD,
                                               const TargetLibraryInfo *TLI) {
  SmallVector<Constant*, 8> Ops;
  for (User::const_op_iterator i = CE->op_begin(), e = CE->op_end();
       i != e; ++i) {
    Constant *NewC = cast<Constant>(*i);
    // Recursively fold the ConstantExpr's operands.
    if (ConstantExpr *NewCE = dyn_cast<ConstantExpr>(NewC))
      NewC = ConstantFoldConstantExpression(NewCE, TD, TLI);
    Ops.push_back(NewC);
  }

  if (CE->isCompare())
    return ConstantFoldCompareInstOperands(CE->getPredicate(), Ops[0], Ops[1],
                                           TD, TLI);
  return ConstantFoldInstOperands(CE->getOpcode(), CE->getType(), Ops, TD, TLI);
}

/// ConstantFoldInstOperands - Attempt to constant fold an instruction with the
/// specified opcode and operands.  If successful, the constant result is
/// returned, if not, null is returned.  Note that this function can fail when
/// attempting to fold instructions like loads and stores, which have no
/// constant expression form.
///
/// TODO: This function neither utilizes nor preserves nsw/nuw/inbounds/etc
/// information, due to only being passed an opcode and operands. Constant
/// folding using this function strips this information.
///
Constant *llvm::ConstantFoldInstOperands(unsigned Opcode, Type *DestTy,
                                         ArrayRef<Constant *> Ops,
                                         const DataLayout *TD,
                                         const TargetLibraryInfo *TLI) {
  // Handle easy binops first.
  if (Instruction::isBinaryOp(Opcode)) {
    if (isa<ConstantExpr>(Ops[0]) || isa<ConstantExpr>(Ops[1]))
      if (Constant *C = SymbolicallyEvaluateBinop(Opcode, Ops[0], Ops[1], TD))
        return C;

    return ConstantExpr::get(Opcode, Ops[0], Ops[1]);
  }

  switch (Opcode) {
  default: return 0;
  case Instruction::ICmp:
  case Instruction::FCmp: llvm_unreachable("Invalid for compares");
  case Instruction::Call:
    if (Function *F = dyn_cast<Function>(Ops.back()))
      if (canConstantFoldCallTo(F))
        return ConstantFoldCall(F, Ops.slice(0, Ops.size() - 1), TLI);
    return 0;
  case Instruction::PtrToInt:
    // If the input is a inttoptr, eliminate the pair.  This requires knowing
    // the width of a pointer, so it can't be done in ConstantExpr::getCast.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ops[0])) {
      if (TD && CE->getOpcode() == Instruction::IntToPtr) {
        Constant *Input = CE->getOperand(0);
        unsigned InWidth = Input->getType()->getScalarSizeInBits();
        if (TD->getPointerSizeInBits() < InWidth) {
          Constant *Mask =
            ConstantInt::get(CE->getContext(), APInt::getLowBitsSet(InWidth,
                                                  TD->getPointerSizeInBits()));
          Input = ConstantExpr::getAnd(Input, Mask);
        }
        // Do a zext or trunc to get to the dest size.
        return ConstantExpr::getIntegerCast(Input, DestTy, false);
      }
    }
    return ConstantExpr::getCast(Opcode, Ops[0], DestTy);
  case Instruction::IntToPtr:
    // If the input is a ptrtoint, turn the pair into a ptr to ptr bitcast if
    // the int size is >= the ptr size.  This requires knowing the width of a
    // pointer, so it can't be done in ConstantExpr::getCast.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ops[0]))
      if (TD &&
          TD->getPointerSizeInBits() <= CE->getType()->getScalarSizeInBits() &&
          CE->getOpcode() == Instruction::PtrToInt)
        return FoldBitCast(CE->getOperand(0), DestTy, *TD);

    return ConstantExpr::getCast(Opcode, Ops[0], DestTy);
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
      return ConstantExpr::getCast(Opcode, Ops[0], DestTy);
  case Instruction::BitCast:
    if (TD)
      return FoldBitCast(Ops[0], DestTy, *TD);
    return ConstantExpr::getBitCast(Ops[0], DestTy);
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1]);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2]);
  case Instruction::GetElementPtr:
    if (Constant *C = CastGEPIndices(Ops, DestTy, TD, TLI))
      return C;
    if (Constant *C = SymbolicallyEvaluateGEP(Ops, DestTy, TD, TLI))
      return C;

    return ConstantExpr::getGetElementPtr(Ops[0], Ops.slice(1));
  }
}

/// ConstantFoldCompareInstOperands - Attempt to constant fold a compare
/// instruction (icmp/fcmp) with the specified operands.  If it fails, it
/// returns a constant expression of the specified operands.
///
Constant *llvm::ConstantFoldCompareInstOperands(unsigned Predicate,
                                                Constant *Ops0, Constant *Ops1,
                                                const DataLayout *TD,
                                                const TargetLibraryInfo *TLI) {
  // fold: icmp (inttoptr x), null         -> icmp x, 0
  // fold: icmp (ptrtoint x), 0            -> icmp x, null
  // fold: icmp (inttoptr x), (inttoptr y) -> icmp trunc/zext x, trunc/zext y
  // fold: icmp (ptrtoint x), (ptrtoint y) -> icmp x, y
  //
  // ConstantExpr::getCompare cannot do this, because it doesn't have TD
  // around to know if bit truncation is happening.
  if (ConstantExpr *CE0 = dyn_cast<ConstantExpr>(Ops0)) {
    if (TD && Ops1->isNullValue()) {
      Type *IntPtrTy = TD->getIntPtrType(CE0->getContext());
      if (CE0->getOpcode() == Instruction::IntToPtr) {
        // Convert the integer value to the right size to ensure we get the
        // proper extension or truncation.
        Constant *C = ConstantExpr::getIntegerCast(CE0->getOperand(0),
                                                   IntPtrTy, false);
        Constant *Null = Constant::getNullValue(C->getType());
        return ConstantFoldCompareInstOperands(Predicate, C, Null, TD, TLI);
      }

      // Only do this transformation if the int is intptrty in size, otherwise
      // there is a truncation or extension that we aren't modeling.
      if (CE0->getOpcode() == Instruction::PtrToInt &&
          CE0->getType() == IntPtrTy) {
        Constant *C = CE0->getOperand(0);
        Constant *Null = Constant::getNullValue(C->getType());
        return ConstantFoldCompareInstOperands(Predicate, C, Null, TD, TLI);
      }
    }

    if (ConstantExpr *CE1 = dyn_cast<ConstantExpr>(Ops1)) {
      if (TD && CE0->getOpcode() == CE1->getOpcode()) {
        Type *IntPtrTy = TD->getIntPtrType(CE0->getContext());

        if (CE0->getOpcode() == Instruction::IntToPtr) {
          // Convert the integer value to the right size to ensure we get the
          // proper extension or truncation.
          Constant *C0 = ConstantExpr::getIntegerCast(CE0->getOperand(0),
                                                      IntPtrTy, false);
          Constant *C1 = ConstantExpr::getIntegerCast(CE1->getOperand(0),
                                                      IntPtrTy, false);
          return ConstantFoldCompareInstOperands(Predicate, C0, C1, TD, TLI);
        }

        // Only do this transformation if the int is intptrty in size, otherwise
        // there is a truncation or extension that we aren't modeling.
        if ((CE0->getOpcode() == Instruction::PtrToInt &&
             CE0->getType() == IntPtrTy &&
             CE0->getOperand(0)->getType() == CE1->getOperand(0)->getType()))
          return ConstantFoldCompareInstOperands(Predicate, CE0->getOperand(0),
                                                 CE1->getOperand(0), TD, TLI);
      }
    }

    // icmp eq (or x, y), 0 -> (icmp eq x, 0) & (icmp eq y, 0)
    // icmp ne (or x, y), 0 -> (icmp ne x, 0) | (icmp ne y, 0)
    if ((Predicate == ICmpInst::ICMP_EQ || Predicate == ICmpInst::ICMP_NE) &&
        CE0->getOpcode() == Instruction::Or && Ops1->isNullValue()) {
      Constant *LHS =
        ConstantFoldCompareInstOperands(Predicate, CE0->getOperand(0), Ops1,
                                        TD, TLI);
      Constant *RHS =
        ConstantFoldCompareInstOperands(Predicate, CE0->getOperand(1), Ops1,
                                        TD, TLI);
      unsigned OpC =
        Predicate == ICmpInst::ICMP_EQ ? Instruction::And : Instruction::Or;
      Constant *Ops[] = { LHS, RHS };
      return ConstantFoldInstOperands(OpC, LHS->getType(), Ops, TD, TLI);
    }
  }

  return ConstantExpr::getCompare(Predicate, Ops0, Ops1);
}


/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *llvm::ConstantFoldLoadThroughGEPConstantExpr(Constant *C,
                                                       ConstantExpr *CE) {
  if (!CE->getOperand(1)->isNullValue())
    return 0;  // Do not allow stepping over the value!

  // Loop over all of the operands, tracking down which value we are
  // addressing.
  for (unsigned i = 2, e = CE->getNumOperands(); i != e; ++i) {
    C = C->getAggregateElement(CE->getOperand(i));
    if (C == 0) return 0;
  }
  return C;
}

/// ConstantFoldLoadThroughGEPIndices - Given a constant and getelementptr
/// indices (with an *implied* zero pointer index that is not in the list),
/// return the constant value being addressed by a virtual load, or null if
/// something is funny and we can't decide.
Constant *llvm::ConstantFoldLoadThroughGEPIndices(Constant *C,
                                                  ArrayRef<Constant*> Indices) {
  // Loop over all of the operands, tracking down which value we are
  // addressing.
  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    C = C->getAggregateElement(Indices[i]);
    if (C == 0) return 0;
  }
  return C;
}


//===----------------------------------------------------------------------===//
//  Constant Folding for Calls
//

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool
llvm::canConstantFoldCallTo(const Function *F) {
  switch (F->getIntrinsicID()) {
  case Intrinsic::sqrt:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::bswap:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::convert_from_fp16:
  case Intrinsic::convert_to_fp16:
  case Intrinsic::x86_sse_cvtss2si:
  case Intrinsic::x86_sse_cvtss2si64:
  case Intrinsic::x86_sse_cvttss2si:
  case Intrinsic::x86_sse_cvttss2si64:
  case Intrinsic::x86_sse2_cvtsd2si:
  case Intrinsic::x86_sse2_cvtsd2si64:
  case Intrinsic::x86_sse2_cvttsd2si:
  case Intrinsic::x86_sse2_cvttsd2si64:
    return true;
  default:
    return false;
  case 0: break;
  }

  if (!F->hasName()) return false;
  StringRef Name = F->getName();

  // In these cases, the check of the length is required.  We don't want to
  // return true for a name like "cos\0blah" which strcmp would return equal to
  // "cos", but has length 8.
  switch (Name[0]) {
  default: return false;
  case 'a':
    return Name == "acos" || Name == "asin" ||
      Name == "atan" || Name == "atan2";
  case 'c':
    return Name == "cos" || Name == "ceil" || Name == "cosf" || Name == "cosh";
  case 'e':
    return Name == "exp" || Name == "exp2";
  case 'f':
    return Name == "fabs" || Name == "fmod" || Name == "floor";
  case 'l':
    return Name == "log" || Name == "log10";
  case 'p':
    return Name == "pow";
  case 's':
    return Name == "sin" || Name == "sinh" || Name == "sqrt" ||
      Name == "sinf" || Name == "sqrtf";
  case 't':
    return Name == "tan" || Name == "tanh";
  }
}

static Constant *ConstantFoldFP(double (*NativeFP)(double), double V,
                                Type *Ty) {
  sys::llvm_fenv_clearexcept();
  V = NativeFP(V);
  if (sys::llvm_fenv_testexcept()) {
    sys::llvm_fenv_clearexcept();
    return 0;
  }

  if (Ty->isFloatTy())
    return ConstantFP::get(Ty->getContext(), APFloat((float)V));
  if (Ty->isDoubleTy())
    return ConstantFP::get(Ty->getContext(), APFloat(V));
  llvm_unreachable("Can only constant fold float/double");
}

static Constant *ConstantFoldBinaryFP(double (*NativeFP)(double, double),
                                      double V, double W, Type *Ty) {
  sys::llvm_fenv_clearexcept();
  V = NativeFP(V, W);
  if (sys::llvm_fenv_testexcept()) {
    sys::llvm_fenv_clearexcept();
    return 0;
  }

  if (Ty->isFloatTy())
    return ConstantFP::get(Ty->getContext(), APFloat((float)V));
  if (Ty->isDoubleTy())
    return ConstantFP::get(Ty->getContext(), APFloat(V));
  llvm_unreachable("Can only constant fold float/double");
}

/// ConstantFoldConvertToInt - Attempt to an SSE floating point to integer
/// conversion of a constant floating point. If roundTowardZero is false, the
/// default IEEE rounding is used (toward nearest, ties to even). This matches
/// the behavior of the non-truncating SSE instructions in the default rounding
/// mode. The desired integer type Ty is used to select how many bits are
/// available for the result. Returns null if the conversion cannot be
/// performed, otherwise returns the Constant value resulting from the
/// conversion.
static Constant *ConstantFoldConvertToInt(const APFloat &Val,
                                          bool roundTowardZero, Type *Ty) {
  // All of these conversion intrinsics form an integer of at most 64bits.
  unsigned ResultWidth = cast<IntegerType>(Ty)->getBitWidth();
  assert(ResultWidth <= 64 &&
         "Can only constant fold conversions to 64 and 32 bit ints");

  uint64_t UIntVal;
  bool isExact = false;
  APFloat::roundingMode mode = roundTowardZero? APFloat::rmTowardZero
                                              : APFloat::rmNearestTiesToEven;
  APFloat::opStatus status = Val.convertToInteger(&UIntVal, ResultWidth,
                                                  /*isSigned=*/true, mode,
                                                  &isExact);
  if (status != APFloat::opOK && status != APFloat::opInexact)
    return 0;
  return ConstantInt::get(Ty, UIntVal, /*isSigned=*/true);
}

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *
llvm::ConstantFoldCall(Function *F, ArrayRef<Constant *> Operands,
                       const TargetLibraryInfo *TLI) {
  if (!F->hasName()) return 0;
  StringRef Name = F->getName();

  Type *Ty = F->getReturnType();
  if (Operands.size() == 1) {
    if (ConstantFP *Op = dyn_cast<ConstantFP>(Operands[0])) {
      if (F->getIntrinsicID() == Intrinsic::convert_to_fp16) {
        APFloat Val(Op->getValueAPF());

        bool lost = false;
        Val.convert(APFloat::IEEEhalf, APFloat::rmNearestTiesToEven, &lost);

        return ConstantInt::get(F->getContext(), Val.bitcastToAPInt());
      }
      if (!TLI)
        return 0;

      if (!Ty->isFloatTy() && !Ty->isDoubleTy())
        return 0;

      /// We only fold functions with finite arguments. Folding NaN and inf is
      /// likely to be aborted with an exception anyway, and some host libms
      /// have known errors raising exceptions.
      if (Op->getValueAPF().isNaN() || Op->getValueAPF().isInfinity())
        return 0;

      /// Currently APFloat versions of these functions do not exist, so we use
      /// the host native double versions.  Float versions are not called
      /// directly but for all these it is true (float)(f((double)arg)) ==
      /// f(arg).  Long double not supported yet.
      double V = Ty->isFloatTy() ? (double)Op->getValueAPF().convertToFloat() :
                                     Op->getValueAPF().convertToDouble();
      switch (Name[0]) {
      case 'a':
        if (Name == "acos" && TLI->has(LibFunc::acos))
          return ConstantFoldFP(acos, V, Ty);
        else if (Name == "asin" && TLI->has(LibFunc::asin))
          return ConstantFoldFP(asin, V, Ty);
        else if (Name == "atan" && TLI->has(LibFunc::atan))
          return ConstantFoldFP(atan, V, Ty);
        break;
      case 'c':
        if (Name == "ceil" && TLI->has(LibFunc::ceil))
          return ConstantFoldFP(ceil, V, Ty);
        else if (Name == "cos" && TLI->has(LibFunc::cos))
          return ConstantFoldFP(cos, V, Ty);
        else if (Name == "cosh" && TLI->has(LibFunc::cosh))
          return ConstantFoldFP(cosh, V, Ty);
        else if (Name == "cosf" && TLI->has(LibFunc::cosf))
          return ConstantFoldFP(cos, V, Ty);
        break;
      case 'e':
        if (Name == "exp" && TLI->has(LibFunc::exp))
          return ConstantFoldFP(exp, V, Ty);

        if (Name == "exp2" && TLI->has(LibFunc::exp2)) {
          // Constant fold exp2(x) as pow(2,x) in case the host doesn't have a
          // C99 library.
          return ConstantFoldBinaryFP(pow, 2.0, V, Ty);
        }
        break;
      case 'f':
        if (Name == "fabs" && TLI->has(LibFunc::fabs))
          return ConstantFoldFP(fabs, V, Ty);
        else if (Name == "floor" && TLI->has(LibFunc::floor))
          return ConstantFoldFP(floor, V, Ty);
        break;
      case 'l':
        if (Name == "log" && V > 0 && TLI->has(LibFunc::log))
          return ConstantFoldFP(log, V, Ty);
        else if (Name == "log10" && V > 0 && TLI->has(LibFunc::log10))
          return ConstantFoldFP(log10, V, Ty);
        else if (F->getIntrinsicID() == Intrinsic::sqrt &&
                 (Ty->isFloatTy() || Ty->isDoubleTy())) {
          if (V >= -0.0)
            return ConstantFoldFP(sqrt, V, Ty);
          else // Undefined
            return Constant::getNullValue(Ty);
        }
        break;
      case 's':
        if (Name == "sin" && TLI->has(LibFunc::sin))
          return ConstantFoldFP(sin, V, Ty);
        else if (Name == "sinh" && TLI->has(LibFunc::sinh))
          return ConstantFoldFP(sinh, V, Ty);
        else if (Name == "sqrt" && V >= 0 && TLI->has(LibFunc::sqrt))
          return ConstantFoldFP(sqrt, V, Ty);
        else if (Name == "sqrtf" && V >= 0 && TLI->has(LibFunc::sqrtf))
          return ConstantFoldFP(sqrt, V, Ty);
        else if (Name == "sinf" && TLI->has(LibFunc::sinf))
          return ConstantFoldFP(sin, V, Ty);
        break;
      case 't':
        if (Name == "tan" && TLI->has(LibFunc::tan))
          return ConstantFoldFP(tan, V, Ty);
        else if (Name == "tanh" && TLI->has(LibFunc::tanh))
          return ConstantFoldFP(tanh, V, Ty);
        break;
      default:
        break;
      }
      return 0;
    }

    if (ConstantInt *Op = dyn_cast<ConstantInt>(Operands[0])) {
      switch (F->getIntrinsicID()) {
      case Intrinsic::bswap:
        return ConstantInt::get(F->getContext(), Op->getValue().byteSwap());
      case Intrinsic::ctpop:
        return ConstantInt::get(Ty, Op->getValue().countPopulation());
      case Intrinsic::convert_from_fp16: {
        APFloat Val(Op->getValue());

        bool lost = false;
        APFloat::opStatus status =
          Val.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &lost);

        // Conversion is always precise.
        (void)status;
        assert(status == APFloat::opOK && !lost &&
               "Precision lost during fp16 constfolding");

        return ConstantFP::get(F->getContext(), Val);
      }
      default:
        return 0;
      }
    }

    // Support ConstantVector in case we have an Undef in the top.
    if (isa<ConstantVector>(Operands[0]) ||
        isa<ConstantDataVector>(Operands[0])) {
      Constant *Op = cast<Constant>(Operands[0]);
      switch (F->getIntrinsicID()) {
      default: break;
      case Intrinsic::x86_sse_cvtss2si:
      case Intrinsic::x86_sse_cvtss2si64:
      case Intrinsic::x86_sse2_cvtsd2si:
      case Intrinsic::x86_sse2_cvtsd2si64:
        if (ConstantFP *FPOp =
              dyn_cast_or_null<ConstantFP>(Op->getAggregateElement(0U)))
          return ConstantFoldConvertToInt(FPOp->getValueAPF(),
                                          /*roundTowardZero=*/false, Ty);
      case Intrinsic::x86_sse_cvttss2si:
      case Intrinsic::x86_sse_cvttss2si64:
      case Intrinsic::x86_sse2_cvttsd2si:
      case Intrinsic::x86_sse2_cvttsd2si64:
        if (ConstantFP *FPOp =
              dyn_cast_or_null<ConstantFP>(Op->getAggregateElement(0U)))
          return ConstantFoldConvertToInt(FPOp->getValueAPF(),
                                          /*roundTowardZero=*/true, Ty);
      }
    }

    if (isa<UndefValue>(Operands[0])) {
      if (F->getIntrinsicID() == Intrinsic::bswap)
        return Operands[0];
      return 0;
    }

    return 0;
  }

  if (Operands.size() == 2) {
    if (ConstantFP *Op1 = dyn_cast<ConstantFP>(Operands[0])) {
      if (!Ty->isFloatTy() && !Ty->isDoubleTy())
        return 0;
      double Op1V = Ty->isFloatTy() ?
                      (double)Op1->getValueAPF().convertToFloat() :
                      Op1->getValueAPF().convertToDouble();
      if (ConstantFP *Op2 = dyn_cast<ConstantFP>(Operands[1])) {
        if (Op2->getType() != Op1->getType())
          return 0;

        double Op2V = Ty->isFloatTy() ?
                      (double)Op2->getValueAPF().convertToFloat():
                      Op2->getValueAPF().convertToDouble();

        if (F->getIntrinsicID() == Intrinsic::pow) {
          return ConstantFoldBinaryFP(pow, Op1V, Op2V, Ty);
        }
        if (!TLI)
          return 0;
        if (Name == "pow" && TLI->has(LibFunc::pow))
          return ConstantFoldBinaryFP(pow, Op1V, Op2V, Ty);
        if (Name == "fmod" && TLI->has(LibFunc::fmod))
          return ConstantFoldBinaryFP(fmod, Op1V, Op2V, Ty);
        if (Name == "atan2" && TLI->has(LibFunc::atan2))
          return ConstantFoldBinaryFP(atan2, Op1V, Op2V, Ty);
      } else if (ConstantInt *Op2C = dyn_cast<ConstantInt>(Operands[1])) {
        if (F->getIntrinsicID() == Intrinsic::powi && Ty->isFloatTy())
          return ConstantFP::get(F->getContext(),
                                 APFloat((float)std::pow((float)Op1V,
                                                 (int)Op2C->getZExtValue())));
        if (F->getIntrinsicID() == Intrinsic::powi && Ty->isDoubleTy())
          return ConstantFP::get(F->getContext(),
                                 APFloat((double)std::pow((double)Op1V,
                                                   (int)Op2C->getZExtValue())));
      }
      return 0;
    }

    if (ConstantInt *Op1 = dyn_cast<ConstantInt>(Operands[0])) {
      if (ConstantInt *Op2 = dyn_cast<ConstantInt>(Operands[1])) {
        switch (F->getIntrinsicID()) {
        default: break;
        case Intrinsic::sadd_with_overflow:
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::ssub_with_overflow:
        case Intrinsic::usub_with_overflow:
        case Intrinsic::smul_with_overflow:
        case Intrinsic::umul_with_overflow: {
          APInt Res;
          bool Overflow;
          switch (F->getIntrinsicID()) {
          default: llvm_unreachable("Invalid case");
          case Intrinsic::sadd_with_overflow:
            Res = Op1->getValue().sadd_ov(Op2->getValue(), Overflow);
            break;
          case Intrinsic::uadd_with_overflow:
            Res = Op1->getValue().uadd_ov(Op2->getValue(), Overflow);
            break;
          case Intrinsic::ssub_with_overflow:
            Res = Op1->getValue().ssub_ov(Op2->getValue(), Overflow);
            break;
          case Intrinsic::usub_with_overflow:
            Res = Op1->getValue().usub_ov(Op2->getValue(), Overflow);
            break;
          case Intrinsic::smul_with_overflow:
            Res = Op1->getValue().smul_ov(Op2->getValue(), Overflow);
            break;
          case Intrinsic::umul_with_overflow:
            Res = Op1->getValue().umul_ov(Op2->getValue(), Overflow);
            break;
          }
          Constant *Ops[] = {
            ConstantInt::get(F->getContext(), Res),
            ConstantInt::get(Type::getInt1Ty(F->getContext()), Overflow)
          };
          return ConstantStruct::get(cast<StructType>(F->getReturnType()), Ops);
        }
        case Intrinsic::cttz:
          // FIXME: This should check for Op2 == 1, and become unreachable if
          // Op1 == 0.
          return ConstantInt::get(Ty, Op1->getValue().countTrailingZeros());
        case Intrinsic::ctlz:
          // FIXME: This should check for Op2 == 1, and become unreachable if
          // Op1 == 0.
          return ConstantInt::get(Ty, Op1->getValue().countLeadingZeros());
        }
      }

      return 0;
    }
    return 0;
  }
  return 0;
}
