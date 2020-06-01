//===- llvm/MatrixBuilder.h - Builder to lower matrix ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MatrixBuilder class, which is used as a convenient way
// to lower matrix operations to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MATRIXBUILDER_H
#define LLVM_IR_MATRIXBUILDER_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {

class Function;
class Twine;
class Module;

template <class IRBuilderTy> class MatrixBuilder {
  IRBuilderTy &B;
  Module *getModule() { return B.GetInsertBlock()->getParent()->getParent(); }

public:
  MatrixBuilder(IRBuilderTy &Builder) : B(Builder) {}

  /// Create a columnwise, strided matrix load.
  /// \p DataPtr - Start address of the matrix read
  /// \p Rows    - Number of rows in matrix (must be a constant)
  /// \p Columns - Number of columns in matrix (must be a constant)
  /// \p Stride  - Space between columns
  CallInst *CreateMatrixColumnwiseLoad(Value *DataPtr, unsigned Rows,
                                       unsigned Columns, Value *Stride,
                                       const Twine &Name = "") {

    // Deal with the pointer
    PointerType *PtrTy = cast<PointerType>(DataPtr->getType());
    Type *EltTy = PtrTy->getElementType();

    Type *RetType = VectorType::get(EltTy, Rows * Columns);

    Value *Ops[] = {DataPtr, Stride, B.getInt32(Rows), B.getInt32(Columns)};
    Type *OverloadedTypes[] = {RetType, PtrTy};

    Function *TheFn = Intrinsic::getDeclaration(
        getModule(), Intrinsic::matrix_columnwise_load, OverloadedTypes);

    return B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, Name);
  }

  /// Create a columnwise, strided matrix store.
  /// \p Matrix  - Matrix to store
  /// \p Ptr     - Pointer to write back to
  /// \p Stride  - Space between columns
  CallInst *CreateMatrixColumnwiseStore(Value *Matrix, Value *Ptr,
                                        Value *Stride, unsigned Rows,
                                        unsigned Columns,
                                        const Twine &Name = "") {
    Value *Ops[] = {Matrix, Ptr, Stride, B.getInt32(Rows), B.getInt32(Columns)};
    Type *OverloadedTypes[] = {Matrix->getType(), Ptr->getType()};

    Function *TheFn = Intrinsic::getDeclaration(
        getModule(), Intrinsic::matrix_columnwise_store, OverloadedTypes);

    return B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, Name);
  }

  /// Create a llvm.matrix.transpose call, transposing \p Matrix with \p Rows
  /// rows and \p Columns columns.
  CallInst *CreateMatrixTranspose(Value *Matrix, unsigned Rows,
                                  unsigned Columns, const Twine &Name = "") {
    auto *OpType = cast<VectorType>(Matrix->getType());
    Type *ReturnType =
        VectorType::get(OpType->getElementType(), Rows * Columns);

    Type *OverloadedTypes[] = {ReturnType};
    Value *Ops[] = {Matrix, B.getInt32(Rows), B.getInt32(Columns)};
    Function *TheFn = Intrinsic::getDeclaration(
        getModule(), Intrinsic::matrix_transpose, OverloadedTypes);

    return B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, Name);
  }

  /// Create a llvm.matrix.multiply call, multiplying matrixes \p LHS and \p
  /// RHS.
  CallInst *CreateMatrixMultiply(Value *LHS, Value *RHS, unsigned LHSRows,
                                 unsigned LHSColumns, unsigned RHSColumns,
                                 const Twine &Name = "") {
    auto *LHSType = cast<VectorType>(LHS->getType());
    auto *RHSType = cast<VectorType>(RHS->getType());

    Type *ReturnType =
        VectorType::get(LHSType->getElementType(), LHSRows * RHSColumns);

    Value *Ops[] = {LHS, RHS, B.getInt32(LHSRows), B.getInt32(LHSColumns),
                    B.getInt32(RHSColumns)};
    Type *OverloadedTypes[] = {ReturnType, LHSType, RHSType};

    Function *TheFn = Intrinsic::getDeclaration(
        getModule(), Intrinsic::matrix_multiply, OverloadedTypes);
    return B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, Name);
  }

  /// Insert a single element \p NewVal into \p Matrix at indices (\p RowIdx, \p
  /// ColumnIdx).
  Value *CreateMatrixInsert(Value *Matrix, Value *NewVal, Value *RowIdx,
                            Value *ColumnIdx, unsigned NumRows) {
    return B.CreateInsertElement(
        Matrix, NewVal,
        B.CreateAdd(B.CreateMul(ColumnIdx, ConstantInt::get(
                                               ColumnIdx->getType(), NumRows)),
                    RowIdx));
  }

  /// Add matrixes \p LHS and \p RHS. Support both integer and floating point
  /// matrixes.
  Value *CreateAdd(Value *LHS, Value *RHS) {
    assert(LHS->getType()->isVectorTy() || RHS->getType()->isVectorTy());
    if (LHS->getType()->isVectorTy() && !RHS->getType()->isVectorTy())
      RHS = B.CreateVectorSplat(
          cast<VectorType>(LHS->getType())->getNumElements(), RHS,
          "scalar.splat");
    else if (!LHS->getType()->isVectorTy() && RHS->getType()->isVectorTy())
      LHS = B.CreateVectorSplat(
          cast<VectorType>(RHS->getType())->getNumElements(), LHS,
          "scalar.splat");

    return cast<VectorType>(LHS->getType())
                   ->getElementType()
                   ->isFloatingPointTy()
               ? B.CreateFAdd(LHS, RHS)
               : B.CreateAdd(LHS, RHS);
  }

  /// Subtract matrixes \p LHS and \p RHS. Support both integer and floating
  /// point matrixes.
  Value *CreateSub(Value *LHS, Value *RHS) {
    assert(LHS->getType()->isVectorTy() || RHS->getType()->isVectorTy());
    if (LHS->getType()->isVectorTy() && !RHS->getType()->isVectorTy())
      RHS = B.CreateVectorSplat(
          cast<VectorType>(LHS->getType())->getNumElements(), RHS,
          "scalar.splat");
    else if (!LHS->getType()->isVectorTy() && RHS->getType()->isVectorTy())
      LHS = B.CreateVectorSplat(
          cast<VectorType>(RHS->getType())->getNumElements(), LHS,
          "scalar.splat");

    return cast<VectorType>(LHS->getType())
                   ->getElementType()
                   ->isFloatingPointTy()
               ? B.CreateFSub(LHS, RHS)
               : B.CreateSub(LHS, RHS);
  }

  /// Multiply matrix \p LHS with scalar \p RHS.
  Value *CreateScalarMultiply(Value *LHS, Value *RHS) {
    Value *ScalarVector =
        B.CreateVectorSplat(cast<VectorType>(LHS->getType())->getNumElements(),
                            RHS, "scalar.splat");
    if (RHS->getType()->isFloatingPointTy())
      return B.CreateFMul(LHS, ScalarVector);

    return B.CreateMul(LHS, ScalarVector);
  }

  /// Extracts the element at (\p RowIdx, \p ColumnIdx) from \p Matrix.
  Value *CreateExtractElement(Value *Matrix, Value *RowIdx, Value *ColumnIdx,
                              unsigned NumRows, Twine const &Name = "") {

    unsigned MaxWidth = std::max(RowIdx->getType()->getScalarSizeInBits(),
                                 ColumnIdx->getType()->getScalarSizeInBits());
    Type *IntTy = IntegerType::get(RowIdx->getType()->getContext(), MaxWidth);
    RowIdx = B.CreateZExt(RowIdx, IntTy);
    ColumnIdx = B.CreateZExt(ColumnIdx, IntTy);
    Value *NumRowsV = B.getIntN(MaxWidth, NumRows);
    return B.CreateExtractElement(
        Matrix, B.CreateAdd(B.CreateMul(ColumnIdx, NumRowsV), RowIdx),
        "matext");
  }
};

} // end namespace llvm

#endif // LLVM_IR_MATRIXBUILDER_H
