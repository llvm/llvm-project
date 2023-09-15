//===- GENXTypes.cpp - MLIR GENX Types ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXTypes.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <iterator>

using namespace mlir;
using namespace mlir::GENX;

//===----------------------------------------------------------------------===//
// JointMatrixType
//===----------------------------------------------------------------------===//

struct GENX::detail::JointMatrixTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned, unsigned, MatrixLayout>;

  static JointMatrixTypeStorage *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<JointMatrixTypeStorage>())
        JointMatrixTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, rows, columns, matrixLayout);
  }

  JointMatrixTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), rows(std::get<1>(key)),
        columns(std::get<2>(key)), matrixLayout(std::get<3>(key)) {}

  Type elementType;
  unsigned rows;
  unsigned columns;
  MatrixLayout matrixLayout;
};

JointMatrixType JointMatrixType::get(Type elementType, unsigned rows,
                                     unsigned columns,
                                     MatrixLayout matrixLayout) {
  return Base::get(elementType.getContext(), elementType, rows, columns,
                   matrixLayout);
}

Type JointMatrixType::getElementType() const { return getImpl()->elementType; }

MatrixLayout JointMatrixType::getMatrixLayout() const {
  return getImpl()->matrixLayout;
}

unsigned JointMatrixType::getNumRows() const { return getImpl()->rows; }

unsigned JointMatrixType::getNumColumns() const { return getImpl()->columns; }

//===----------------------------------------------------------------------===//
// GENX Dialect
//===----------------------------------------------------------------------===//

void GENXDialect::registerTypes() { addTypes<JointMatrixType>(); }
