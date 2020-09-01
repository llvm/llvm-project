//===- SDBMExprDetail.h - MLIR SDBM Expression storage details --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of SDBMExpr, in particular underlying
// storage types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SDBMEXPRDETAIL_H
#define MLIR_IR_SDBMEXPRDETAIL_H

#include "mlir/Dialect/SDBM/SDBMExpr.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class SDBMDialect;

namespace detail {

// Base storage class for SDBMExpr.
struct SDBMExprStorage : public StorageUniquer::BaseStorage {
  SDBMExprKind getKind() { return kind; }

  SDBMDialect *dialect;
  SDBMExprKind kind;
};

// Storage class for SDBM sum and stripe expressions.
struct SDBMBinaryExprStorage : public SDBMExprStorage {
  using KeyTy = std::tuple<unsigned, SDBMDirectExpr, SDBMConstantExpr>;

  bool operator==(const KeyTy &key) const {
    return static_cast<SDBMExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == lhs && std::get<2>(key) == rhs;
  }

  static SDBMBinaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMBinaryExprStorage>();
    result->lhs = std::get<1>(key);
    result->rhs = std::get<2>(key);
    result->dialect = result->lhs.getDialect();
    result->kind = static_cast<SDBMExprKind>(std::get<0>(key));
    return result;
  }

  SDBMDirectExpr lhs;
  SDBMConstantExpr rhs;
};

// Storage class for SDBM difference expressions.
struct SDBMDiffExprStorage : public SDBMExprStorage {
  using KeyTy = std::pair<SDBMDirectExpr, SDBMTermExpr>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == lhs && std::get<1>(key) == rhs;
  }

  static SDBMDiffExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMDiffExprStorage>();
    result->lhs = std::get<0>(key);
    result->rhs = std::get<1>(key);
    result->dialect = result->lhs.getDialect();
    result->kind = SDBMExprKind::Diff;
    return result;
  }

  SDBMDirectExpr lhs;
  SDBMTermExpr rhs;
};

// Storage class for SDBM constant expressions.
struct SDBMConstantExprStorage : public SDBMExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static SDBMConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMConstantExprStorage>();
    result->constant = key;
    result->kind = SDBMExprKind::Constant;
    return result;
  }

  int64_t constant;
};

// Storage class for SDBM dimension and symbol expressions.
struct SDBMTermExprStorage : public SDBMExprStorage {
  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<SDBMExprKind>(key.first) &&
           position == key.second;
  }

  static SDBMTermExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMTermExprStorage>();
    result->kind = static_cast<SDBMExprKind>(key.first);
    result->position = key.second;
    return result;
  }

  unsigned position;
};

// Storage class for SDBM negation expressions.
struct SDBMNegExprStorage : public SDBMExprStorage {
  using KeyTy = SDBMDirectExpr;

  bool operator==(const KeyTy &key) const { return key == expr; }

  static SDBMNegExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMNegExprStorage>();
    result->expr = key;
    result->dialect = key.getDialect();
    result->kind = SDBMExprKind::Neg;
    return result;
  }

  SDBMDirectExpr expr;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_IR_SDBMEXPRDETAIL_H
