//===- Polynomial.h - Storage class details for polynomial types --------*-
// C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALDETAIL_H_
#define INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALDETAIL_H_

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace polynomial {
namespace detail {

// A Polynomial is stored as an ordered list of monomial terms, each of which
// is a tuple of coefficient and exponent.
struct PolynomialStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<PolynomialStorage, Monomial> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, ArrayRef<Monomial>>;

  unsigned numTerms;

  MLIRContext *context;

  /// The monomial terms for this polynomial.
  ArrayRef<Monomial> terms() const {
    return {getTrailingObjects<Monomial>(), numTerms};
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numTerms && std::get<1>(key) == terms();
  }

  // Constructs a PolynomialStorage from a key. The context must be set by the
  // caller.
  static PolynomialStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto terms = std::get<1>(key);
    auto byteSize = PolynomialStorage::totalSizeToAlloc<Monomial>(terms.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(PolynomialStorage));
    auto *res = new (rawMem) PolynomialStorage();
    res->numTerms = std::get<0>(key);
    std::uninitialized_copy(terms.begin(), terms.end(),
                            res->getTrailingObjects<Monomial>());
    return res;
  }
};

} // namespace detail
} // namespace polynomial
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::polynomial::detail::PolynomialStorage)

#endif // INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALDETAIL_H_
