//===- Types.cpp - MLIR Type Classes --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Types.h"
#include "TypeDetail.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

Dialect &Type::getDialect() const {
  return impl->getAbstractType().getDialect();
}

MLIRContext *Type::getContext() const { return getDialect().getContext(); }

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

FunctionType FunctionType::get(TypeRange inputs, TypeRange results,
                               MLIRContext *context) {
  return Base::get(context, inputs, results);
}

unsigned FunctionType::getNumInputs() const { return getImpl()->numInputs; }

ArrayRef<Type> FunctionType::getInputs() const {
  return getImpl()->getInputs();
}

unsigned FunctionType::getNumResults() const { return getImpl()->numResults; }

ArrayRef<Type> FunctionType::getResults() const {
  return getImpl()->getResults();
}

/// Helper to call a callback once on each index in the range
/// [0, `totalIndices`), *except* for the indices given in `indices`.
/// `indices` is allowed to have duplicates and can be in any order.
inline void iterateIndicesExcept(unsigned totalIndices,
                                 ArrayRef<unsigned> indices,
                                 function_ref<void(unsigned)> callback) {
  llvm::BitVector skipIndices(totalIndices);
  for (unsigned i : indices)
    skipIndices.set(i);

  for (unsigned i = 0; i < totalIndices; ++i)
    if (!skipIndices.test(i))
      callback(i);
}

/// Returns a new function type without the specified arguments and results.
FunctionType
FunctionType::getWithoutArgsAndResults(ArrayRef<unsigned> argIndices,
                                       ArrayRef<unsigned> resultIndices) {
  ArrayRef<Type> newInputTypes = getInputs();
  SmallVector<Type, 4> newInputTypesBuffer;
  if (!argIndices.empty()) {
    unsigned originalNumArgs = getNumInputs();
    iterateIndicesExcept(originalNumArgs, argIndices, [&](unsigned i) {
      newInputTypesBuffer.emplace_back(getInput(i));
    });
    newInputTypes = newInputTypesBuffer;
  }

  ArrayRef<Type> newResultTypes = getResults();
  SmallVector<Type, 4> newResultTypesBuffer;
  if (!resultIndices.empty()) {
    unsigned originalNumResults = getNumResults();
    iterateIndicesExcept(originalNumResults, resultIndices, [&](unsigned i) {
      newResultTypesBuffer.emplace_back(getResult(i));
    });
    newResultTypes = newResultTypesBuffer;
  }

  return get(newInputTypes, newResultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

OpaqueType OpaqueType::get(Identifier dialect, StringRef typeData,
                           MLIRContext *context) {
  return Base::get(context, dialect, typeData);
}

OpaqueType OpaqueType::getChecked(Identifier dialect, StringRef typeData,
                                  MLIRContext *context, Location location) {
  return Base::getChecked(location, dialect, typeData);
}

/// Returns the dialect namespace of the opaque type.
Identifier OpaqueType::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

/// Returns the raw type data of the opaque type.
StringRef OpaqueType::getTypeData() const { return getImpl()->typeData; }

/// Verify the construction of an opaque type.
LogicalResult OpaqueType::verifyConstructionInvariants(Location loc,
                                                       Identifier dialect,
                                                       StringRef typeData) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitError(loc, "invalid dialect namespace '") << dialect << "'";
  return success();
}
