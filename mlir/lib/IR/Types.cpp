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
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

unsigned Type::getKind() const { return impl->getKind(); }

Dialect &Type::getDialect() const {
  return impl->getAbstractType().getDialect();
}

MLIRContext *Type::getContext() const { return getDialect().getContext(); }

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

FunctionType FunctionType::get(TypeRange inputs, TypeRange results,
                               MLIRContext *context) {
  return Base::get(context, Type::Kind::Function, inputs, results);
}

unsigned FunctionType::getNumInputs() const { return getImpl()->numInputs; }

ArrayRef<Type> FunctionType::getInputs() const {
  return getImpl()->getInputs();
}

unsigned FunctionType::getNumResults() const { return getImpl()->numResults; }

ArrayRef<Type> FunctionType::getResults() const {
  return getImpl()->getResults();
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

OpaqueType OpaqueType::get(Identifier dialect, StringRef typeData,
                           MLIRContext *context) {
  return Base::get(context, Type::Kind::Opaque, dialect, typeData);
}

OpaqueType OpaqueType::getChecked(Identifier dialect, StringRef typeData,
                                  MLIRContext *context, Location location) {
  return Base::getChecked(location, Kind::Opaque, dialect, typeData);
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
