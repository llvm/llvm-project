//===- SMTTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace smt;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SMT/IR/SMTTypes.cpp.inc"

void SMTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SMT/IR/SMTTypes.cpp.inc"
      >();
}

bool smt::isAnyNonFuncSMTValueType(Type type) {
  return isAnySMTValueType(type) && !isa<SMTFuncType>(type);
}

bool smt::isAnySMTValueType(Type type) {
  return isa<BoolType, BitVectorType, ArrayType, IntType, SortType,
             SMTFuncType>(type);
}

//===----------------------------------------------------------------------===//
// BitVectorType
//===----------------------------------------------------------------------===//

LogicalResult
BitVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                      int64_t width) {
  if (width <= 0U)
    return emitError() << "bit-vector must have at least a width of one";
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type domainType, Type rangeType) {
  if (!isAnySMTValueType(domainType))
    return emitError() << "domain must be any SMT value type";
  if (!isAnySMTValueType(rangeType))
    return emitError() << "range must be any SMT value type";

  return success();
}

//===----------------------------------------------------------------------===//
// SMTFuncType
//===----------------------------------------------------------------------===//

LogicalResult SMTFuncType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<Type> domainTypes, Type rangeType) {
  if (domainTypes.empty())
    return emitError() << "domain must not be empty";
  if (!llvm::all_of(domainTypes, isAnyNonFuncSMTValueType))
    return emitError() << "domain types must be any non-function SMT type";
  if (!isAnyNonFuncSMTValueType(rangeType))
    return emitError() << "range type must be any non-function SMT type";

  return success();
}

//===----------------------------------------------------------------------===//
// SortType
//===----------------------------------------------------------------------===//

LogicalResult SortType::verify(function_ref<InFlightDiagnostic()> emitError,
                               StringAttr identifier,
                               ArrayRef<Type> sortParams) {
  if (!llvm::all_of(sortParams, isAnyNonFuncSMTValueType))
    return emitError()
           << "sort parameter types must be any non-function SMT type";

  return success();
}
