//===- Complex.cpp - C Interface for Complex dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Complex.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Complex, complex,
                                      mlir::complex::ComplexDialect)

bool mlirAttributeIsAComplex(MlirAttribute attr) {
  return isa<complex::NumberAttr>(unwrap(attr));
}

MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx, MlirType type,
                                       double real, double imag) {
  return wrap(
      complex::NumberAttr::get(cast<ComplexType>(unwrap(type)), real, imag));
}

MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation loc, MlirType type,
                                              double real, double imag) {
  return wrap(complex::NumberAttr::getChecked(
      unwrap(loc), cast<ComplexType>(unwrap(type)), real, imag));
}

double mlirComplexAttrGetRealDouble(MlirAttribute attr) {
  return cast<complex::NumberAttr>(unwrap(attr)).getReal().convertToDouble();
}

double mlirComplexAttrGetImagDouble(MlirAttribute attr) {
  return cast<complex::NumberAttr>(unwrap(attr)).getImag().convertToDouble();
}

MlirTypeID mlirComplexAttrGetTypeID(void) {
  return wrap(complex::NumberAttr::getTypeID());
}
