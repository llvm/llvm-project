//===- PythonTestCAPI.cpp - C API for the PythonTest dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "PythonTestDialect.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test,
                                      python_test::PythonTestDialect)

bool mlirAttributeIsAPythonTestTestAttribute(MlirAttribute attr) {
  return llvm::isa<python_test::TestAttrAttr>(unwrap(attr));
}

MlirAttribute mlirPythonTestTestAttributeGet(MlirContext context) {
  return wrap(python_test::TestAttrAttr::get(unwrap(context)));
}

MlirTypeID mlirPythonTestTestAttributeGetTypeID(void) {
  return wrap(python_test::TestAttrAttr::getTypeID());
}

bool mlirTypeIsAPythonTestTestType(MlirType type) {
  return llvm::isa<python_test::TestTypeType>(unwrap(type));
}

MlirType mlirPythonTestTestTypeGet(MlirContext context) {
  return wrap(python_test::TestTypeType::get(unwrap(context)));
}

MlirTypeID mlirPythonTestTestTypeGetTypeID(void) {
  return wrap(python_test::TestTypeType::getTypeID());
}

bool mlirTypeIsAPythonTestTestTensorValue(MlirValue value) {
  return mlirTypeIsATensor(wrap(unwrap(value).getType()));
}
