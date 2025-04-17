//===- ExportSMTLIB.cpp - C Interface to ExportSMTLIB ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a C Interface for export SMTLIB.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target/ExportSMTLIB.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"

using namespace mlir;

MlirLogicalResult mlirTranslateOperationToSMTLIB(MlirOperation module,
                                                 MlirStringCallback callback,
                                                 void *userData,
                                                 bool inlineSingleUseValues,
                                                 bool indentLetBody) {
  mlir::detail::CallbackOstream stream(callback, userData);
  smt::SMTEmissionOptions options;
  options.inlineSingleUseValues = inlineSingleUseValues;
  options.indentLetBody = indentLetBody;
  return wrap(smt::exportSMTLIB(unwrap(module), stream));
}

MlirLogicalResult mlirTranslateModuleToSMTLIB(MlirModule module,
                                              MlirStringCallback callback,
                                              void *userData,
                                              bool inlineSingleUseValues,
                                              bool indentLetBody) {
  return mlirTranslateOperationToSMTLIB(mlirModuleGetOperation(module),
                                        callback, userData,
                                        inlineSingleUseValues, indentLetBody);
}
