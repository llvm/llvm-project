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

MlirLogicalResult mlirExportSMTLIB(MlirModule module,
                                   MlirStringCallback callback,
                                   void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(smt::exportSMTLIB(unwrap(module), stream));
}
