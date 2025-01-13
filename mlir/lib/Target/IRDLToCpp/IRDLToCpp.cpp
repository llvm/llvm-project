//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"

using namespace mlir;

LogicalResult irdl::translateIRDLDialectToCppDeclHeader(irdl::DialectOp dialect,
                                                        raw_ostream &output) {
  return failure();
}

LogicalResult irdl::translateIRDLDialectToCppDef(irdl::DialectOp dialect,
                                                 raw_ostream &output) {
  return failure();
}
