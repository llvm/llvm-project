//===- MPI.cpp - MPI dialect implementation -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mpi;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MPI/IR/MPI.cpp.inc"

#include "mlir/Dialect/MPI/IR/MPIDialect.cpp.inc"

void MPIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MPI/IR/MPIOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MPI/IR/MPITypesGen.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MPI/IR/MPIAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect, type, and op definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MPI/IR/MPITypesGen.cpp.inc"

#include "mlir/Dialect/MPI/IR/MPIEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MPI/IR/MPIAttrDefs.cpp.inc"
