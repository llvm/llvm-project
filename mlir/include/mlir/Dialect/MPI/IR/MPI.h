//===- MPI.h - MPI dialect ----------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MPI_IR_MPI_H_
#define MLIR_DIALECT_MPI_IR_MPI_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// MPIDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MPI/IR/MPIDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MPI/IR/MPITypesGen.h.inc"

#include "mlir/Dialect/MPI/IR/MPIEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MPI/IR/MPIAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MPI/IR/MPIOps.h.inc"

#endif // MLIR_DIALECT_MPI_IR_MPI_H_
