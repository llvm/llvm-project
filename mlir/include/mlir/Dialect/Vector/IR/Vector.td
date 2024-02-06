//===- Vector.td - Vector Dialect --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOR
#define MLIR_DIALECT_VECTOR_IR_VECTOR

include "mlir/IR/OpBase.td"

def Vector_Dialect : Dialect {
  let name = "vector";
  let cppNamespace = "::mlir::vector";

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = ["arith::ArithDialect"];
}

// Base class for Vector dialect ops.
class Vector_Op<string mnemonic, list<Trait> traits = []> :
    Op<Vector_Dialect, mnemonic, traits>;

#endif // MLIR_DIALECT_VECTOR_IR_VECTOR
