//===-- TosaBytecode.td - Tosa bytecode defs -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the TOSA bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_DIALECT_BYTECODE
#define TOSA_DIALECT_BYTECODE

include "mlir/IR/BytecodeBase.td"

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.

def TosaDialectTypes : DialectTypes<"Tosa"> {
  let elems = [];
}

#endif // TOSA_DIALECT_BYTECODE

