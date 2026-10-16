//===- BuiltinDialect.h - MLIR Builtin Dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect that contains all of the attributes,
// operations, and types that are necessary for the validity of the IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINDIALECT_H_
#define MLIR_IR_BUILTINDIALECT_H_

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// BuiltinDialectVersion
//===----------------------------------------------------------------------===//

struct BuiltinDialectVersion : public mlir::DialectVersion {
  BuiltinDialectVersion(int64_t version) : version(version) {}

  int64_t getVersion() const { return version; }

  static BuiltinDialectVersion getCurrentVersion() { return {0}; }

  bool operator<(const BuiltinDialectVersion &other) const {
    return version < other.version;
  }

private:
  int64_t version;
};

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h.inc"

#endif // MLIR_IR_BUILTINDIALECT_H_
