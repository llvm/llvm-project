//===- BytecodeWriter.h - MLIR Bytecode Writer ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to write MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEWRITER_H
#define MLIR_BYTECODE_BYTECODEWRITER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Write the bytecode for the given operation to the provided output stream.
/// For streams where it matters, the given stream should be in "binary" mode.
/// `producer` is an optional string that can be used to identify the producer
/// of the bytecode when reading. It has no functional effect on the bytecode
/// serialization.
void writeBytecodeToFile(Operation *op, raw_ostream &os,
                         StringRef producer = "MLIR" LLVM_VERSION_STRING);

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEWRITER_H
