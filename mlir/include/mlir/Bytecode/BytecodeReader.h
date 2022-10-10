//===- BytecodeReader.h - MLIR Bytecode Reader ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEREADER_H
#define MLIR_BYTECODE_BYTECODEREADER_H

#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"

namespace llvm {
class MemoryBufferRef;
} // namespace llvm

namespace mlir {
/// Returns true if the given buffer starts with the magic bytes that signal
/// MLIR bytecode.
bool isBytecode(llvm::MemoryBufferRef buffer);

/// Read the operations defined within the given memory buffer, containing MLIR
/// bytecode, into the provided block.
LogicalResult readBytecodeFile(llvm::MemoryBufferRef buffer, Block *block,
                               const ParserConfig &config);
} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEREADER_H
