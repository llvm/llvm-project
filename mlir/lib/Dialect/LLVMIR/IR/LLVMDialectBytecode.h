//===- LLVMDialectBytecode.h - LLVM Bytecode Implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines hooks into the LLVM dialect bytecode
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_DIALECT_LLVM_IR_LLVMDIALECTBYTECODE_H
#define LIB_MLIR_DIALECT_LLVM_IR_LLVMDIALECTBYTECODE_H

namespace mlir::LLVM {
class LLVMDialect;

namespace detail {
/// Add the interfaces necessary for encoding the LLVM dialect components in
/// bytecode.
void addBytecodeInterface(LLVMDialect *dialect);
} // namespace detail
} // namespace mlir::LLVM

#endif // LIB_MLIR_DIALECT_LLVM_IR_LLVMDIALECTBYTECODE_H
