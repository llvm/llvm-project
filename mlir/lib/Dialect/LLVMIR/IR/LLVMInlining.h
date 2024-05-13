//===- LLVMInlining.h - Registration of LLVMInlinerInterface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows registering the LLVM DialectInlinerInterface with the LLVM dialect
// during initialization.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLVMIR_IR_LLVMINLINING_H
#define DIALECT_LLVMIR_IR_LLVMINLINING_H

namespace mlir {
namespace LLVM {

class LLVMDialect;

namespace detail {

/// Register the `LLVMInlinerInterface` implementation of
/// `DialectInlinerInterface` with the LLVM dialect.
void addLLVMInlinerInterface(LLVMDialect *dialect);

} // namespace detail

} // namespace LLVM
} // namespace mlir

#endif // DIALECT_LLVMIR_IR_LLVMINLINING_H
