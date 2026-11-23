//===--- llvm/CodeGen/WasmEHInfo.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data for Wasm exception handling schemes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WASMEHINFO_H
#define LLVM_CODEGEN_WASMEHINFO_H

namespace llvm {

namespace WebAssembly {
enum Tag { CPP_EXCEPTION = 0, C_LONGJMP = 1 };
} // namespace WebAssembly

} // namespace llvm

#endif // LLVM_CODEGEN_WASMEHINFO_H
