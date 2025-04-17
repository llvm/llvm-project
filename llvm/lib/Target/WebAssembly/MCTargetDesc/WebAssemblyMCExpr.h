//===- WebAssembly specific MC expression classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines WebAssembly-specific relocation specifiers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCEXPR_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCEXPR_H

namespace llvm::WebAssembly {
enum Specifier {
  S_None,
  S_FUNCINDEX, // Wasm function index
  S_GOT,
  S_GOT_TLS,   // Wasm global index of TLS symbol
  S_MBREL,     // Memory address relative to __memory_base
  S_TBREL,     // Table index relative to __table_base
  S_TLSREL,    // Memory address relative to __tls_base
  S_TYPEINDEX, // Reference to a symbol's type (signature)
};
} // namespace llvm::WebAssembly

#endif
