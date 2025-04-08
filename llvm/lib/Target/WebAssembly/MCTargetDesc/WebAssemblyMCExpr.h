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
  S_TYPEINDEX,
  S_TBREL,
  S_MBREL,
  S_TLSREL,
  S_GOT,
  S_GOT_TLS,
  S_FUNCINDEX,
};
} // namespace llvm::WebAssembly

#endif
