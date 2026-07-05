//===-- WebAssemblyMCAsmInfo.h - WebAssembly asm properties -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssemblyMCAsmInfo class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCASMINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCASMINFO_H

#include "llvm/MC/MCAsmInfoWasm.h"

namespace llvm {

class Triple;

class WebAssemblyMCAsmInfo final : public MCAsmInfoWasm {
public:
  explicit WebAssemblyMCAsmInfo(const Triple &T,
                                const MCTargetOptions &Options);
  ~WebAssemblyMCAsmInfo() override;
};

namespace WebAssembly {
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
}
} // end namespace llvm

#endif
