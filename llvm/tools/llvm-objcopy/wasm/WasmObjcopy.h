//===- WasmObjcopy.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJCOPY_WASM_WASMOBJCOPY_H
#define LLVM_TOOLS_LLVM_OBJCOPY_WASM_WASMOBJCOPY_H

namespace llvm {
class Error;

namespace object {
class WasmObjectFile;
} // end namespace object

namespace objcopy {
struct CopyConfig;
class Buffer;

namespace wasm {
Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::WasmObjectFile &In, Buffer &Out);

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_LLVM_OBJCOPY_WASM_WASMOBJCOPY_H
