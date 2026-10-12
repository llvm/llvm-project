//===- WasmSSABase.td - Base defs for wasmssa dialect -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WasmSSA_BASE
#define WasmSSA_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def WasmSSA_Dialect : Dialect {
  let name = "wasmssa";
  let cppNamespace = "::mlir::wasmssa";
  let description = [{
    The `wasmssa` dialect is intended to represent WebAssembly
    modules in SSA form for easier manipulation.
  }];
  let useDefaultTypePrinterParser = true;
}

#endif //WasmSSA_BASE
