//===-- Symbol.td - Symbol definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the symbol handle.
//
//===----------------------------------------------------------------------===//

def : Enum {
  let name = "ol_symbol_kind_t";
  let desc = "The kind of a symbol";
  let etors =[
    Etor<"KERNEL", "a kernel object">,
  ];
}
