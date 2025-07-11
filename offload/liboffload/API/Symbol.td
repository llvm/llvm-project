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
    Etor<"GLOBAL_VARIABLE", "a global variable">,
  ];
}

def : Function {
    let name = "olGetSymbol";
    let desc = "Get a symbol (kernel or global variable) identified by `Name` in the given program.";
    let details = [
        "Symbol handles are owned by the program and do not need to be manually destroyed."
    ];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
        Param<"const char*", "Name", "name of the symbol to look up", PARAM_IN>,
        Param<"ol_symbol_kind_t", "Kind", "symbol kind to look up", PARAM_IN>,
        Param<"ol_symbol_handle_t*", "Symbol", "output pointer for the symbol", PARAM_OUT>,
    ];
    let returns = [];
}
