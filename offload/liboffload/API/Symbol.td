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
    let name = "olGetGlobalVariable";
    let desc = "Get a global variable named `GlobalName` in the given program.";
    let details = [
        "Symbol handles are owned by the program, so do not need to be destroyed."
    ];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
        Param<"const char*", "GlobalName", "name of the global variable in the program", PARAM_IN>,
        Param<"ol_symbol_handle_t*", "Global", "output pointer for the fetched global", PARAM_OUT>
    ];
    let returns = [];
}
