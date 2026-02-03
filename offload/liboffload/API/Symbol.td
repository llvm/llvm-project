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

def ol_symbol_kind_t : Enum {
  let desc = "The kind of a symbol";
  let etors =[
    Etor<"KERNEL", "a kernel object">,
    Etor<"GLOBAL_VARIABLE", "a global variable">,
  ];
}

def olGetSymbol : Function {
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

def ol_symbol_info_t : Enum {
  let desc = "Supported symbol info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"KIND", "ol_symbol_kind_t", "The kind of this symbol.">,
    TaggedEtor<"GLOBAL_VARIABLE_ADDRESS", "void *", "The address in memory for this global variable.">,
    TaggedEtor<"GLOBAL_VARIABLE_SIZE", "size_t", "The size in bytes for this global variable.">,
  ];
}

def olGetSymbolInfo : Function {
  let desc = "Queries the given property of the symbol.";
  let details = [
    "`olGetSymbolInfoSize` can be used to query the storage size "
    "required for the given query."
  ];
  let params = [
    Param<"ol_symbol_handle_t", "Symbol", "handle of the symbol", PARAM_IN>,
    Param<"ol_symbol_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If PropSize is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_SYMBOL_KIND", [
      "If the requested info isn't applicable to the type of symbol."
    ]>,
    Return<"OL_ERRC_INVALID_SYMBOL">
  ];
}

def olGetSymbolInfoSize : Function {
  let desc = "Returns the storage size of the given symbol query.";
  let details = [];
  let params = [
    Param<"ol_symbol_handle_t", "Symbol", "handle of the symbol", PARAM_IN>,
    Param<"ol_symbol_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SYMBOL">,
    Return<"OL_ERRC_SYMBOL_KIND", [
      "If the requested info isn't applicable to the type of symbol."
    ]>,
  ];
}
