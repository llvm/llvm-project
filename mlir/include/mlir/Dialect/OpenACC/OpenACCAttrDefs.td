//=== OpenACCAttrDefs.td - OpenACC Attributes definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_ATTR_DEFS
#define OPENACC_ATTR_DEFS

include "mlir/IR/OpBase.td"

class OpenACC_Attr<string name, string attrMnemonic,
                   list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<OpenACC_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

// Attribute to describe the declare data clause used on variable.
// Intended to be used at the variable creation site (on the global op or the
// corresponding allocation operation). This is used in conjunction with the
// declare operations (`acc.declare_enter` and `acc.declare_exit`) since those
// describe how the data action is performed. The attribute itself makes it
// easier to find out whether the variable is in a declare clause and what kind
// of clause it is.
def DeclareAttr : OpenACC_Attr<"Declare", "declare"> {
  let parameters = (ins "DataClauseAttr":$dataClause,
                        DefaultValuedParameter<"bool", "false">:$implicit);
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [AttrBuilder<(ins "DataClauseAttr":$dataClause), [{
      return $_get($_ctxt, dataClause, /*implicit=*/false);
    }]>
  ];
}

// Attribute to attach functions that perform the pre/post allocation actions or
// pre/post deallocation actions as described in section 2.13.
def DeclareActionAttr : OpenACC_Attr<"DeclareAction", "declare_action"> {
  let parameters = (ins OptionalParameter<"SymbolRefAttr">:$preAlloc,
                        OptionalParameter<"SymbolRefAttr">:$postAlloc,
                        OptionalParameter<"SymbolRefAttr">:$preDealloc,
                        OptionalParameter<"SymbolRefAttr">:$postDealloc);
  let assemblyFormat = "`<` struct(params) `>`";
}

// Attribute for saving variable names - this can be attached to non-acc-dialect
// operations in order to ensure the name is preserved.
def OpenACC_VarNameAttr : OpenACC_Attr<"VarName", "var_name"> {
  let parameters = (ins StringRefParameter<"">:$name);
  let assemblyFormat = "`<` $name `>`";
}

def RoutineInfoAttr : OpenACC_Attr<"RoutineInfo", "routine_info"> {
  let summary = "Keeps track of associated acc routine information";

  let description = [{
    This attribute is used to create the association between a function and
    its `acc.routine` operation. A `func.func` uses this if its name
    was referenced in an `acc routine` directive.
  }];

  let parameters = (ins ArrayRefParameter<"SymbolRefAttr", "">:$accRoutines);
  let assemblyFormat = "`<` `[` `` $accRoutines `]` `>`";
}

def SpecializedRoutineAttr : OpenACC_Attr<"SpecializedRoutine",
                                          "specialized_routine"> {
  let summary = "Marks a specialized device version of an acc routine";

  let description = [{
    This attribute is attached to a function that was specialized from a host
    function marked with `acc.routine_info`. It captures the parallelism level,
    a reference to the original `acc.routine` operation, and the original
    function name (since the specialized function may be renamed).

    Example - before specialization:
    ```mlir
    acc.routine @routine_gang func(@foo) gang
    acc.routine @routine_vector func(@foo) vector

    func.func @foo() attributes {
      acc.routine_info = #acc.routine_info<[@routine_gang, @routine_vector]>
    } { ... }
    ```

    After specialization, there are three functions: the original function and
    two specialized versions (one per parallelism level):
    ```mlir
    acc.routine @routine_gang func(@foo) gang
    acc.routine @routine_vector func(@foo) vector

    // Original function (unchanged)
    func.func @foo() attributes {
      acc.routine_info = #acc.routine_info<[@routine_gang, @routine_vector]>
    } { ... }

    // Specialized for gang parallelism
    func.func @foo_gang() attributes {
      acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "foo">
    } { ... }

    // Specialized for vector parallelism
    func.func @foo_vector() attributes {
      acc.specialized_routine = #acc.specialized_routine<@routine_vector, <vector>, "foo">
    } { ... }
    ```
  }];

  let parameters = (ins
    "SymbolRefAttr":$routine,
    "ParLevelAttr":$level,
    "StringAttr":$funcName
  );

  let assemblyFormat = "`<` $routine `,` $level `,` $funcName `>`";
}

#endif // OPENACC_ATTR_DEFS
