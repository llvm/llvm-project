//===-- MemRefAttrs.td - MemRef Attributes definition file -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_ATTRDEFS
#define MEMREF_ATTRDEFS

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/AttrTypeBase.td"

// All of the attributes will extend this class.
class MemRef_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<MemRef_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// AliasScopeAttr
//===----------------------------------------------------------------------===//

def MemRef_AliasScopeAttr : MemRef_Attr<"AliasScope", "alias_scope"> {
  let parameters = (ins
    "Attribute":$id,
    OptionalParameter<"StringAttr">:$description
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "MLIRContext*":$context,
      CArg<"StringAttr", "{}">:$description
    ), [{
      return $_get(context, DistinctAttr::create(UnitAttr::get(context)), description);
    }]>
  ];

  let summary = "TBD";

  let description = [{
    TBD
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;

  // Generate mnemonic alias for the attribute.
  let genMnemonicAlias = 1;
}

def MemRef_AliasScopeArrayAttr
    : TypedArrayAttrBase<MemRef_AliasScopeAttr,
                         MemRef_AliasScopeAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// AliasAttr
//===----------------------------------------------------------------------===//

def MemRef_AliasingAttr : MemRef_Attr<"Aliasing", "aliasing"> {
  let parameters = (ins
    "ArrayAttr":$alias_scopes,
    "ArrayAttr":$noalias
  );

  let summary = "TBD";

  let description = [{
    TBD
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "MLIRContext*":$context,
      CArg<"ArrayRef<Attribute>", "{}">:$alias_scopes,
      CArg<"ArrayRef<Attribute>", "{}">:$noalias
    ), [{
      return $_get(context, ArrayAttr::get(context, alias_scopes), ArrayAttr::get(context, noalias));
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // MEMREF_ATTRDEFS
