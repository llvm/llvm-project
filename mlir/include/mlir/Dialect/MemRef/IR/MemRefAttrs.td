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

  let summary = "MemRef dialect alias scope";

  let description = [{
    Defines an alias scope that can be attached to a memory-accessing operation.
    Such scopes can be used in combination with `noalias` metadata to indicate
    that sets of memory-affecting operations in one scope do not alias with
    memory-affecting operations in another scope.

    This attribute must be used in conjunctions with
    `memref.alias_domain_scope`.

    Example:
    ```
    #alias_scope1 = #memref.alias_scope<id = distinct[0]<>, description = "scope">
    #alias_scope2 = #memref.alias_scope<id = distinct[1]<>>

    func.func @memref_alias_scope(%arg1 : memref<?xf32>, %arg2 : memref<?xf32>, %arg3: index) -> f32 {
      %0 = memref.alias_domain_scope "The Domain" -> f32 {
        %val = memref.load %arg1[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope1], noalias=[#alias_scope2]> } : memref<?xf32>
        memref.store %val, %arg2[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope2], noalias=[#alias_scope1]> } : memref<?xf32>
        memref.alias_domain_scope.return %val: f32
      }
      return %0 : f32
    }
    ```

    This attribute is modeled against LLVM alias scopes, see the following link
    for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
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

  let summary = "MemRef dialect aliasing attribute";

  let description = [{
    This attribute combines `alias_scopes` and `noalias` arrays into single
    attribute.
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
