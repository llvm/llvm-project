//===-- PtrAttrDefs.td - Ptr Attributes definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_ATTRDEFS
#define PTR_ATTRDEFS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"

// All of the attributes will extend this class.
class Ptr_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Ptr_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// AliasScopeDomainAttr
//===----------------------------------------------------------------------===//

def Ptr_AliasScopeDomainAttr : Ptr_Attr<"AliasScopeDomain",
                                        "alias_scope_domain"> {
  let parameters = (ins
    "DistinctAttr":$id,
    OptionalParameter<"StringAttr">:$description
  );

  let builders = [
    AttrBuilder<(ins CArg<"StringAttr", "{}">:$description), [{
      return $_get($_ctxt,
                   DistinctAttr::create(UnitAttr::get($_ctxt)), description);
    }]>
  ];

  let summary = "Ptr dialect alias scope domain metadata";

  let description = [{
    Defines a domain that may be associated with an alias scope.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// AliasScopeAttr
//===----------------------------------------------------------------------===//

def Ptr_AliasScopeAttr : Ptr_Attr<"AliasScope", "alias_scope"> {
  let parameters = (ins
    "DistinctAttr":$id,
    "AliasScopeDomainAttr":$domain,
    OptionalParameter<"StringAttr">:$description
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "AliasScopeDomainAttr":$domain,
      CArg<"StringAttr", "{}">:$description
    ), [{
      MLIRContext *ctx = domain.getContext();
      return $_get(ctx, DistinctAttr::create(UnitAttr::get(ctx)), domain, description);
    }]>
  ];

  let description = [{
    Defines an alias scope that can be attached to a memory-accessing operation.
    Such scopes can be used in combination with `noalias` metadata to indicate
    that sets of memory-affecting operations in one scope do not alias with
    memory-affecting operations in another scope.

    Example:
    ```mlir
    #domain = #ptr.alias_scope_domain<id = distinct[1]<>, description = "Optional domain description">
    #scope1 = #ptr.alias_scope<id = distinct[2]<>, domain = #domain>
    #scope2 = #ptr.alias_scope<id = distinct[3]<>, domain = #domain, description = "Optional scope description">
    ptr.func @foo(%ptr1 : !ptr.ptr) {
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %c4 = llvm.mlir.constant(4 : i32) : i32
        %1 = ptr.ptrtoint %ptr1 : !ptr.ptr to i32
        %2 = llvm.add %1, %c1 : i32
        %ptr2 = ptr.inttoptr %2 : i32 to !ptr.ptr
        ptr.store %c0, %ptr1 { alias_scopes = [#scope1], ptr.noalias = [#scope2] } : i32, !ptr.ptr
        ptr.store %c4, %ptr2 { alias_scopes = [#scope2], ptr.noalias = [#scope1] } : i32, !ptr.ptr
        llvm.return
    }
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];

  let summary = "Ptr dialect alias scope";

  let assemblyFormat = "`<` struct(params) `>`";
}

def Ptr_AliasScopeArrayAttr
    : TypedArrayAttrBase<Ptr_AliasScopeAttr,
                         Ptr_AliasScopeAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// AccessGroupAttr
//===----------------------------------------------------------------------===//

def Ptr_AccessGroupAttr : Ptr_Attr<"AccessGroup", "access_group"> {

  let parameters = (ins "DistinctAttr":$id);

  let builders = [
    AttrBuilder<(ins), [{
      return $_get($_ctxt, DistinctAttr::create(UnitAttr::get($_ctxt)));
    }]>
  ];

  let summary = "Ptr dialect access group metadata";

  let description = [{
    Defines an access group metadata that can be set on any instruction
    that potentially accesses memory via the `AccessGroupOpInterface` or on
    branch instructions in the loop latch block via the `parallelAccesses`
    parameter of `LLVM::LoopAnnotationAttr`.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#llvm-access-group-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def Ptr_AccessGroupArrayAttr
    : TypedArrayAttrBase<Ptr_AccessGroupAttr,
                         Ptr_AccessGroupAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// TBAARootAttr
//===----------------------------------------------------------------------===//

def Ptr_TBAARootAttr : Ptr_Attr<"TBAARoot", "tbaa_root", [], "TBAANodeAttr"> {
  let parameters = (ins OptionalParameter<"StringAttr">:$id);

  let summary = "Ptr dialect TBAA root metadata";
  let description = [{
    Defines a TBAA root node.

    Example:
    ```mlir
    #cpp_root = #ptr.tbaa_root<identity = "Simple C/C++ TBAA">
    #other_root = #ptr.tbaa_root
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "(`<` struct(params)^ `>`)?";
}

//===----------------------------------------------------------------------===//
// TBAATypeDescriptorAttr
//===----------------------------------------------------------------------===//

def Ptr_TBAAMemberAttr : Ptr_Attr<"TBAAMember", "tbaa_member"> {
  let parameters = (ins
    "TBAANodeAttr":$typeDesc,
    "int64_t":$offset
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TBAANodeAttr":$typeDesc,
                                        "int64_t":$offset), [{
      return $_get(typeDesc.getContext(), typeDesc, offset);
    }]>
  ];

  let assemblyFormat = "`<` params `>`";
}

def Ptr_TBAAMemberAttrArray : ArrayRefParameter<"TBAAMemberAttr"> {
  let printer = [{
    $_printer << '{';
    llvm::interleaveComma($_self, $_printer, [&](TBAAMemberAttr attr) {
        $_printer.printStrippedAttrOrType(attr);
    });
    $_printer << '}';
  }];

  let parser = [{
    [&]() -> FailureOr<SmallVector<TBAAMemberAttr>> {
        using Result = SmallVector<TBAAMemberAttr>;
        if ($_parser.parseLBrace())
            return failure();
        FailureOr<Result> result = FieldParser<Result>::parse($_parser);
        if (failed(result))
            return failure();
        if ($_parser.parseRBrace())
            return failure();
        return result;
    }()
  }];
}

def Ptr_TBAATypeDescriptorAttr : Ptr_Attr<"TBAATypeDescriptor",
    "tbaa_type_desc", [], "TBAANodeAttr"> {
  let parameters = (ins
    StringRefParameter<>:$id,
    Ptr_TBAAMemberAttrArray:$members
  );

  let summary = "Ptr dialect TBAA type metadata";

  let description = [{
    Defines a TBAA node describing a type.

    Example:
    ```mlir
    #tbaa_root = #ptr.tbaa_root<identity = "Simple C/C++ TBAA">
    #tbaa_type_desc1 = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc2 = #ptr.tbaa_type_desc<id = "long long", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc3 = #ptr.tbaa_type_desc<id = "agg2_t", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 8>}>
    #tbaa_type_desc4 = #ptr.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc1, 0>}>
    #tbaa_type_desc5 = #ptr.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc4, 4>}>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TBAATagAttr
//===----------------------------------------------------------------------===//

def Ptr_TBAATagAttr : Ptr_Attr<"TBAATag", "tbaa_tag"> {
  let parameters = (ins
    "TBAATypeDescriptorAttr":$base_type,
    "TBAATypeDescriptorAttr":$access_type,
    "int64_t":$offset,
    DefaultValuedParameter<"bool", "false">:$constant
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TBAATypeDescriptorAttr":$baseType,
                                        "TBAATypeDescriptorAttr":$accessType,
                                        "int64_t":$offset), [{
      return $_get(baseType.getContext(), baseType, accessType, offset,
                    /*constant=*/false);
    }]>
  ];

  let summary = "Ptr dialect TBAA tag metadata";

  let description = [{
    Defines a TBAA node describing a memory access.

    Example:
    ```mlir
    #tbaa_root = #ptr.tbaa_root<identity = "Simple C/C++ TBAA">
    #tbaa_type_desc1 = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc2 = #ptr.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc1, 0>}>
    #tbaa_type_desc3 = #ptr.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc4, 4>}>
    #tbaa_tag = #ptr.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc2, offset = 0, constant = true>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def Ptr_TBAATagArrayAttr
    : TypedArrayAttrBase<Ptr_TBAATagAttr,
                         Ptr_TBAATagAttr.summary # " array"> {
  let constBuilderCall = ?;
}

#endif // PTR_ATTRDEFS
