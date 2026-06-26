//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
#define MLIR_DIALECT_EMITC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormExpressionsPass : Pass<"form-expressions"> {
  let summary = "Form C-style expressions from C-operator ops";
  let description = [{
    The pass wraps emitc ops modelling C operators in emitc.expression ops and
    then folds single-use expressions into their users where possible.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
}

def WrapFuncInClassPass : Pass<"wrap-emitc-func-in-class", "ModuleOp"> {
  let summary = "Wrap functions in classes, using arguments as fields.";
  let description = [{
    This pass transforms `emitc.func` operations into `emitc.class` operations.
    Function arguments become fields of the class, and the function body is moved
    to a new member method within the class. By default, this is `operator()()`.

    If the corresponding function argument has attributes (accessed via `argAttrs`), 
    these attributes are attached to the field operation. 
    Otherwise, the field is created without additional attributes.

    Example:
    
    ```mlir
    emitc.func @model(%input_data : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}) attributes { } {
      %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
      %1 = subscript %input_data[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
      return
    }
    // becomes 
    emitc.class @modelClass {
      emitc.field @input_tensor : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}
      emitc.func @operator() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @input_tensor : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    ```
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [
      Option<"funcName", "func-name", "std::string",
      /*default=*/[{"operator()"}],
      "The name of the newly generated member function with body "
      "matching the input function.">
  ];
}

def MLGOAddReflectionMapPass : Pass<"mlgo-add-reflection-map", "ModuleOp"> {
  let summary = "Add a reflection map and a helper method to EmitC classes for runtime field lookup.";
  let description = [{
    This pass adds a `reflectionMap` field and an accompanying `getBufferForName` method to 
    EmitC classes, enabling runtime lookup of class fields by name. 
    This requires that the class has fields with attributes.
    Each `emitc.field` is expected to have an attribute (configured by the
    `field-attr-name` option) that is an array containing a single string
    attribute. If a field possesses both the attribute specified by
    `field-attr-name` and an attribute specified in `excluded-field-attrs`,
    the `field-attr-name` attribute takes precedence and the field will be
    included in the reflection map.
    
    Example:

    ```mlir
    emitc.class @MyClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
      emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
      emitc.func @execute() { ... }
    }

    Becomes:

    emitc.class @MyClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
      emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
      emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
        #emitc.opaque<"{ { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) }, { \22some_feature\22, reinterpret_cast<char*>(&fieldName1) } }">
      emitc.func @getBufferForName(%arg0: !emitc.opaque<"std::string">)
          -> !emitc.ptr<!emitc.opaque<"char">> {
        %0 = get_field @reflectionMap
          : !emitc.opaque<"const std::map<std::string, char*>">
        %1 = member_call_opaque %0 "at"(%arg0)
          : !emitc.opaque<"const std::map<std::string, char*>">,
            (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
        return %1 : !emitc.ptr<!emitc.opaque<"char">>
      }
      emitc.func @execute() { ... }
    }
    ```
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [
    Option<"fieldAttrName", "field-attr-name", "std::string",
      /*default=*/"",
      "Attribute key used to extract field names from a field's "
      "attribute dictionary">,
    ListOption<"excludedFieldAttrs", "excluded-field-attrs", "std::string",
      "Attribute keys that, if present on a field, exclude it "
      "from the reflection map">
  ];
}

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
