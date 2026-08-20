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
    This pass adds a `reflectionMap` field and an accompanying `getBufferForName`
    method to EmitC classes, enabling runtime lookup of class fields by name. 
    Each `emitc.field` is expected to have an attribute (configured by the
    `included-field-attrs` option) that is an array containing a single string
    attribute which will be used as the reflection map key. If none of the specified
    attributes are present on the field, the field will be excluded from the
    reflection map.
    
    Example:

    ```mlir
    emitc.class @MyClass {
      emitc.field @field0 : !emitc.array<1xf32> {emitc.field_ref = ["feature0"]}
      emitc.func @execute() { ... }
    }

    Becomes:

    emitc.class @MyClass {
      emitc.field @field0 : !emitc.array<1xf32> {emitc.field_ref = ["feature0"]}
      // reflectionMap maps the string "feature0" to the address of @field0
      emitc.field @reflectionMap : const std::map<string, char*> = {{"feature0", &field0}}
      emitc.func @getBufferForName(%name: string) -> char* {
        return reflectionMap.at(%name)
      }
      emitc.func @execute() { ... }
    }
    ```
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [
    ListOption<"includedFieldAttrs", "included-field-attrs", "std::string",
      "Attribute keys used to extract field names from a field's "
      "attribute dictionary">
  ];
}

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
