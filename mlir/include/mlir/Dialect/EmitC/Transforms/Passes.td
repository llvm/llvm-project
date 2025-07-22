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

def WrapFuncInClassPass : Pass<"wrap-emitc-func-in-class"> {
  let summary = "Wrap functions in classes, using arguments as fields.";
  let description = [{
    This pass transforms `emitc.func` operations into `emitc.class` operations.
    Function arguments become fields of the class, and the function body is moved
    to a new `execute` method within the class.
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
      emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @input_tensor : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    ```
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
}

def AddReflectionMapPass : Pass<"add-reflection-map"> {
  let summary =
      "Add a reflection map function to EmitC classes for runtime field lookup";
  let description = [{
        This pass adds a `getBufferForName` function to EmitC classes that enables 
        runtime lookup of field buffers by their string names. 
        This enables runtime introspection and dynamic access to class fields by name,
        which is useful for interfacing with external systems that need to access 
        tensors/buffers by their semantic names.

        Example transformation:
        ```mlir
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
          emitc.func @execute() { ... }
        }
        ```

        Becomes:
        ```mlir
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
          emitc.func @getBufferForName(%name : !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char*"> {
            %map = "emitc.constant"(){value = #emitc.opaque<"{"another_feature", reinterpret_cast<char*>(&another_feature)}, {"some_feature", reinterpret_cast<char*>(&some_feature)}">} : () -> !emitc.opaque<"std::map<std::string, char*>">
            return %null : !emitc.opaque<"char*">
          }
          emitc.func @execute() { ... }
        }
        ```
    }];
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
  let options = [Option<"namedAttribute", "named-attribute", "std::string",
                        /*default=*/"",
                        "Attribute key used to extract field names from fields "
                        "dictionary attributes">];
}

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
