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

def AddReflectionMapPass : Pass<"add-reflection-map", "ModuleOp"> {
  let summary =
      "Add a reflection map function to EmitC classes for runtime field lookup";
  let description = [{
        This pass adds a `getBufferForName` function to EmitC classes that enables 
        runtime lookup of field buffers by their string names. 
        This requires that the class has fields with attributes and a function named `execute`.
        The `fieldop` attribute is expected to be a dictionary where:
        - The keys are `namedAttribute`.
        - The values are arrays containing a single string attribute.


        Example:

        ```mlir
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
          emitc.func @execute() { ... }
        }

        // becomes:
        
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
          emitc.func @getBufferForName(%name : !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char*"> {
            %0 = "emitc.constant"() <{value = #emitc.opaque<"{  { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) },  { \22some_feature\22, reinterpret_cast<char*>(&fieldName1) } }">}> : () -> !emitc.opaque<"const std::map<std::string, char*>">
            %1 = call_opaque "find"(%0, %arg0) : (!emitc.opaque<"const std::map<std::string, char*>">, !emitc.opaque<"std::string_view">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
            %2 = call_opaque "end"(%0) : (!emitc.opaque<"const std::map<std::string, char*>">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
            %3 = call_opaque "operator=="(%1, %2) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">, !emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> i1
            %4 = "emitc.constant"() <{value = #emitc.opaque<"nullptr">}> : () -> !emitc.opaque<"char">
            %5 = call_opaque "second"(%1) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> !emitc.opaque<"char">
            %6 = conditional %3, %4, %5 : !emitc.opaque<"char">
            return %6 : !emitc.opaque<"char">
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
