//===- PDLBase.td - PDL base definitions -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base support for MLIR PDL operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDLBASE
#define MLIR_DIALECT_PDL_IR_PDLBASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PDL Dialect
//===----------------------------------------------------------------------===//

def PDL_Dialect : Dialect {
  string summary = "High level pattern definition dialect";
  string description = [{
    PDL presents a high level abstraction for the rewrite pattern infrastructure
    available in MLIR. This abstraction allows for representing patterns
    transforming MLIR, as MLIR. This allows for applying all of the benefits
    that the general MLIR infrastructure provides, to the infrastructure itself.
    This means that pattern matching can be more easily verified for
    correctness, targeted by frontends, and optimized.

    PDL abstracts over various different aspects of patterns and core MLIR data
    structures. Patterns are specified via a `pdl.pattern` operation. These
    operations contain a region body for the "matcher" code, and terminate with
    a `pdl.rewrite` that either dispatches to an external rewriter or contains
    a region for the rewrite specified via `pdl`. The types of values in `pdl`
    are handle types to MLIR C++ types, with `!pdl.attribute`, `!pdl.operation`,
    and `!pdl.type` directly mapping to `mlir::Attribute`, `mlir::Operation*`,
    and `mlir::Value` respectively.

    An example pattern is shown below:

    ```mlir
    // pdl.pattern contains metadata similarly to a `RewritePattern`.
    pdl.pattern : benefit(1) {
      // External input operand values are specified via `pdl.input` operations.
      // Result types are constrainted via `pdl.type` operations.

      %resultType = pdl.type
      %inputOperand = pdl.input
      %root, %results = pdl.operation "foo.op"(%inputOperand) -> %resultType
      pdl.rewrite %root {
        pdl.replace %root with (%inputOperand)
      }
    }
    ```

    The above pattern simply replaces an operation with its first operand. Note
    how the input operation is specified structurally, similarly to how it would
    look in memory. This is a simple example and pdl provides support for many
    other features such as applying external constraints or external generator
    methods. These features and more are detailed below.
  }];

  let name = "pdl";
  let cppNamespace = "::mlir::pdl";
}

//===----------------------------------------------------------------------===//
// PDL Types
//===----------------------------------------------------------------------===//

class PDL_Handle<string underlying> :
    DialectType<PDL_Dialect, CPred<"$_self.isa<" # underlying # ">()">,
                underlying>,
    BuildableType<"$_builder.getType<" # underlying # ">()">;

// Handle for `mlir::Attribute`.
def PDL_Attribute : PDL_Handle<"mlir::pdl::AttributeType">;

// Handle for `mlir::Operation*`.
def PDL_Operation : PDL_Handle<"mlir::pdl::OperationType">;

// Handle for `mlir::Type`.
def PDL_Type : PDL_Handle<"mlir::pdl::TypeType">;

// Handle for `mlir::Value`.
def PDL_Value : PDL_Handle<"mlir::pdl::ValueType">;

// A positional value is a location on a pattern DAG, which may be an operation,
// an attribute, or an operand/result.
def PDL_PositionalValue :
    AnyTypeOf<[PDL_Attribute, PDL_Operation, PDL_Type, PDL_Value],
              "Positional Value">;

#endif // MLIR_DIALECT_PDL_IR_PDLBASE
