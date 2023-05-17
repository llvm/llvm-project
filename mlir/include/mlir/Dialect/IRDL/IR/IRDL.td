//===- IRDL.td - IR Definition Language Dialect ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IR Definition Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDL
#define MLIR_DIALECT_IRDL_IR_IRDL

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

def IRDL_Dialect : Dialect {
  let summary = "IR Definition Language Dialect";
  let description = [{
    IRDL is an SSA-based declarative representation of dynamic dialects.
    It allows the definition of dialects, operations, attributes, and types,
    with a declarative description of their verifiers. IRDL code is meant to
    be generated and not written by hand. As such, the design focuses on ease
    of generation/analysis instead of ease of writing/reading.

    Users can define a new dialect with `irdl.dialect`, operations with
    `irdl.operation`, types with `irdl.type`, and attributes with
    `irdl.attribute`.

    An example dialect is shown below:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is_type : f32
        %1 = irdl.is_type : f64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }

      irdl.operation @mul {
        %0 = irdl.is_type : f32
        %1 = irdl.is_type : f64
        %2 = irdl.any_of(%0, %1)
        %3 = irdl.parametric_type : "cmath.complex"<%2>
        irdl.operands(%3, %3)
        irdl.results(%3)
      }
    }
    ```

    This program defines a `cmath` dialect that defines a `complex` type, and
    a `mul` operation. Both express constraints over their parameters using
    SSA constraint operations. Informally, one can see those SSA values as
    constraint variables that evaluate to a single type at constraint
    evaluation. For example, the result of the `irdl.any_of` stored in `%2`
    in the `mul` operation will collapse into either `f32` or `f64` for the
    entirety of this instance of `mul` constraint evaluation. As such,
    both operands and the result of `mul` must be of equal type (and not just
    satisfy the same constraint).

    IRDL variables are handle over `mlir::Attribute`. In order to support
    manipulating `mlir::Type`, IRDL wraps all types in an `mlir::TypeAttr`
    attribute. The rationale of this is to simplify the dialect.
  }];

  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;

  let name = "irdl";
  let cppNamespace = "::mlir::irdl";
}

#endif // MLIR_DIALECT_IRDL_IR_IRDL
