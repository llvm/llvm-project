//===- IRDLTypes.td - IRDL Types ---------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types IRDL uses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLTYPES
#define MLIR_DIALECT_IRDL_IR_IRDLTYPES

include "mlir/IR/AttrTypeBase.td"
include "IRDL.td"

class IRDL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IRDL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IRDL_AttributeType : IRDL_Type<"Attribute", "attribute"> {
  let summary = "IRDL handle to an `mlir::Attribute`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Attribute`,
    so it can be used in an IRDL operation, type, or attribute definition.
    This type can also represent a handle to an instance of an `mlir::Type`,
    by wrapping it in a `mlir::TypeAttr`. 

    Example:

    ```mlir
    irdl.dialect cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.operands(%1)
        irdl.results(%0)
      }
    }
    ```

    Here, `%0` and `%1` are both of type `!irdl.attribute`. Note that in
    particular, `%1` will be a handle to a `mlir::TypeAttr` wrapping an
    instance of a `cmath.complex` type.
  }];
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLTYPES
