//===- SMTAttributes.td - Attributes for SMT dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SMT dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_TD
#define MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def BitVectorAttr : AttrDef<SMTDialect, "BitVector", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "bv";
  let description = [{
    This attribute represents a constant value of the `(_ BitVec width)` sort as
    described in the [SMT bit-vector
    theory](https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml).

    The constant is as #bX (binary) or #xX (hexadecimal) in SMT-LIB
    where X is the value in the corresponding format without any further
    prefixing. Here, the bit-vector constant is given as a regular non-negative
    integer literal and the associated bit-vector type indicating the bit-width.

    Examples:
    ```mlir
    #smt.bv<5> : !smt.bv<4>
    #smt.bv<92> : !smt.bv<8>
    ```

    The explicit type-suffix is mandatory to uniquely represent the attribute,
    i.e., this attribute should always be used in the extended form (using the
    `quantified` keyword in the operation assembly format string).

    The bit-width must be greater than zero (i.e., at least one digit has to be
    present).
  }];

  let parameters = (ins APIntParameter<"">:$value);

  let hasCustomAssemblyFormat = true;
  let genVerifyDecl = true;

  let builders = [
    AttrBuilder<(ins "llvm::StringRef":$value)>,
    AttrBuilder<(ins "uint64_t":$value, "unsigned":$width)>,
  ];

  let extraClassDeclaration = [{
    /// Return the bit-vector constant as a SMT-LIB formatted string.
    std::string getValueAsString(bool prefix = true) const;
  }];
}

#endif // MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_TD
