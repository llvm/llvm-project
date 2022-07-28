//===- ComplexAttributes.td - Definitions for complex attributes -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPLEX_ATTRIBUTE
#define COMPLEX_ATTRIBUTE

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Complex/IR/ComplexBase.td"

//===----------------------------------------------------------------------===//
// Complex Attributes.
//===----------------------------------------------------------------------===//

class Complex_Attr<string attrName, string attrMnemonic>
    : AttrDef<Complex_Dialect, attrName> {
  let mnemonic = attrMnemonic;
}

def Complex_NumberAttr : Complex_Attr<"Number", "number"> {
  let summary = "A complex number attribute";

  let description = [{
    A complex number attribute.

    Example:

    ```mlir
    #complex.number<:f64 1.0, 2.0>
    ```
  }];

  let parameters = (ins APFloatParameter<"">:$real, APFloatParameter<"">:$imag, AttributeSelfTypeParameter<"">:$type);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // COMPLEX_ATTRIBUTE
