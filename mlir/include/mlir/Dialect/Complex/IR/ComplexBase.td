//===- ComplexBase.td - Base definitions for complex dialect -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPLEX_BASE
#define COMPLEX_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Complex_Dialect : Dialect {
  let name = "complex";
  let cppNamespace = "::mlir::complex";
  let description = [{
    The complex dialect is intended to hold complex numbers creation and
    arithmetic ops.
  }];

  let dependentDialects = ["arith::ArithDialect"];
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Complex_ComplexRangeFlags
//===----------------------------------------------------------------------===//

def Complex_CRF_improved  : I32BitEnumAttrCaseBit<"improved", 0>;
def Complex_CRF_basic : I32BitEnumAttrCaseBit<"basic", 1>;
def Complex_CRF_none  : I32BitEnumAttrCaseBit<"none", 2>;

def Complex_ComplexRangeFlags : I32BitEnumAttr<
    "ComplexRangeFlags",
    "Complex range flags",
    [Complex_CRF_improved, Complex_CRF_basic, Complex_CRF_none]> {
  let cppNamespace = "::mlir::complex";
}

#endif // COMPLEX_BASE
