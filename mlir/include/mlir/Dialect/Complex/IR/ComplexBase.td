//===- ComplexBase.td - Base definitions for complex dialect -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPLEX_BASE
#define COMPLEX_BASE

include "mlir/IR/OpBase.td"

def Complex_Dialect : Dialect {
  let name = "complex";
  let cppNamespace = "::mlir::complex";
  let description = [{
    The complex dialect is intended to hold complex numbers creation and
    arithmetic ops.
  }];
}

#endif // COMPLEX_BASE
