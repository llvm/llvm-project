//===-- TestDialect.td - Test dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TESTPARAMETRIC_DIALECT
#define TESTPARAMETRIC_DIALECT

include "mlir/IR/OpBase.td"

def TestParametric_Dialect : Dialect {
  let name = "testparametric";
  let cppNamespace = "::testparametric";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
 private:
  }];
}

#endif // TESTPARAMETRIC_DIALECT
