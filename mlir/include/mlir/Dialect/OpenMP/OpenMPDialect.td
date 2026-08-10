//===- OpenMPDialect.td - OpenMP dialect definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_DIALECT
#define OPENMP_DIALECT

include "mlir/IR/DialectBase.td"

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "::mlir::omp";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect, ::mlir::func::FuncDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif  // OPENMP_DIALECT
