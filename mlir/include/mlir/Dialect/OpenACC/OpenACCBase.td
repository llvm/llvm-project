//===- OpenACCBase.td - OpenACC dialect definition ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines MLIR OpenACC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_BASE
#define OPENACC_BASE

include "mlir/IR/AttrTypeBase.td"

def OpenACC_Dialect : Dialect {
  let name = "acc";

  let summary = "An OpenACC dialect for MLIR.";

  let description = [{
    This dialect models the construct from the OpenACC 3.3 directive language.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  let cppNamespace = "::mlir::acc";
  let dependentDialects = ["::mlir::memref::MemRefDialect","::mlir::LLVM::LLVMDialect"];
}

#endif // OPENACC_BASE
