//===- CIRTypes.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_TYPES
#define MLIR_CIR_DIALECT_CIR_TYPES

include "mlir/Dialect/CIR/IR/CIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic> : TypeDef<CIR_Dialect,
                                                           name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def CIR_PointerType :
    CIR_Type<"Pointer", "ptr"> {

  let summary = "CIR pointer type";
  let description = [{
    `CIR.ptr` is a type returned by any op generating a pointer in C++.
  }];

  let parameters = (ins "mlir::Type":$pointee);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// One type to bind them all
//===----------------------------------------------------------------------===//

def CIR_AnyCIRType : AnyTypeOf<[CIR_PointerType]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
