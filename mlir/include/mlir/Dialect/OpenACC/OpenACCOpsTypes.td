//===- OpenACCOpsTypes.td - OpenACC operation types definitions -*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines MLIR OpenACC operation types.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS_TYPES
#define OPENACC_OPS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/OpenACC/OpenACCBase.td"

class OpenACC_Type<string name, string typeMnemonic> : TypeDef<OpenACC_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def OpenACC_DataBoundsType : OpenACC_Type<"DataBounds", "data_bounds_ty"> {
  let summary = "Type for representing acc data clause bounds information";
}

def OpenACC_DeclareTokenType : OpenACC_Type<"DeclareToken", "declare_token"> {
  let summary = "declare token type";
  let description = [{
    `acc.declare_token` is a type returned by a `declare_enter` operation and
    can be passed to a `declare_exit` operation to represent an implicit
    data region.
  }];
}

#endif // OPENACC_OPS_TYPES
