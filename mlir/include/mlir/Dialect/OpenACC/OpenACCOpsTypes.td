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

#endif // OPENACC_OPS_TYPES
