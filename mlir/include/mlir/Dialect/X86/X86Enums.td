//===-- X86Enums.td - X86 dialect enum definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the enums of the X86 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86_ENUMS
#define X86_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// MLUTarget
//===----------------------------------------------------------------------===//

// The target CPU feature for the vector.contract multi-level unroll pattern
// determines native shapes and supported data types for operands and the
// accumulator, and corresponds to one of the nanokernel patterns.
def MLUTarget : I32EnumAttr<"MLUTarget",
    "Target CPU feature for the vector.contract multi-level unroll pattern", [
  I32EnumAttrCase<"AMX_BF16", 0, "amx-bf16">, //-> VectorContractToAMXDotProduct
  I32EnumAttrCase<"AMX_INT8", 1, "amx-int8">, //-> VectorContractToAMXDotProduct
]> {
  let cppNamespace = "::mlir::x86";
  let genSpecializedAttr = 0;
}

#endif // X86_ENUMS
