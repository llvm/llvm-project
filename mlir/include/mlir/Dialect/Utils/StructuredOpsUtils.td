//===- StructuredOpsUtils.td - structured ops enums --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_OPS_UTILS
#define STRUCTURED_OPS_UTILS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def IteratorType : I32EnumAttr<"IteratorType", "Iterator type", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"reduction", 1>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::utils";
}

#endif // STRUCTURED_OPS_UTILS
