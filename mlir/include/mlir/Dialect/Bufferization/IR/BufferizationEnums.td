//===- BufferizationEnums.td - Bufferization enums ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in Bufferization.
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_ENUMS
#define BUFFERIZATION_ENUMS

include "mlir/IR/EnumAttr.td"

def LayoutMapOption : I32EnumAttr<"LayoutMapOption", 
                                  "option for map layout", [
  I32EnumAttrCase<"InferLayoutMap", 0>,
  I32EnumAttrCase<"IdentityLayoutMap", 1>,
  I32EnumAttrCase<"FullyDynamicLayoutMap", 2>
]> {
  let cppNamespace = "::mlir::bufferization";
}

#endif // BUFFERIZATION_ENUMS
