//===- LoopAnnotationAttrInterface.td ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for attributes on loop-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPANNOTATIONATTRINTERFACE
#define MLIR_INTERFACES_LOOPANNOTATIONATTRINTERFACE

include "mlir/IR/OpBase.td"

def LoopAnnotationAttrInterface : AttrInterface<"LoopAnnotationAttrInterface"> {
  let description = [{
    Metadata that should live on loop-like operations.
  }];
  let cppNamespace = "::mlir";

  let methods = [
  ];
}

#endif // MLIR_INTERFACES_LOOPANNOTATIONATTRINTERFACE
