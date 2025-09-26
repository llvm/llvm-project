//===- TestAnalysisOps.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_ANALYSIS_OPS
#define TEST_ANALYSIS_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

/// Transform dialect operations for testing analysis in MLIR

def TestGetBackwardSlice :
    Op<Transform_Dialect, "test.get_backward_slice",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>,
         ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Test `getBackwardSlice` by cloning the slice starting at `op`.
  }];

  let arguments =
    (ins TransformHandleTypeInterface:$op);
  
  let results = (outs);

  let assemblyFormat = [{
    $op attr-dict `:` type($op)
  }];
}

#endif // TEST_ANALYSIS_OPS
