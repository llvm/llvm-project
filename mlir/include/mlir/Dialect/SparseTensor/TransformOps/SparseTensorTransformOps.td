//=== SparseTensorTransformOps.td ----------------------------*-tablegen *-===//
//
// Tablegen rules for Sparse Tensor transformation operations.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_TRANSFORM_OPS
#define SPARSETENSOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SparseTensorMatchOps
//===----------------------------------------------------------------------===//

def MatchSparseInOut : Op<Transform_Dialect, "sparse_tensor.match.sparse_inout", [
     MatchOpInterface,
     SingleOpMatcher,
     MemoryEffectsOpInterface]> {
  let description = [{
     Checks if the payload op has any sparse inputs and/or outputs.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $target attr-dict `:`
    custom<SemiFunctionType>(type($target), type($result), "false")
  }];
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration # [{
    ::mlir::Value getOperandHandle() { return getTarget(); }
  }];
}

#endif // SPARSETENSOR_TRANSFORM_OPS
