//===- LinalgPatterns.td - Linalg transformation patterns --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the pattern definition file for declarative Linalg transformation.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TRANSFORMS
#define LINALG_TRANSFORMS

include "mlir/Dialect/Linalg/IR/LinalgOps.td"
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"
include "mlir/Dialect/AffineOps/AffineOps.td"

def HasNoLinalgTransformMarker : CPred<[{
  !op.getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker)
}]>;

class HasLinalgTransformMarker<string str> : CPred<[{
  op.getAttrOfType<StringAttr>(
    LinalgTransforms::kLinalgTransformMarker) &&
  op.getAttrOfType<StringAttr>(
    LinalgTransforms::kLinalgTransformMarker).getValue() == "}] # str # [{"}]>;

class IsProducedByOpOfType<string str> :
  CPred<"isProducedByOpOfType<" # str # ">(op, $0)">;

class AffineMapDomainHasDim<int n> : CPred<[{
  op.getAttrOfType<ArrayAttr>(getIndexingMapsAttrName()).getValue()[0].
  cast<AffineMapAttr>().getValue().getNumDims() ==}] # n # [{}]>;

class HasOperandsOfType<string type>: CPred<[{
    llvm::any_of(op.getOperands(),
        [](Value v) {
          return dyn_cast_or_null<}] # type # [{>(v.getDefiningOp());
        })
}]>;

//===----------------------------------------------------------------------===//
// Linalg fusion patterns.
//===----------------------------------------------------------------------===//
//
// In the future, tile sizes should be derived from op properties + machine
// description but we do not need to wait on this to start having useful
// patterns.
class TileAndFuseLinalgOp<
    list<int> sizes, list<int> operandIndices, string value> : NativeCodeCall<
  "if (failed(tileAndFuseLinalgOpAndSetMarker($_builder, op, {" #
  StrJoinInt<sizes>.result # "}, {" # StrJoinInt<operandIndices>.result # "}," #
      " \"" # value # "\")))" #
  "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg tiling patterns.
//===----------------------------------------------------------------------===//
//
// In the future, tile sizes should be derived from op properties + machine
// description but we do not need to wait on this to start having useful
// patterns.
// `permutation` is an optional parameter to specify the ordering of the
// tiled loops. If provided, it must be a list of integers with the same number
// of elements as `sizes`.
class TileLinalgOp<list<int> sizes, string value, list<int> permutation=[]> :
  NativeCodeCall<
    "if (failed(tileLinalgOpAndSetMarker($_builder, op, {" #
    StrJoinInt<sizes>.result # "}, \"" # value # "\", {" #
    StrJoinInt<permutation>.result # "})))" #
    "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg to loop patterns.
//===----------------------------------------------------------------------===//
class LinalgOpToLoops<string OpType> : NativeCodeCall<
  "if (failed(linalgOpToLoops<" # OpType # ">($_builder, op))) " #
  "  return matchFailure();">;

class LinalgOpToParallelLoops<string OpType> : NativeCodeCall<
  "if (failed(linalgOpToParallelLoops<" # OpType # ">($_builder, op))) " #
  "  return matchFailure();">;

class LinalgOpToAffineLoops<string OpType> : NativeCodeCall<
  "if (failed(linalgOpToAffineLoops<" # OpType # ">($_builder, op))) " #
  "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg to vector patterns precondition and DRR.
//===----------------------------------------------------------------------===//
def PreconditionVectorizeLinalgOp : CPred<
  "succeeded(vectorizeLinalgOpPrecondition(op))">;
def VectorizeLinalgOp : NativeCodeCall<
  "vectorizeLinalgOp($_builder, op)">;


//===----------------------------------------------------------------------===//
// Linalg generic permutation patterns precondition and DRR.
//===----------------------------------------------------------------------===//
class PreconditionPermuteGenericLinalgOp<list<int> permutation> : CPred<
  "succeeded(permuteGenericLinalgOpPrecondition(op, {" #
  StrJoinInt<permutation>.result # "}))">;
class PermuteGenericLinalgOp<list<int> permutation, string value> :
  NativeCodeCall<
    "permuteGenericLinalgOp($_builder, op, {" # StrJoinInt<permutation>.result #
    "}, \"" # value # "\")">;

//===----------------------------------------------------------------------===//
// Linalg promote subview operands precondition and DRR.
//===----------------------------------------------------------------------===//
def PreconditionPromoteSubviewsLinalgOp : CPred<
  "succeeded(promoteSubviewsLinalgOpPrecondition(op))">;
def PromoteSubviewsLinalgOp : NativeCodeCall<
  "promoteSubviewsLinalgOp($_builder, op)">;
#endif // LINALG_TRANSFORMS
