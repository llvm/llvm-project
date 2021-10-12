//===- VectorInterfaces.td - Vector interfaces -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for operations on vectors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VECTORINTERFACES
#define MLIR_INTERFACES_VECTORINTERFACES

include "mlir/IR/OpBase.td"

def VectorUnrollOpInterface : OpInterface<"VectorUnrollOpInterface"> {
  let description = [{
    Encodes properties of an operation on vectors that can be unrolled.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the shape ratio of unrolling to the target vector shape
        `targetShape`. Return `None` if the op cannot be unrolled to the target
        vector shape.
      }],
      /*retTy=*/"::mlir::Optional<::mlir::SmallVector<int64_t, 4>>",
      /*methodName=*/"getShapeForUnroll",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op->getNumResults() == 1);
        auto vt = $_op.getResult().getType().
          template dyn_cast<::mlir::VectorType>();
        if (!vt)
          return ::mlir::None;
        ::mlir::SmallVector<int64_t, 4> res(vt.getShape().begin(), vt.getShape().end());
        return res;
      }]
    >,
  ];
}

def VectorTransferOpInterface : OpInterface<"VectorTransferOpInterface"> {
  let description = [{
    Encodes properties of a transfer read or write operation.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/"Return the `in_bounds` attribute name.",
      /*retTy=*/"::mlir::StringRef",
      /*methodName=*/"getInBoundsAttrName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/ [{ return "in_bounds"; }]
    >,
    StaticInterfaceMethod<
      /*desc=*/"Return the `permutation_map` attribute name.",
      /*retTy=*/"::mlir::StringRef",
      /*methodName=*/"getPermutationMapAttrName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/ [{ return "permutation_map"; }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return `true` if dimension `dim` is in-bounds. Return `false`
                 otherwise. }],
      /*retTy=*/"bool",
      /*methodName=*/"isDimInBounds",
      /*args=*/(ins "unsigned":$dim),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.isBroadcastDim(dim)
            || ($_op.in_bounds()
                && $_op.in_bounds()->template cast<::mlir::ArrayAttr>()[dim]
                                    .template cast<::mlir::BoolAttr>().getValue());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the memref or ranked tensor operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"source",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.source();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the vector operand or result.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"vector",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.vector();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the indices operands.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"indices",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.indices();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the permutation map.",
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"permutation_map",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.permutation_map();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if op involves a 0-d tensor/memref and a vector
        of shape {1}. This is temporary until we have 0-d vectors.
        // TODO: turn this into 0-d vectors + empty permutation_map.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isZeroD",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (getShapedType().getRank() > 0)
          return false;
        if (getVectorType().getShape() != ArrayRef<int64_t>{1})
          return false;
        AffineMap map = AffineMap::get(
          /*numDims=*/0, /*numSymbols=*/0,
          getAffineConstantExpr(0, $_op->getContext()));
        if ($_op.permutation_map() != map)
          return false;
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Returns true if the specified dimension is a broadcast. }],
      /*retTy=*/"bool",
      /*methodName=*/"isBroadcastDim",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto expr = $_op.permutation_map().getResult(idx);
        return expr.template isa<::mlir::AffineConstantExpr>() &&
               expr.template dyn_cast<::mlir::AffineConstantExpr>().getValue() == 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Returns true if at least one of the dimensions in the
                  permutation map is a broadcast.}],
      /*retTy=*/"bool",
      /*methodName=*/"hasBroadcastDim",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // 0-d transfers are not considered broadcasts but they need to be 
        // represented with a vector<1xt> until we have 0-d vectors.
        if ($_op.isZeroD()) return false;
        for (unsigned i = 0; i < $_op.permutation_map().getNumResults(); ++i) {
          if ($_op.isBroadcastDim(i))
            return true;
        }
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the `in_bounds` boolean ArrayAttr.",
      /*retTy=*/"::mlir::Optional<::mlir::ArrayAttr>",
      /*methodName=*/"in_bounds",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.in_bounds();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the ShapedType.",
      /*retTy=*/"::mlir::ShapedType",
      /*methodName=*/"getShapedType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/
        "return $_op.source().getType().template cast<::mlir::ShapedType>();"
    >,
    InterfaceMethod<
      /*desc=*/"Return the VectorType.",
      /*retTy=*/"::mlir::VectorType",
      /*methodName=*/"getVectorType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.vector().getType().template dyn_cast<::mlir::VectorType>();
        }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the mask type if the op has a mask.",
      /*retTy=*/"::mlir::VectorType",
      /*methodName=*/"getMaskType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.mask()
            ? ::mlir::vector::detail::transferMaskType(
                $_op.getVectorType(), $_op.permutation_map())
            : ::mlir::VectorType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return the number of dimensions that participate in the
                  permutation map.}],
      /*retTy=*/"unsigned",
      /*methodName=*/"getTransferRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/
        "return $_op.permutation_map().getNumResults();"
    >,
    InterfaceMethod<
      /*desc=*/[{ Return the number of leading shaped dimensions that do not
                  participate in the permutation map.}],
      /*retTy=*/"unsigned",
      /*methodName=*/"getLeadingShapedRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/
        "return $_op.getShapedType().getRank() - $_op.getTransferRank();"
    >,
    InterfaceMethod<
      /*desc=*/[{ Returns true if at least one of the dimensions may be
                  out-of-bounds.}],
      /*retTy=*/"bool",
      /*methodName=*/"hasOutOfBoundsDim",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (unsigned idx = 0, e = $_op.getTransferRank(); idx < e; ++idx)
          if (!$_op.isDimInBounds(idx))
            return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
      Helper function to account for the fact that `permutationMap` results and
      `op.indices` sizes may not match and may not be aligned. The first
      `getLeadingShapedRank()` indices may just be indexed and not
      transferred from/into the vector.
      For example:
      ```
         vector.transfer %0[%i, %j, %k, %c0] :
           memref<?x?x?x?xf32>, vector<2x4xf32>
      ```
      with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
      Provide a zip function to coiterate on 2 running indices: `resultIdx` and
      `indicesIdx` which accounts for this misalignment.
      }],
      /*retTy=*/"void",
      /*methodName=*/"zipResultAndIndexing",
      /*args=*/(ins "::llvm::function_ref<void(int64_t, int64_t)>":$fun),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (int64_t resultIdx = 0,
                   indicesIdx = $_op.getLeadingShapedRank(),
                   eResult = $_op.getTransferRank();
           resultIdx < eResult;
           ++resultIdx, ++indicesIdx)
        fun(resultIdx, indicesIdx);
      }]
    >,
  ];
}

#endif // MLIR_INTERFACES_VECTORINTERFACES
