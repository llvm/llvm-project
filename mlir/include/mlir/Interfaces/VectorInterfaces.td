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
      /*retTy=*/"Optional<SmallVector<int64_t, 4>>",
      /*methodName=*/"getShapeForUnroll",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.getOperation()->getNumResults() == 1);
        auto vt = $_op.getResult().getType().
          template dyn_cast<VectorType>();
        if (!vt)
          return None;
        SmallVector<int64_t, 4> res(vt.getShape().begin(), vt.getShape().end());
        return res;
      }]
    >,
  ];
}

def VectorTransferOpInterface : OpInterface<"VectorTransferOpInterface"> {
  let description = [{
    Encodes properties of an operation on vectors that can be unrolled.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/"Return the `masked` attribute name.",
      /*retTy=*/"StringRef",
      /*methodName=*/"getMaskedAttrName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/ [{ return "masked"; }]
    >,
    StaticInterfaceMethod<
      /*desc=*/"Return the `permutation_map` attribute name.",
      /*retTy=*/"StringRef",
      /*methodName=*/"getPermutationMapAttrName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/ [{ return "permutation_map"; }]
    >,
    InterfaceMethod<
      /*desc=*/[{
      Return `false` when the `masked` attribute at dimension
      `dim` is set to `false`. Return `true` otherwise.}],
      /*retTy=*/"bool",
      /*methodName=*/"isMaskedDim",
      /*args=*/(ins "unsigned":$dim),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return !$_op.masked() ||
          $_op.masked()->template cast<ArrayAttr>()[dim]
                        .template cast<BoolAttr>().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the memref operand.",
      /*retTy=*/"Value",
      /*methodName=*/"memref",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.memref();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the vector operand or result.",
      /*retTy=*/"Value",
      /*methodName=*/"vector",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.vector();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the indices operands.",
      /*retTy=*/"ValueRange",
      /*methodName=*/"indices",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.indices();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the permutation map.",
      /*retTy=*/"AffineMap",
      /*methodName=*/"permutation_map",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.permutation_map();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the `masked` boolean ArrayAttr.",
      /*retTy=*/"Optional<ArrayAttr>",
      /*methodName=*/"masked",
      /*args=*/(ins),
      /*methodBody=*/"return $_op.masked();"
      /*defaultImplementation=*/
    >,
    InterfaceMethod<
      /*desc=*/"Return the MemRefType.",
      /*retTy=*/"MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/
        "return $_op.memref().getType().template cast<MemRefType>();"
    >,
    InterfaceMethod<
      /*desc=*/"Return the VectorType.",
      /*retTy=*/"VectorType",
      /*methodName=*/"getVectorType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.vector().getType().template dyn_cast<VectorType>();
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
      /*desc=*/[{ Return the number of leading memref dimensions that do not
                  participate in the permutation map.}],
      /*retTy=*/"unsigned",
      /*methodName=*/"getLeadingMemRefRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/
        "return $_op.getMemRefType().getRank() - $_op.getTransferRank();"
    >,
    InterfaceMethod<
      /*desc=*/[{ Returns true if at least one of the dimensions is masked.}],
      /*retTy=*/"bool",
      /*methodName=*/"hasMaskedDim",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (unsigned idx = 0, e = $_op.getTransferRank(); idx < e; ++idx)
          if ($_op.isMaskedDim(idx))
            return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
      Helper function to account for the fact that `permutationMap` results and
      `op.indices` sizes may not match and may not be aligned. The first
      `getLeadingMemRefRank()` indices may just be indexed and not transferred
      from/into the vector.
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
      /*args=*/(ins "llvm::function_ref<void(int64_t, int64_t)>":$fun),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (int64_t resultIdx = 0,
                   indicesIdx = $_op.getLeadingMemRefRank(),
                   eResult = $_op.getTransferRank();
           resultIdx < eResult;
           ++resultIdx, ++indicesIdx)
        fun(resultIdx, indicesIdx);
      }]
    >,
  ];
}

#endif // MLIR_INTERFACES_VECTORINTERFACES
