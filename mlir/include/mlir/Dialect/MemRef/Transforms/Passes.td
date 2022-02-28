//===-- Passes.td - MemRef transformation definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ExpandOps : Pass<"memref-expand", "FuncOp"> {
  let summary = "Legalize memref operations to be convertible to LLVM.";
  let constructor = "mlir::memref::createExpandOpsPass()";
}

def FoldSubViewOps : Pass<"fold-memref-subview-ops"> {
  let summary = "Fold memref.subview ops into consumer load/store ops";
  let description = [{
    The pass folds loading/storing from/to subview ops to loading/storing
    from/to the original memref.
  }];
  let constructor = "mlir::memref::createFoldSubViewOpsPass()";
  let dependentDialects = [
      "AffineDialect", "memref::MemRefDialect", "vector::VectorDialect"
  ];
}

def NormalizeMemRefs : Pass<"normalize-memrefs", "ModuleOp"> {
  let summary = "Normalize memrefs";
   let description = [{
    This pass transforms memref types with a non-trivial
    [layout map](https://mlir.llvm.org/docs/LangRef/#layout-map) into
    memref types with an identity layout map, e.g. (i, j) -> (i, j). This
    pass is inter-procedural, in the sense that it can modify function
    interfaces and call sites that pass memref types. In order to modify
    memref types while preserving the original behavior, users of those
    memref types are also modified to incorporate the resulting layout map.
    For instance, an [AffineLoadOp]
    (https://mlir.llvm.org/docs/Dialects/Affine/#affineload-affineloadop)
    will be updated to compose the layout map with with the affine expression
    contained in the op. Operations marked with the [MemRefsNormalizable]
    (https://mlir.llvm.org/docs/Traits/#memrefsnormalizable) trait are
    expected to be normalizable. Supported operations include affine
    operations, memref.alloc, memref.dealloc, and func.return.

    Given an appropriate layout map specified in the code, this transformation
    can express tiled or linearized access to multi-dimensional data
    structures, but will not modify memref types without an explicit layout
    map.

    Currently this pass is limited to only modify
    functions where all memref types can be normalized. If a function
    contains any operations that are not MemRefNormalizable, then the function
    and any functions that call or call it will not be modified.

    Input

    ```mlir
    #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
    func @matmul(%A: memref<16xf64, #tile>,
                 %B: index, %C: memref<16xf64>) -> (memref<16xf64, #tile>) {
      affine.for %arg3 = 0 to 16 {
            %a = affine.load %A[%arg3] : memref<16xf64, #tile>
            %p = arith.mulf %a, %a : f64
            affine.store %p, %A[%arg3] : memref<16xf64, #tile>
      }
      %c = memref.alloc() : memref<16xf64, #tile>
      %d = affine.load %c[0] : memref<16xf64, #tile>
      return %A: memref<16xf64, #tile>
    }
    ```

    Output

    ```mlir
    func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
      -> memref<4x4xf64> {
      affine.for %arg3 = 0 to 16 {
        %3 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
        %4 = arith.mulf %3, %3 : f64
        affine.store %4, %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
      }
      %0 = memref.alloc() : memref<4x4xf64>
      %1 = affine.apply #map1()
      %2 = affine.load %0[0, 0] : memref<4x4xf64>
      return %arg0 : memref<4x4xf64>
    }
    ```

    Input

    ```
    #linear8 = affine_map<(i, j) -> (i * 8 + j)>
    func @linearize(%arg0: memref<8x8xi32, #linear8>,
                    %arg1: memref<8x8xi32, #linear8>,
                    %arg2: memref<8x8xi32, #linear8>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      affine.for %arg3 = %c0 to %c8  {
      affine.for %arg4 = %c0 to %c8  {
        affine.for %arg5 = %c0 to %c8 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<8x8xi32, #linear8>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xi32, #linear8>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
          %3 = arith.muli %0, %1 : i32
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
        }
      }
      }
      return
    }
    ```

    Output

    ```mlir
    func @linearize(%arg0: memref<64xi32>,
                    %arg1: memref<64xi32>,
                    %arg2: memref<64xi32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    affine.for %arg3 = %c0 to %c8 {
      affine.for %arg4 = %c0 to %c8 {
        affine.for %arg5 = %c0 to %c8 {
          %0 = affine.load %arg0[%arg3 * 8 + %arg5] : memref<64xi32>
          %1 = affine.load %arg1[%arg5 * 8 + %arg4] : memref<64xi32>
          %2 = affine.load %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
          %3 = arith.muli %0, %1 : i32
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
        }
      }
    }
    return
  }
  ```
  }];
  let constructor = "mlir::memref::createNormalizeMemRefsPass()";
  let dependentDialects = ["AffineDialect"];
}

def ResolveRankedShapeTypeResultDims :
    Pass<"resolve-ranked-shaped-type-result-dims"> {
  let summary = "Resolve memref.dim of result values of ranked shape type";
  let description = [{
    The pass resolves memref.dim of result of operations that
    implement the `ReifyRankedShapedTypeOpInterface` in terms of
    shapes of its operands.
  }];
  let constructor =
      "mlir::memref::createResolveRankedShapeTypeResultDimsPass()";
  let dependentDialects = [
    "memref::MemRefDialect", "tensor::TensorDialect"
  ];
}

def ResolveShapedTypeResultDims : Pass<"resolve-shaped-type-result-dims"> {
  let summary = "Resolve memref.dim of result values";
  let description = [{
    The pass resolves memref.dim of result of operations that
    implement the `InferShapedTypeOpInterface` or
    `ReifyRankedShapedTypeOpInterface` in terms of shapes of its
    operands.
  }];
  let constructor = "mlir::memref::createResolveShapedTypeResultDimsPass()";
  let dependentDialects = [
    "AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect"
  ];
}

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES

