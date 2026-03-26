// RUN: mlir-opt %s -transform-interpreter -canonicalize -split-input-file | FileCheck %s

func.func @gemm(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
   linalg.matmul ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
               outs(%c: memref<?x?xf32>)
   return
}

//      CHECK: func @gemm
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//      CHECK: scf.for
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:       %[[svA:.+]] = memref.subview %[[ARG0]]
//      CHECK:       %[[svB:.+]] = memref.subview %[[ARG1]]
//      CHECK:       %[[svC:.+]] = memref.subview %[[ARG2]]

//      CHECK:       %[[tmpA:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[VA:.*]] = memref.view %[[tmpA]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[svAA:.+]] = memref.subview %[[VA]]

//      CHECK:       %[[tmpC:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[VC:.*]] = memref.view %[[tmpC]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[svCC:.+]] = memref.subview %[[VC]]

//      CHECK:       linalg.copy ins(%[[svA]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[svAA]] : memref<?x?xf32, strided<[16, 1]>>)
//      CHECK:       linalg.copy ins(%[[svC]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[svCC]] : memref<?x?xf32, strided<[16, 1]>>)
//      CHECK:       linalg.matmul ins(%[[VA]], %[[svB]]{{.*}} outs(%[[VC]]
//      CHECK:       linalg.copy ins(%[[svCC]] : memref<?x?xf32, strided<[16, 1]>>) outs(%[[svC]] : memref<?x?xf32, strided<[?, 1], offset: ?>>)
//      CHECK:       memref.dealloc %[[tmpA]]
//      CHECK:       memref.dealloc %[[tmpC]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.promote %1 { operands_to_promote = [0, 2], force_full_tiles = [false, false], use_full_tiles_by_default } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_f32(%A: memref<512x256xf32>, %B: memref<256x512xf32>, %C: memref<256x256xf32>, %s0: index, %s1: index, %s2: index) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  scf.for %arg4 = %c0 to %c512 step %s0 {
    scf.for %arg5 = %c0 to %c512 step %s1 {
      scf.for %arg6 = %c0 to %c256 step %s2 {
        %i0 = affine.min affine_map<(d0)[s0] -> (-d0 + 512, s0)>(%arg4)[%s0]
        %i1 = affine.min affine_map<(d0)[s0] -> (-d0 + 512, s0)>(%arg5)[%s1]
        %i2 = affine.min affine_map<(d0)[s0] -> (-d0 + 256, s0)>(%arg6)[%s2]
        %0 = memref.subview %A[%arg4, %arg6][%i0, %i2][1, 1] : memref<512x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %1 = memref.subview %B[%arg6, %arg5][%i2, %i1][1, 1] : memref<256x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %2 = memref.subview %C[%arg4, %arg5][%i0, %i1][1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        linalg.matmul
          ins(%0, %1: memref<?x?xf32, strided<[256, 1], offset: ?>>,
                      memref<?x?xf32, strided<[512, 1], offset: ?>>)
          outs(%2: memref<?x?xf32, strided<[256, 1], offset: ?>>)
      }
    }
  }
  return
}

// CHECK-LABEL:   func.func @matmul_f32(
// CHECK-SAME:      %[[ARG0:.*]]: memref<512x256xf32>
// CHECK-SAME:      %[[ARG1:.*]]: memref<256x512xf32>
// CHECK-SAME:      %[[ARG2:.*]]: memref<256x256xf32>
// CHECK-SAME:      %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index

// CHECK:                 %[[i0:.*]] = affine.min
// CHECK:                 %[[i1:.*]] = affine.min
// CHECK:                 %[[i2:.*]] = affine.min

// CHECK:                 %[[VAL_13:.*]] = arith.muli %[[i0]], %[[i2]] : index
// CHECK:                 %[[VAL_14:.*]] = arith.muli %[[VAL_13]], %[[C4]] : index
// CHECK:                 %[[VAL_15:.*]] = memref.alloc(%[[VAL_14]]) : memref<?xi8>

// CHECK:                 %[[VAL_18:.*]] = arith.muli %[[i2]], %[[i1]] : index
// CHECK:                 %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[C4]] : index
// CHECK:                 %[[VAL_20:.*]] = memref.alloc(%[[VAL_19]]) : memref<?xi8>

// CHECK:                 %[[VAL_23:.*]] = arith.muli %[[i0]], %[[i1]] : index
// CHECK:                 %[[VAL_24:.*]] = arith.muli %[[VAL_23]], %[[C4]] : index
// CHECK:                 %[[VAL_25:.*]] = memref.alloc(%[[VAL_24]]) : memref<?xi8>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { use_original_subview_size } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
