// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

func.func @promote_subview_matmul(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %2 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        linalg.matmul ins(%3, %4: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                  memref<?x?xf32, strided<[?, ?], offset: ?>>)
                     outs(%5: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      }
    }
  }
  return
}
// CHECK-LABEL: func @promote_subview_matmul
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c2000:.*]] = arith.constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = arith.constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = arith.constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:               %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:               %[[s1:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:               %[[s2:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:               %[[a0:.*]] = memref.alloc() : memref<32000000xi8>
// CHECK:               %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:               %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
// CHECK:               %[[a1:.*]] = memref.alloc() : memref<48000000xi8>
// CHECK:               %[[v1:.*]] = memref.view %[[a1]]{{.*}} : memref<48000000xi8> to memref<?x?xf32>
// CHECK:               %[[l1:.*]] = memref.subview %[[v1]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
// CHECK:               %[[a2:.*]] = memref.alloc() : memref<24000000xi8>
// CHECK:               %[[v2:.*]] = memref.view %[[a2]]{{.*}} : memref<24000000xi8> to memref<?x?xf32>
// CHECK:               %[[l2:.*]] = memref.subview %[[v2]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
// CHECK:               linalg.copy ins(%[[s0]] : memref<?x?xf32, strided{{.*}}>) outs(%[[l0]] : memref<?x?xf32, strided{{.*}}>)
// CHECK:               linalg.copy ins(%[[s1]] : memref<?x?xf32, strided{{.*}}>) outs(%[[l1]] : memref<?x?xf32, strided{{.*}}>)
// CHECK:               linalg.copy ins(%[[s2]] : memref<?x?xf32, strided{{.*}}>) outs(%[[l2]] : memref<?x?xf32, strided{{.*}}>)
// CHECK:               linalg.matmul
// CHECK-SAME:                 ins(%[[v0]], %[[v1]] : memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME:                outs(%[[v2]] : memref<?x?xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { operands_to_promote = [0, 1, 2], use_full_tiles_by_default } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @promote_first_subview_matmul(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  %2 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        linalg.matmul {__internal_linalg_transform__ = "_promote_first_view_"}
          ins(%3, %4: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                      memref<?x?xf32, strided<[?, ?], offset: ?>>)
         outs(%5: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      }
    }
  }
  return
}
// CHECK-LABEL: func @promote_first_subview_matmul
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c2000:.*]] = arith.constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = arith.constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = arith.constant 4000 : index
// CHECK:   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:         %[[s1:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:         %[[s2:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() : memref<32000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.view
// CHECK-NOT:     memref.subview
// CHECK:         linalg.copy ins(%[[s0]] : memref<?x?xf32, strided{{.*}}>) outs(%[[l0]] : memref<?x?xf32, strided{{.*}}>)
// CHECK-NOT:     linalg.copy
// CHECK:         linalg.matmul
// CHECK-SAME:           ins(%[[v0]], %[[s1]] : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
// CHECK-SAME:          outs(%[[s2]] : memref<?x?xf32, strided<[?, ?], offset: ?>>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { operands_to_promote = [0], use_full_tiles_by_default } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @aligned_promote_fill(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf = arith.constant 1.0 : f32
  %3 = memref.subview %arg0[%c0, %c0][%c2000, %c4000][%c1, %c1] :
         memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  linalg.fill
   ins(%cf : f32) outs(%3 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
  return
}
// CHECK-LABEL: func @aligned_promote_fill
// CHECK:         %[[cf:.*]] = arith.constant 1.{{.*}} : f32
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, strided{{.*}}> to memref<?x?xf32, strided{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() {alignment = 32 : i64} : memref<32000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
// CHECK:         linalg.fill ins({{.*}} : f32) outs(%[[v0]] : memref<?x?xf32>)
// CHECK:         linalg.copy ins(%[[s0]] : memref<?x?xf32, strided{{.*}}>) outs(%[[l0]] : memref<?x?xf32, strided{{.*}}>)
// CHECK:         linalg.fill ins(%[[cf]] : f32) outs(%[[v0]] : memref<?x?xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { operands_to_promote = [1], use_full_tile_buffers = [false, true], alignment = 32} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @aligned_promote_fill_complex(%arg0: memref<?x?xcomplex<f32>, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf = arith.constant 1.0 : f32
  %cc = complex.create %cf, %cf : complex<f32>
  %3 = memref.subview %arg0[%c0, %c0][%c2000, %c4000][%c1, %c1] :
         memref<?x?xcomplex<f32>, strided<[?, 1], offset: ?>> to memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>
  linalg.fill ins(%cc : complex<f32>)
             outs(%3 : memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>)
  return
}
// CHECK-LABEL: func @aligned_promote_fill_complex
// CHECK:         %[[cc:.*]] = complex.create {{.*}} : complex<f32>
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xcomplex<f32>, strided{{.*}}> to memref<?x?xcomplex<f32>, strided{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() {alignment = 32 : i64} : memref<64000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<64000000xi8> to memref<?x?xcomplex<f32>>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xcomplex<f32>> to memref<?x?xcomplex<f32>, strided<[?, 1]>>
// CHECK:         linalg.fill ins({{.*}} : complex<f32>) outs(%[[v0]] : memref<?x?xcomplex<f32>>)
// CHECK:         linalg.copy ins(%[[s0]] : memref<?x?xcomplex<f32>, strided{{.*}}>) outs(%[[l0]] : memref<?x?xcomplex<f32>, strided{{.*}}>)
// CHECK:         linalg.fill ins(%[[cc]] : complex<f32>) outs(%[[v0]] : memref<?x?xcomplex<f32>>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { operands_to_promote = [1], use_full_tile_buffers = [false, true], alignment = 32} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
