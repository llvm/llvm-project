// RUN: mlir-opt %s -test-linalg-transform-patterns=test-patterns -split-input-file -test-transform-dialect-interpreter | FileCheck %s

// Map corresponding to a 2D memory access where the stride along the last dim is known to be 1.
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$nm:.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[$km:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>

func.func @dot(%x: memref<?xf32, strided<[1], offset: ?>>,
          %y: memref<?xf32, strided<[1], offset: ?>>,
          %v: memref<f32>) {
  linalg.dot { __internal_linalg_transform__ = "MEM" }
    ins(%x, %y: memref<?xf32, strided<[1], offset: ?>>,
                memref<?xf32, strided<[1], offset: ?>>)
    outs(%v: memref<f32>)

  return
}
// CHECK-LABEL: func @dot
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[c8000:.*]] = arith.constant 8000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c1]] {
// CHECK:               load
// CHECK:               load
// CHECK:               load
// CHECK:               arith.mulf
// CHECK:               arith.addf
// CHECK:               store

func.func @matvec(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %x: memref<?xf32, strided<[1], offset: ?>>,
             %y: memref<?xf32, strided<[1], offset: ?>>) {
  linalg.matvec
    ins(%A, %x: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                memref<?xf32, strided<[1], offset: ?>>)
    outs(%y: memref<?xf32, strided<[1], offset: ?>>)
  return
}
// CHECK-LABEL: func @matvec
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[c6:.*]] = arith.constant 6 : index
// CHECK:         scf.parallel {{.*}} step (%[[c5]])
// CHECK:           scf.for {{.*}} step %[[c6]]
// CHECK:             linalg.matvec
// CHECK:               ins({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>)
// CHECK:              outs({{.*}}: memref<?xf32, strided<[1], offset: ?>>)

func.func @matmul(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %C: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  linalg.matmul { __internal_linalg_transform__ = "MEM" }
    ins(%A, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                memref<?x?xf32, strided<[?, 1], offset: ?>>)
    outs(%C: memref<?x?xf32, strided<[?, 1], offset: ?>>)
  return
}
// CHECK-LABEL: func @matmul
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[c20:.*]] = arith.constant 20 : index
// CHECK-DAG:     %[[c30:.*]] = arith.constant 30 : index
// CHECK-DAG:     %[[c40:.*]] = arith.constant 40 : index
// CHECK-DAG:     %[[c200:.*]] = arith.constant 200 : index
// CHECK-DAG:     %[[c300:.*]] = arith.constant 300 : index
// CHECK-DAG:     %[[c400:.*]] = arith.constant 400 : index
// CHECK-DAG:     %[[c2000:.*]] = arith.constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = arith.constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = arith.constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK:                 scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK:                   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK:                     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK:                       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK:                         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK:                           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK:                             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK:                               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK:                                 linalg.matmul
// CHECK:                                   ins({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>)
// CHECK:                                  outs({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>)

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#generic_matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = #matmul_accesses,
  library_call = "linalg_matmul",
  iterator_types = ["parallel", "parallel", "reduction"]
}
func.func @permute_generic(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
           %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
           %C: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  linalg.generic #generic_matmul_trait
    ins(%A, %B : memref<?x?xf32, strided<[?, 1], offset: ?>>,
                 memref<?x?xf32, strided<[?, 1], offset: ?>>)
   outs(%C : memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e: f32
  }
  return
}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    transform.structured.interchange %0 { iterator_interchange = [1, 2, 0]}
  }
}
// CHECK-LABEL:  func @permute_generic
// CHECK:        linalg.generic {
// CHECK-SAME:   indexing_maps = [#[[$kn]], #[[$nm]], #[[$km]]],
// CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"],
// CHECK-SAME:   library_call = "linalg_matmul"}
// CHECK:          memref<?x?xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>>

func.func @matvec_perm(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %x: memref<?xf32, strided<[1], offset: ?>>,
             %y: memref<?xf32, strided<[1], offset: ?>>) {
  linalg.matvec {__internal_linalg_transform__ = "__with_perm__"}
    ins(%A, %x: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                memref<?xf32, strided<[1], offset: ?>>)
   outs(%y: memref<?xf32, strided<[1], offset: ?>>)
  return
}
// CHECK-LABEL: func @matvec_perm
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[c6:.*]] = arith.constant 6 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c6]]
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c5]]
// CHECK:             linalg.matvec
// CHECK:               ins({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>)
// CHECK:              outs({{.*}}: memref<?xf32, strided<[1], offset: ?>>)

func.func @matmul_perm(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %C: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  linalg.matmul {__internal_linalg_transform__ = "__with_perm__"}
    ins(%A, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                memref<?x?xf32, strided<[?, 1], offset: ?>>)
   outs(%C : memref<?x?xf32, strided<[?, 1], offset: ?>>)
  return
}
// CHECK-LABEL: func @matmul_perm
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c20:.*]] = arith.constant 20 : index
// CHECK-DAG:     %[[c30:.*]] = arith.constant 30 : index
// CHECK-DAG:     %[[c40:.*]] = arith.constant 40 : index
// CHECK-DAG:     %[[c200:.*]] = arith.constant 200 : index
// CHECK-DAG:     %[[c300:.*]] = arith.constant 300 : index
// CHECK-DAG:     %[[c400:.*]] = arith.constant 400 : index
// CHECK-DAG:     %[[c2000:.*]] = arith.constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = arith.constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = arith.constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK:                 scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK:                   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK:                     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK:                       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK:                         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK:                                 linalg.matmul
// CHECK:                                  ins({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>)
// CHECK:                                   outs({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>)

func.func @tile_permute_parallel_loop(%arg0: memref<?x?xf32>,
                                 %arg1: memref<?x?xf32>,
                                 %arg2: memref<?x?xf32>) {
  linalg.matmul {__internal_linalg_transform__ = "par__with_perm__"}
    ins(%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
   outs(%arg2: memref<?x?xf32>)
  return
}
// CHECK-LABEL: func @tile_permute_parallel_loop
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.*]] = memref.dim %[[ARG0]], %c0
//   CHECK-DAG:   %[[D1:.*]] = memref.dim %[[ARG0]], %c1
//   CHECK-DAG:   %[[D2:.*]] = memref.dim %[[ARG1]], %c1
//       CHECK:   scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[D2]]) step (%[[C8]])
//       CHECK:     scf.for %{{.*}} = %[[C0]] to %[[D1]] step %[[C4]]
//       CHECK:       scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[D0]]) step (%[[C16]])
