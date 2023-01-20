// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-linalg-transform-patterns=test-patterns -split-input-file | FileCheck %s

// -----

func.func @dot(%x: memref<?xf32, strided<[1], offset: ?>>,
          %y: memref<?xf32, strided<[1], offset: ?>>,
          %v: memref<f32>) {
  linalg.dot ins(%x, %y: memref<?xf32, strided<[1], offset: ?>>,
                         memref<?xf32, strided<[1], offset: ?>>)
            outs(%v: memref<f32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.dot"]} in %arg1
    %1, %loop = transform.structured.tile %0 [8000] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func @dot
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c8000:.*]] = arith.constant 8000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK:           linalg.dot

// -----

func.func @matvec(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %x: memref<?xf32, strided<[1], offset: ?>>,
             %y: memref<?xf32, strided<[1], offset: ?>>) {
  linalg.matvec
    ins(%A, %x: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                memref<?xf32, strided<[1], offset: ?>>)
    outs(%y: memref<?xf32, strided<[1], offset: ?>>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1
    %1, %loops:2 = transform.structured.tile %0 [5, 6] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func @matvec
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[c6:.*]] = arith.constant 6 : index
// CHECK:         scf.for {{.*}} step %[[c5]]
// CHECK:           scf.for {{.*}} step %[[c6]]
// CHECK:             linalg.matvec
// CHECK:               ins({{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>)
// CHECK:              outs({{.*}}: memref<?xf32, strided<[1], offset: ?>>)

// -----

func.func @matmul(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %C: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                            memref<?x?xf32, strided<[?, 1], offset: ?>>)
               outs(%C: memref<?x?xf32, strided<[?, 1], offset: ?>>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %loops:3 = transform.structured.tile %0 [2000, 3000, 4000] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %2, %loops_2:3 = transform.structured.tile %1 [200, 300, 400] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %3, %loops_3:3 = transform.structured.tile %2 [20, 30, 40] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %4, %loops_4:3 = transform.structured.tile %3 [2, 3, 4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
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

// -----

// Map corresponding to a 2D memory access where the stride along the last dim is known to be 1.
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$nm:.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[$km:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>

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

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  transform.structured.interchange %0 iterator_interchange = [1, 2, 0]
}

// CHECK-LABEL:  func @permute_generic
// CHECK:        linalg.generic {
// CHECK-SAME:   indexing_maps = [#[[$kn]], #[[$nm]], #[[$km]]],
// CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"],
// CHECK-SAME:   library_call = "linalg_matmul"}
// CHECK:          memref<?x?xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>>

// -----

func.func @matvec_perm(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %x: memref<?xf32, strided<[1], offset: ?>>,
             %y: memref<?xf32, strided<[1], offset: ?>>) {
  linalg.matvec ins(%A, %x: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                            memref<?xf32, strided<[1], offset: ?>>)
               outs(%y: memref<?xf32, strided<[1], offset: ?>>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1
    %1, %loops:2 = transform.structured.tile %0 [5, 6] {interchange = [1, 0]} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
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

// -----

func.func @matmul_perm(%A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
             %C: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                            memref<?x?xf32, strided<[?, 1], offset: ?>>)
               outs(%C : memref<?x?xf32, strided<[?, 1], offset: ?>>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %loops:3 = transform.structured.tile %0 [2000, 3000, 4000] {interchange = [1, 2, 0]} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %2, %loops_2:3 = transform.structured.tile %1 [200, 300, 400] {interchange = [1, 0, 2]} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %3, %loops_3:3 = transform.structured.tile %2 [20, 30, 40] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
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
