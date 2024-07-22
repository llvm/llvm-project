// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification="parallelization-strategy=any-storage-any-loop" | \
// RUN:   FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}
// CHECK-LABEL:  func.func @matvec(
//  CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<16x32xf32, #sparse{{[0-9]*}}>,
//  CHECK-SAME:    %[[TMP_arg1:.*]]: tensor<32xf32>,
//  CHECK-SAME:    %[[TMP_arg2:.*]]: tensor<16xf32>) -> tensor<16xf32> {
//   CHECK-DAG:  %[[TMP_c16:.*]] = arith.constant 16 : index
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_0:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index}
//   CHECK-DAG:  %[[TMP_1:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index}
//   CHECK-DAG:  %[[TMP_2:.*]] = sparse_tensor.values %[[TMP_arg0]]
//   CHECK-DAG:  %[[TMP_3:.*]] = bufferization.to_memref %[[TMP_arg1]] : memref<32xf32>
//   CHECK-DAG:  %[[TMP_4:.*]] = bufferization.to_memref %[[TMP_arg2]] : memref<16xf32>
//       CHECK:  scf.parallel (%[[TMP_arg3:.*]]) = (%[[TMP_c0]]) to (%[[TMP_c16]]) step (%[[TMP_c1]]) {
//       CHECK:    %[[TMP_6:.*]] = memref.load %[[TMP_4]][%[[TMP_arg3]]] : memref<16xf32>
//       CHECK:    %[[TMP_7:.*]] = memref.load %[[TMP_0]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_8:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_9:.*]] = memref.load %[[TMP_0]][%[[TMP_8]]] : memref<?xindex>
//       CHECK:    %[[TMP_10:.*]] = scf.parallel (%[[TMP_arg4:.*]]) = (%[[TMP_7]]) to (%[[TMP_9]]) step (%[[TMP_c1]]) init (%[[TMP_6]]) -> f32 {
//       CHECK:      %[[TMP_11:.*]] = memref.load %[[TMP_1]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_12:.*]] = memref.load %[[TMP_2]][%[[TMP_arg4]]] : memref<?xf32>
//       CHECK:      %[[TMP_13:.*]] = memref.load %[[TMP_3]][%[[TMP_11]]] : memref<32xf32>
//       CHECK:      %[[TMP_14:.*]] = arith.mulf %[[TMP_12]], %[[TMP_13]] : f32
//       CHECK:      scf.reduce(%[[TMP_14]]  : f32) {
//       CHECK:      ^bb0(%[[TMP_arg5:.*]]: f32, %[[TMP_arg6:.*]]: f32):
//       CHECK:        %[[TMP_15:.*]] = arith.addf %[[TMP_arg5]], %[[TMP_arg6]] : f32
//       CHECK:        scf.reduce.return %[[TMP_15]] : f32
//       CHECK:      }
//       CHECK:    }
//       CHECK:    memref.store %[[TMP_10]], %[[TMP_4]][%[[TMP_arg3]]] : memref<16xf32>
//       CHECK:    scf.reduce
//       CHECK:  }
//       CHECK:  %[[TMP_5:.*]] = bufferization.to_tensor %[[TMP_4]] : memref<16xf32>
//       CHECK:  return %[[TMP_5]] : tensor<16xf32>
func.func @matvec(%arga: tensor<16x32xf32, #CSR>,
                  %argb: tensor<32xf32>,
	          %argx: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<16x32xf32, #CSR>, tensor<32xf32>)
     outs(%argx: tensor<16xf32>) {
    ^bb(%A: f32, %b: f32, %x: f32):
      %0 = arith.mulf %A, %b : f32
      %1 = arith.addf %0, %x : f32
      linalg.yield %1 : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
