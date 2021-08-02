// RUN: mlir-opt %s -test-vector-contraction-conversion=vector-outerproduct=1 | FileCheck %s

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}
#matvecmax_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"],
  kind = #vector.kind<max>
}

#mattransvec_accesses = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#mattransvec_trait = {
  indexing_maps = #mattransvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#vecmat_accesses = [
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i)>
]
#vecmat_trait = {
  indexing_maps = #vecmat_accesses,
  iterator_types = ["parallel", "reduction"]
}

#vecmattrans_accesses = [
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (i)>
]
#vecmattrans_trait = {
  indexing_maps = #vecmattrans_accesses,
  iterator_types = ["parallel", "reduction"]
}

#redpar_vecmattrans_accesses = [
  affine_map<(i, j) -> (i)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>
]
#redpar_vecmattrans_trait = {
  indexing_maps = #redpar_vecmattrans_accesses,
  iterator_types = ["reduction", "parallel"]
}

// CHECK-LABEL: func @matvec2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.transpose %[[T0]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[T2]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: memref.store %[[T9]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @matvec2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #matvec_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}

// CHECK-LABEL: func @matvecmax2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.transpose %[[T0]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[T2]] {kind = #vector.kind<max>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<max>} : vector<2xf32>, f32
// CHECK: memref.store %[[T9]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @matvecmax2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                   %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #matvecmax_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}

// CHECK-LABEL: func @mattransvec2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.extract %[[T0]][0] : vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[T2]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[T0]][1] : vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: memref.store %[[T8]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @mattransvec2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                     %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #mattransvec_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}

// CHECK-LABEL: func @vecmat2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.transpose %[[T0]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[T2]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: memref.store %[[T9]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @vecmat2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #vecmat_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}

// CHECK-LABEL: func @vecmattrans2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.extract %[[T0]][0] : vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[T2]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[T0]][1] : vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: memref.store %[[T8]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @vecmattrans2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                     %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #vecmattrans_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}

// CHECK-LABEL: func @redpar_vecmattrans2x2
// CHECK-SAME: %[[A:.*0]]: memref<vector<2x2xf32>>
// CHECK-SAME: %[[B:.*1]]: memref<vector<2xf32>>
// CHECK-SAME: %[[C:.*2]]: memref<vector<2xf32>>
// CHECK: %[[T0:.*]] = memref.load %[[A]][] : memref<vector<2x2xf32>>
// CHECK: %[[T1:.*]] = memref.load %[[B]][] : memref<vector<2xf32>>
// CHECK: %[[T2:.*]] = memref.load %[[C]][] : memref<vector<2xf32>>
// CHECK: %[[T3:.*]] = vector.extract %[[T0]][0] : vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[T2]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[T0]][1] : vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: memref.store %[[T8]], %[[C]][] : memref<vector<2xf32>>
// CHECK: return
func @redpar_vecmattrans2x2(%arg0: memref<vector<2x2xf32>>, %arg1: memref<vector<2xf32>>,
                                                     %arg2: memref<vector<2xf32>>) {
  %A = memref.load %arg0[] : memref<vector<2x2xf32>>
  %x = memref.load %arg1[] : memref<vector<2xf32>>
  %b = memref.load %arg2[] : memref<vector<2xf32>>
  %0 = vector.contract #redpar_vecmattrans_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  memref.store %0, %arg2[] : memref<vector<2xf32>>
  return
}
