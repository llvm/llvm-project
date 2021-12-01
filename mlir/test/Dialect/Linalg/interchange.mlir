// RUN: mlir-opt %s -test-linalg-codegen-strategy="iterator-interchange=4,0,3,1,2" | FileCheck %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="iterator-interchange=4,0,3,1,2" -test-linalg-codegen-strategy="iterator-interchange=1,3,4,2,0" | FileCheck --check-prefix=CANCEL-OUT %s

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>

func @interchange_generic_op(%arg0 : memref<1x2x3x4x5xindex>, %arg1 : memref<1x2x4xindex>) {
  linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]}
  ins(%arg0 : memref<1x2x3x4x5xindex>)
  outs(%arg1 : memref<1x2x4xindex>) {
      ^bb0(%arg2 : index, %arg3 : index) :
        %0 = linalg.index 0 : index
        %1 = linalg.index 1 : index
        %2 = linalg.index 4 : index
        %3 = arith.subi %0, %1 : index
        %4 = arith.addi %3, %2 : index
        %5 = arith.addi %4, %arg2 : index
        linalg.yield %5 : index
      }
  return
}

//    CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4, d2, d0)>
//    CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d2)>
//        CHECK: func @interchange_generic_op
//        CHECK:   linalg.generic
//   CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//   CHECK-SAME:     iterator_types = ["reduction", "parallel", "parallel", "parallel", "reduction"]
//    CHECK-DAG:     %[[IDX0:.+]] = linalg.index 1 : index
//    CHECK-DAG:     %[[IDX1:.+]] = linalg.index 3 : index
//    CHECK-DAG:     %[[IDX4:.+]] = linalg.index 0 : index
//        CHECK:     %[[T0:.+]] = arith.subi %[[IDX0]], %[[IDX1]] : index
//        CHECK:     %[[T1:.+]] = arith.addi %[[T0]], %[[IDX4]] : index
//        CHECK:     %[[T2:.+]] = arith.addi %[[T1]], %{{.*}} : index

//  CANCEL-OUT-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//  CANCEL-OUT-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
//      CANCEL-OUT: func @interchange_generic_op
//      CANCEL-OUT:   linalg.generic
// CANCEL-OUT-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CANCEL-OUT-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]
//  CANCEL-OUT-DAG:     %[[IDX0:.+]] = linalg.index 0 : index
//  CANCEL-OUT-DAG:     %[[IDX1:.+]] = linalg.index 1 : index
//  CANCEL-OUT-DAG:     %[[IDX4:.+]] = linalg.index 4 : index
//      CANCEL-OUT:     %[[T0:.+]] = arith.subi %[[IDX0]], %[[IDX1]] : index
//      CANCEL-OUT:     %[[T1:.+]] = arith.addi %[[T0]], %[[IDX4]] : index
//      CANCEL-OUT:     %[[T2:.+]] = arith.addi %[[T1]], %{{.*}} : index


