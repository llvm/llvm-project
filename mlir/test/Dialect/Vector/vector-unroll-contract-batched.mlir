// RUN: mlir-opt %s -test-vector-unrolling-patterns -split-input-file | FileCheck %s

func.func @vector_contract_batched(%lhs: vector<8x8x4xf32>, %rhs: vector<8x8x4xf32>, %init: vector<8x8x8xf32>) -> vector<8x8x8xf32> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(d0,d1,d2,c0) -> (d0,d1,c0)>,
                           affine_map<(d0,d1,d2,c0) -> (d0,d2,c0)>,
                           affine_map<(d0,d1,d2,c0) -> (d0,d1,d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8x4xf32>, vector<8x8x4xf32> into vector<8x8x8xf32>
  return %0 : vector<8x8x8xf32>
}


//    CHECK-LABEL: vector_contract_batched
// CHECK-COUNT-16: vector.contract
//      CHECK-NOT: vector.contract
//          CHECK: return
