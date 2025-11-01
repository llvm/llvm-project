// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @fp32_batch_reduce_matmul_vector_size_16(%arg0: memref<1x4x32xf32>, %arg1: memref<1x32x96xf32>, %arg2: memref<4x96xf32>) -> memref<4x96xf32> {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c96 = arith.constant 96 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c4 step %c4 {
      scf.for %arg4 = %c0 to %c96 step %c96 {
        %subview = memref.subview %arg2[%arg3, %arg4] [4, 96] [1, 1] : memref<4x96xf32> to memref<4x96xf32, strided<[96, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x96xf32, strided<[96, 1], offset: ?>>, vector<4x96xf32>
        %2 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %1) -> (vector<4x96xf32>) {
          %3 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (vector<4x96xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 4, 1] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x1xf32, strided<[128, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 1, 96] [1, 1, 1] : memref<1x32x96xf32> to memref<1x1x96xf32, strided<[3072, 96, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[128, 32, 1], offset: ?>>, vector<1x4x1xf32>
            %5 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x1x96xf32, strided<[3072, 96, 1], offset: ?>>, vector<1x1x96xf32>
            %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x1xf32>, vector<1x1x96xf32> into vector<4x96xf32>
            scf.yield %6 : vector<4x96xf32>
          }
          scf.yield %3 : vector<4x96xf32>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<4x96xf32>, memref<4x96xf32, strided<[96, 1], offset: ?>>
      }
    }
    return %arg2 : memref<4x96xf32>
  }
}

// CHECK-LABEL: func.func @fp32_batch_reduce_matmul_vector_size_16(
// CHECK-COUNT-24: vector.fma{{.*}}vector<16xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_nanokernel_lowering vector_size = 16
    } : !transform.any_op
    transform.yield
  }
}

// -----

module {
  func.func @fp32_batch_matmul_vector_size_8(%arg0: memref<4x32xf32>, %arg1: memref<32x96xf32>, %arg2: memref<4x96xf32>) -> memref<4x96xf32> {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c96 = arith.constant 96 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c4 step %c4 {
      scf.for %arg4 = %c0 to %c96 step %c96 {
        %subview = memref.subview %arg2[%arg3, %arg4] [4, 96] [1, 1] : memref<4x96xf32> to memref<4x96xf32, strided<[96, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x96xf32, strided<[96, 1], offset: ?>>, vector<4x96xf32>

          %3 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %1) -> (vector<4x96xf32>) {
            %subview_0 = memref.subview %arg0[%arg3, %arg7] [4, 1] [1, 1] : memref<4x32xf32> to memref<4x1xf32, strided<[32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg7, %arg4] [1, 96] [1, 1] : memref<32x96xf32> to memref<1x96xf32, strided<[96, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x1xf32, strided<[32, 1], offset: ?>>, vector<4x1xf32>
            %5 = vector.transfer_read %subview_1[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x96xf32, strided<[96, 1], offset: ?>>, vector<1x96xf32>
            %6 = vector.contract {indexing_maps = [affine_map<(d1, d2, d3) -> (d1, d3)>, affine_map<(d1, d2, d3) -> (d3, d2)>, affine_map<(d1, d2, d3) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<4x1xf32>, vector<1x96xf32> into vector<4x96xf32>
            scf.yield %6 : vector<4x96xf32>
          }

        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<4x96xf32>, memref<4x96xf32, strided<[96, 1], offset: ?>>
      }
    }
    return %arg2 : memref<4x96xf32>
  }
}

// CHECK-LABEL: func.func @fp32_batch_matmul_vector_size_8(
// CHECK-COUNT-48: vector.fma{{.*}}vector<8xf32>
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_nanokernel_lowering vector_size = 8
    } : !transform.any_op
    transform.yield
  }
}

// -----

module {
  func.func @negative_not_tiled(%arg0: memref<1x4x32xf32>, %arg1: memref<1x32x96xf32>, %arg2: memref<4x96xf32>) -> memref<4x96xf32> {
    %c0 = arith.constant 0 : index
    %0 = ub.poison : f32
    %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x4x32xf32>, vector<1x4x32xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x96xf32>, vector<1x32x96xf32>
    %3 = vector.transfer_read %arg2[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x96xf32>, vector<4x96xf32>
    %4 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x32xf32>, vector<1x32x96xf32> into vector<4x96xf32>
    vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x96xf32>, memref<4x96xf32>
    return %arg2 : memref<4x96xf32>
  }
}

// CHECK-LABEL: func.func @negative_not_tiled(
// CHECK-NOT: vector.fma{{.*}}vector<8xf32>
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_nanokernel_lowering vector_size = 8
    } : !transform.any_op
    transform.yield
  }
}

// -----

module {
  func.func @negative_tensor_type(%arg0: tensor<32x32x32xf32>, %arg1: tensor<32x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %1 = scf.for %arg3 = %c0 to %c32 step %c4 iter_args(%arg4 = %arg2) -> (tensor<32x32xf32>) {
      %2 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %arg4) -> (tensor<32x32xf32>) {
        %3 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (tensor<32x32xf32>) {
          %4 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %arg8) -> (tensor<32x32xf32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg7, %arg3, %arg9] [1, 4, 1] [1, 1, 1] : tensor<32x32x32xf32> to tensor<1x4x1xf32>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg7, %arg9, %arg5] [1, 1, 16] [1, 1, 1] : tensor<32x32x32xf32> to tensor<1x1x16xf32>
            %extracted_slice_1 = tensor.extract_slice %arg10[%arg3, %arg5] [4, 16] [1, 1] : tensor<32x32xf32> to tensor<4x16xf32>
            %5 = vector.transfer_read %extracted_slice[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : tensor<1x4x1xf32>, vector<1x4x1xf32>
            %6 = vector.transfer_read %extracted_slice_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : tensor<1x1x16xf32>, vector<1x1x16xf32>
            %7 = vector.transfer_read %extracted_slice_1[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<4x16xf32>, vector<4x16xf32>
            %8 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %7 : vector<1x4x1xf32>, vector<1x1x16xf32> into vector<4x16xf32>
            %9 = vector.transfer_write %8, %extracted_slice_1[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, tensor<4x16xf32>
            %inserted_slice = tensor.insert_slice %9 into %arg10[%arg3, %arg5] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<32x32xf32>
            scf.yield %inserted_slice : tensor<32x32xf32>
          }
          scf.yield %4 : tensor<32x32xf32>
        }
        scf.yield %3 : tensor<32x32xf32>
      }
      scf.yield %2 : tensor<32x32xf32>
    }
    return %1 : tensor<32x32xf32>
  }
}

// CHECK-LABEL: func.func @negative_tensor_type(
// CHECK-NOT: vector.fma{{.*}}vector<16xf32>
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_nanokernel_lowering vector_size = 16
    } : !transform.any_op
    transform.yield
  }
}

// -----

module {
  func.func @negative_accumulator_not_hoisted_outside_K_or_reduction_loop(%arg0: memref<1x4x32xf32>, %arg1: memref<1x32x96xf32>, %arg2: memref<4x96xf32>) -> memref<4x96xf32> {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c96 = arith.constant 96 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c4 step %c4 {
      scf.for %arg4 = %c0 to %c96 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [4, 32] [1, 1] : memref<4x96xf32> to memref<4x32xf32, strided<[96, 1], offset: ?>>
        scf.for %arg5 = %c0 to %c1 step %c1 {
          scf.for %arg6 = %c0 to %c32 step %c1 {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x1xf32, strided<[128, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4] [1, 1, 32] [1, 1, 1] : memref<1x32x96xf32> to memref<1x1x32xf32, strided<[3072, 96, 1], offset: ?>>
            %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[128, 32, 1], offset: ?>>, vector<1x4x1xf32>
            %2 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[3072, 96, 1], offset: ?>>, vector<1x1x32xf32>
            %3 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x32xf32, strided<[96, 1], offset: ?>>, vector<4x32xf32>
            %4 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x1xf32>, vector<1x1x32xf32> into vector<4x32xf32>
            vector.transfer_write %4, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<4x32xf32>, memref<4x32xf32, strided<[96, 1], offset: ?>>
          }
        }
      }
    }
    return %arg2 : memref<4x96xf32>
  }
}

// CHECK-LABEL: func.func @negative_accumulator_not_hoisted_outside_K_or_reduction_loop(
// CHECK-NOT: vector.fma{{.*}}vector<16xf32>
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86vector.vector_contract_nanokernel_lowering vector_size = 16
    } : !transform.any_op
    transform.yield
  }
}
