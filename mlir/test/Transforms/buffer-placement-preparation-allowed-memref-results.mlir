// RUN: mlir-opt -test-buffer-placement-preparation-with-allowed-memref-results -split-input-file %s | FileCheck %s

// Since allowMemrefEscaping is on for Buffer Placement in this test pass, all
// tensor typed function results are converted to memref and remain as function
// results. All memref typed function results will escape from the deallocation
// phase of Buffer Placement.

// CHECK-LABEL: func @void_function_signature_conversion
func @void_function_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>)

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @complex_signature_conversion
func @complex_signature_conversion(%arg0: tensor<5xf32>, %arg1: memref<10xf32>, %arg2: i1, %arg3: f16) -> (i1, tensor<5xf32>, memref<10xf32>, memref<15xf32>, f16) {
  %0 = alloc() : memref<15xf32>
  %1 = linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]
        } %arg0 {
        ^bb0(%gen1_arg0: f32):
          %tmp1 = exp %gen1_arg0 : f32
          linalg.yield %tmp1 : f32
        }: tensor<5xf32> -> tensor<5xf32>
  return %arg2, %1, %arg1, %0, %arg3 : i1, tensor<5xf32>, memref<10xf32>, memref<15xf32>, f16
}
//      CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: f16)
// CHECK-SAME: (i1, memref<5xf32>, memref<10xf32>, memref<15xf32>, f16)
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc()
//      CHECK: %[[LINALG_ALLOC:.*]] = alloc()
//      CHECK: return %[[ARG2]], %[[LINALG_ALLOC]], %[[ARG1]], %[[FIRST_ALLOC]], %[[ARG3]]

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: memref<4x8xf32>) {
  return
}
// CHECK: ({{.*}}: memref<4x8xf32>)

// -----

// CHECK-LABEL: func @no_signature_conversion_is_needed
func @no_signature_conversion_is_needed(%arg0: i1, %arg1: f16) -> (i1, f16){
  return %arg0, %arg1 : i1, f16
}
// CHECK: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: f16) -> (i1, f16)
// CHECK: return %[[ARG0]], %[[ARG1]]

// -----

// CHECK-LABEL: func @simple_signature_conversion
func @simple_signature_conversion(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>) -> [[TYPE]]<[[RANK]]>
// CHECK-NEXT: return %[[ARG0]]

// -----

// CHECK-LABEL: func @func_and_block_signature_conversion
func @func_and_block_signature_conversion(%arg0 : tensor<2xf32>, %cond : i1, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    br ^exit(%arg0 : tensor<2xf32>)
  ^exit(%arg2: tensor<2xf32>):
    return %arg1 : tensor<4x4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]]) -> [[RESULT_TYPE:.*]]
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT:  return %[[ARG1]]

// -----

// CHECK-LABEL: func @callee
func @callee(%arg1: tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>) {
  %buff = alloc() : memref<2xf32>
  return %arg1, %buff : tensor<5xf32>, memref<2xf32>
}
// CHECK: (%[[CALLEE_ARG:.*]]: memref<5xf32>) -> (memref<5xf32>, memref<2xf32>)
// CHECK: %[[ALLOC:.*]] = alloc()
// CHECK: return %[[CALLEE_ARG]], %[[ALLOC]]

// CHECK-LABEL: func @caller
func @caller(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %x:2 = call @callee(%arg0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  %y:2 = call @callee(%x#0) : (tensor<5xf32>) -> (tensor<5xf32>, memref<2xf32>)
  return %y#0 : tensor<5xf32>
}
// CHECK: (%[[CALLER_ARG:.*]]: memref<5xf32>) -> memref<5xf32>
// CHECK: %[[X:.*]]:2 = call @callee(%[[CALLER_ARG]])
// CHECK: %[[Y:.*]]:2 = call @callee(%[[X]]#0)
// CHECK: return %[[Y]]#0





