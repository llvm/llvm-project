// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

!tensorA = tensor<32x32x16x2xbf16>
!tensorB = tensor<32x16x32x2xbf16>
!tensorC = tensor<32x32xbf16>

func.func @brgemm_bf16(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x16x32x2xbf16>, %arg2: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] 
		: tensor<8x32x32x32xbf16> into tensor<8x32x32x16x2xbf16>

  %0 = scf.forall (%arg3, %arg4) in (8, 32) shared_outs(%arg5 = %arg2) -> (tensor<8x32x32x32xbf16>) {
    %extracted_slice = tensor.extract_slice %expanded[%arg3, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] 
		: tensor<8x32x32x16x2xbf16> to !tensorA
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] 
		: tensor<32x32x16x32x2xbf16> to !tensorB
    %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
		: tensor<8x32x32x32xbf16> to !tensorC

    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = 
		["reduction", "reduction", "parallel", "parallel", "reduction"]} 
		ins(%extracted_slice, %extracted_slice_0 : !tensorA, !tensorB) outs(%extracted_slice_1 : !tensorC) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_2 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> !tensorC

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
	: !tensorC into tensor<8x32x32x32xbf16>
    }
  }
  return %0 : tensor<8x32x32x32xbf16>
}

// CHECK-LABEL: @brgemm_bf16
// CHECK: tensor.empty() : tensor<32x32xf32>
// CHECK: linalg.fill ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32> 
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<32x32x16x2xbf16>, tensor<32x16x32x2xbf16>) outs({{.*}} : tensor<32x32xf32>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<32x32x16x2xbf16>, tensor<32x16x32x2xbf16>) outs({{.*}} : tensor<32x32xbf16>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<32x32xf32>, tensor<32x32xbf16>) outs({{.*}} : tensor<32x32xbf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

func.func @batch_matmul_bf16(%arg0: tensor<16x24x32x2xbf16>, %arg1: tensor<16x32x128x2xbf16>, %arg2: tensor<16x24x128xbf16>) -> tensor<16x24x128xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs(%arg2 : tensor<16x24x128xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<16x24x128xbf16>
  return %0 : tensor<16x24x128xbf16>
}

// CHECK-LABEL: @batch_matmul_bf16
// CHECK: tensor.empty() : tensor<16x24x128xf32>
// CHECK: linalg.fill ins(%cst : f32) outs(%0 : tensor<16x24x128xf32>) -> tensor<16x24x128xf32>
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs({{.*}} : tensor<16x24x128xf32>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs({{.*}} : tensor<16x24x128xbf16>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x128xf32>, tensor<16x24x128xbf16>) outs({{.*}} : tensor<16x24x128xbf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d5, d0, d1, d2, d3, d4) -> (d5, d0, d2, d4, d1)>
#map1 = affine_map<(d5, d0, d1, d2, d3, d4) -> (d5, d0, d4, d3, d1)>
#map2 = affine_map<(d5, d0, d1, d2, d3, d4) -> (d5, d0, d2, d3)>

func.func @matmul_many_dim_bf16(%arg0: tensor<2x16x24x32x2xbf16>, %arg1: tensor<2x16x32x128x2xbf16>, %arg2: tensor<2x16x24x128xbf16>) -> tensor<2x16x24x128xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x16x24x32x2xbf16>, tensor<2x16x32x128x2xbf16>) outs(%arg2 : tensor<2x16x24x128xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<2x16x24x128xbf16>
  return %0 : tensor<2x16x24x128xbf16>
}

// CHECK-LABEL: @matmul_many_dim_bf16
// CHECK: tensor.empty() : tensor<2x16x24x128xf32>
// CHECK: linalg.fill ins(%cst : f32) outs(%0 : tensor<2x16x24x128xf32>) -> tensor<2x16x24x128xf32>
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<2x16x24x32x2xbf16>, tensor<2x16x32x128x2xbf16>) outs({{.*}} : tensor<2x16x24x128xf32>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<2x16x24x32x2xbf16>, tensor<2x16x32x128x2xbf16>) outs({{.*}} : tensor<2x16x24x128xbf16>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<2x16x24x128xf32>, tensor<2x16x24x128xbf16>) outs({{.*}} : tensor<2x16x24x128xbf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0,  d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0,  d2, d3, d4) -> (d0, d4, d3)>
#map2 = affine_map<(d0,  d2, d3, d4) -> (d2, d3)>

func.func @brgemm_flat_int8(%arg0: tensor<16x64x256xi8>, %arg1: tensor<16x256x128xi8>, %arg2: tensor<64x128xi8>) -> tensor<64x128xi8> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs(%arg2 : tensor<64x128xi8>) {
  ^bb0(%in: i8, %in_0: i8, %out: i8):
    %1 = arith.muli %in, %in_0 : i8
    %2 = arith.addi %out, %1 : i8
    linalg.yield %2 : i8
  } -> tensor<64x128xi8>
  return %0 : tensor<64x128xi8>
}

// CHECK-LABEL: @brgemm_flat_int8
// CHECK: tensor.empty() : tensor<64x128xi32>
// CHECK: linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<64x128xi32>) -> tensor<64x128xi32>
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs({{.*}} : tensor<64x128xi32>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs({{.*}} : tensor<64x128xi8>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<64x128xi32>, tensor<64x128xi8>) outs({{.*}} : tensor<64x128xi8>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

func.func @negative_sub_op_in_generic(%arg0: tensor<16x24x32x2xbf16>, %arg1: tensor<16x32x128x2xbf16>, %arg2: tensor<16x24x128xbf16>) -> tensor<16x24x128xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs(%arg2 : tensor<16x24x128xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.subf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<16x24x128xbf16>
  return %0 : tensor<16x24x128xbf16>
}

// CHECK-LABEL: @negative_sub_op_in_generic
// CHECK-NOT: tensor.empty() : tensor<32x32xf32>
// CHECK-NOT: linalg.fill ins(%cst : f32) outs(%0 : tensor<16x24x128xf32>) -> tensor<16x24x128xf32>
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs({{.*}} : tensor<16x24x128xf32>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs({{.*}} : tensor<16x24x128xbf16>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x128xf32>, tensor<16x24x128xbf16>) outs({{.*}} : tensor<16x24x128xbf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>

func.func @negative_f16_type(%arg0: tensor<16x24x32x2xf16>, %arg1: tensor<16x32x128x2xf16>, %arg2: tensor<16x24x128xf16>) -> tensor<16x24x128xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x24x32x2xf16>, tensor<16x32x128x2xf16>) outs(%arg2 : tensor<16x24x128xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %1 = arith.mulf %in, %in_0 : f16
    %2 = arith.addf %out, %1 : f16
    linalg.yield %2 : f16
  } -> tensor<16x24x128xf16>
  return %0 : tensor<16x24x128xf16>
}

// CHECK-LABEL: @negative_f16_type
// CHECK-NOT: tensor.empty() : tensor<32x32xf32>
// CHECK-NOT: linalg.fill ins(%cst : f32) outs(%0 : tensor<16x24x128xf32>) -> tensor<16x24x128xf32>
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xf16>, tensor<16x32x128x2xf16>) outs({{.*}} : tensor<16x24x128xf32>) {
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xf16>, tensor<16x32x128x2xf16>) outs({{.*}} : tensor<16x24x128xf16>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x128xf32>, tensor<16x24x128xf16>) outs({{.*}} : tensor<16x24x128xf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d2, d3, d4) -> (d0, d4, d3)>
#map2 = affine_map<(d0, d2, d3, d4) -> (d2, d3)>

func.func @negative_i32_acc(%arg0: tensor<16x64x256xi8>, %arg1: tensor<16x256x128xi8>, %arg2: tensor<64x128xi32>) -> tensor<64x128xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs(%arg2 : tensor<64x128xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %a = arith.extsi %in : i8 to i32
    %b = arith.extsi %in_0 : i8 to i32
    %1 = arith.muli %a, %b : i32
    %2 = arith.addi %out, %1 : i32
    linalg.yield %2 : i32
  } -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: @negative_i32_acc
// CHECK-NOT: tensor.empty() : tensor<64x128xi32>
// CHECK-NOT: linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<64x128xi32>) -> tensor<64x128xi32>
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs({{.*}} : tensor<64x128xi32>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x64x256xi8>, tensor<16x256x128xi8>) outs({{.*}} : tensor<64x128xi8>) {
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<64x128xi32>, tensor<64x128xi8>) outs({{.*}} : tensor<64x128xi8>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

func.func @negative_memref(%arg0: memref<16x24x32x2xbf16>, %arg1: memref<16x32x128x2xbf16>, %arg2: memref<24x128xbf16>) -> memref<24x128xbf16> {
  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<16x24x32x2xbf16>, memref<16x32x128x2xbf16>) outs(%arg2 : memref<24x128xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %0 = arith.mulf %in, %in_0 : bf16
    %1 = arith.addf %out, %0 : bf16
    linalg.yield %1 : bf16
  }
  %alloc = memref.alloc() : memref<24x128xbf16>
  memref.copy %arg2, %alloc : memref<24x128xbf16> to memref<24x128xbf16>
  return %alloc : memref<24x128xbf16>
}

// CHECK-LABEL: @negative_memref
// CHECK-NOT: tensor.empty() : tensor<24x128xf32>
// CHECK-NOT: linalg.fill ins(%cst : f32) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<16x24x32x2xbf16>, tensor<16x32x128x2xbf16>) outs({{.*}} : tensor<24x128xf32>) {
// CHECK: linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<16x24x32x2xbf16>, memref<16x32x128x2xbf16>) outs(%arg2 : memref<24x128xbf16>) { 
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<24x128xf32>, tensor<24x128xbf16>) outs({{.*}} : tensor<24x128xbf16>) {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.convert_linalg_generic_to_32_bit_accumulation
    } : !transform.any_op
    transform.yield
  }
}
