// RUN: mlir-opt -convert-amdgpu-to-rocdl --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @test_swizzle_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @test_swizzle_i32(%arg0 : i32) -> i32 {
// CHECK:  %[[C:.*]] = llvm.mlir.constant(4161 : i32) : i32
// CHECK:  %[[RES:.*]] = rocdl.ds_swizzle %[[ARG0]], %[[C]] : (i32, i32) -> i32
// CHECK:  return %[[RES]] : i32
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_swizzle_f32
// CHECK-SAME: (%[[ARG0:.*]]: f32)
func.func @test_swizzle_f32(%arg0 : f32) -> f32 {
// CHECK:  %[[C:.*]] = llvm.mlir.constant(4161 : i32) : i32
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f32 to i32
// CHECK:  %[[RES:.*]] = rocdl.ds_swizzle %[[CAST]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[RES]] : i32 to f32
// CHECK:  return %[[RES_CAST]] : f32
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_swizzle_f16
// CHECK-SAME: (%[[ARG0:.*]]: f16)
func.func @test_swizzle_f16(%arg0 : f16) -> f16 {
// CHECK:  %[[C:.*]] = llvm.mlir.constant(4161 : i32) : i32
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f16 to i16
// CHECK:  %[[ZEXT:.*]] = llvm.zext %[[CAST]] : i16 to i32
// CHECK:  %[[RES:.*]] = rocdl.ds_swizzle %[[ZEXT]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[TRUNC:.*]] = llvm.trunc %[[RES]] : i32 to i16
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to f16
// CHECK:  return %[[RES_CAST]] : f16
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : f16
  return %0 : f16
}

// CHECK-LABEL: func @test_swizzle_2xi32
// CHECK-SAME: (%[[ARG0:.*]]: vector<2xi32>)
func.func @test_swizzle_2xi32(%arg0 : vector<2xi32>) -> vector<2xi32> {
// CHECK-DAG:  %[[V1:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C:.*]] = llvm.mlir.constant(4161 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:  %[[E0:.*]] = llvm.extractelement %[[ARG0]][%[[C0]] : i32] : vector<2xi32>
// CHECK:  %[[E1:.*]] = llvm.extractelement %[[ARG0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:  %[[S1:.*]] = rocdl.ds_swizzle %[[E0]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[S2:.*]] = rocdl.ds_swizzle %[[E1]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[V2:.*]] = llvm.insertelement %[[S1]], %[[V1]][%[[C0]] : i32] : vector<2xi32>
// CHECK:  %[[V3:.*]] = llvm.insertelement %[[S2]], %[[V2]][%[[C1]] : i32] : vector<2xi32>
// CHECK:  return %[[V3]] : vector<2xi32>
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL: func @test_swizzle_4xf16
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf16>)
func.func @test_swizzle_4xf16(%arg0 : vector<4xf16>) -> vector<4xf16> {
// CHECK-DAG:  %[[V1:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C:.*]] = llvm.mlir.constant(4161 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:  %[[CAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
// CHECK:  %[[E0:.*]] = llvm.extractelement %[[CAST1]][%[[C0]] : i32] : vector<2xi32>
// CHECK:  %[[E1:.*]] = llvm.extractelement %[[CAST1]][%[[C1]] : i32] : vector<2xi32>
// CHECK:  %[[S1:.*]] = rocdl.ds_swizzle %[[E0]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[S2:.*]] = rocdl.ds_swizzle %[[E1]], %[[C]] : (i32, i32) -> i32
// CHECK:  %[[V2:.*]] = llvm.insertelement %[[S1]], %[[V1]][%[[C0]] : i32] : vector<2xi32>
// CHECK:  %[[V3:.*]] = llvm.insertelement %[[S2]], %[[V2]][%[[C1]] : i32] : vector<2xi32>
// CHECK:  %[[CAST2:.*]] = llvm.bitcast %[[V3]] : vector<2xi32> to vector<4xf16>
// CHECK:  return %[[CAST2]] : vector<4xf16>
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : vector<4xf16>
  return %0 : vector<4xf16>
}
