// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx950" | FileCheck %s
// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx1200" | FileCheck %s
  
// CHECK-LABEL: func.func @scalar_ext
// CHECK-SAME: ([[V:%.+]]: f8E5M2)
// CHECK: [[FLOAT:%.+]] = amdgpu.ext_packed_fp8 [[V]][0] : f8E5M2 to f32
// CHECK: [[W:%.+]] = arith.truncf [[FLOAT]] : f32 to f16
// CHECK: return [[W]]
func.func @scalar_ext(%v: f8E5M2) -> f16 {
  %w = arith.extf %v : f8E5M2 to f16
  return %w : f16
}

// No 0-D test because arith.extf hasn't been extended to support it.

// -----

// CHECK-LABEL: func.func @vector_ext_short
// CHECK-SAME: ([[V:%.+]]: vector<2xf8E5M2>)
// CHECK: [[FLOAT0:%.+]] = amdgpu.ext_packed_fp8 [[V]][0] : vector<2xf8E5M2> to vector<2xf32>
// CHECK: [[EXT:%.+]] = arith.extf [[FLOAT0]] : vector<2xf32> to vector<2xf64>
// CHECK: return [[EXT]] : vector<2xf64>

func.func @vector_ext_short(%v: vector<2xf8E5M2>) -> vector<2xf64> {
  %w = arith.extf %v : vector<2xf8E5M2> to vector<2xf64>
  return %w : vector<2xf64>
}

// -----

// CHECK-LABEL: func.func @vector_ext_long
// CHECK-SAME: ([[V:%.+]]: vector<9xf8E4M3FN>)
// CHECK: [[W0:%.+]] = arith.constant dense<0.000000e+00> : vector<9xf32>
// CHECK: [[IN1:%.+]] = vector.extract_strided_slice [[V]] {offsets = [0], sizes = [4], strides = [1]} : vector<9xf8E4M3FN> to vector<4xf8E4M3FN>
// CHECK: [[FLOAT1:%.+]] = amdgpu.ext_packed_fp8 [[IN1]][0] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W1:%.+]] = vector.insert_strided_slice [[FLOAT1]], [[W0]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<9xf32>
// CHECK: [[FLOAT2:%.+]] = amdgpu.ext_packed_fp8 [[IN1]][1] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W2:%.+]] = vector.insert_strided_slice [[FLOAT2]], [[W1]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<9xf32>
// CHECK: [[IN2:%.+]] = vector.extract_strided_slice [[V]] {offsets = [4], sizes = [4], strides = [1]} : vector<9xf8E4M3FN> to vector<4xf8E4M3FN>
// CHECK: [[FLOAT3:%.+]] = amdgpu.ext_packed_fp8 [[IN2]][0] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W3:%.+]] = vector.insert_strided_slice [[FLOAT3]], [[W2]] {offsets = [4], strides = [1]} : vector<2xf32> into vector<9xf32>
// CHECK: [[FLOAT4:%.+]] = amdgpu.ext_packed_fp8 [[IN2]][1] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W4:%.+]] = vector.insert_strided_slice [[FLOAT4]], [[W3]] {offsets = [6], strides = [1]} : vector<2xf32> into vector<9xf32>
// CHECK: [[IN3:%.+]] = vector.extract_strided_slice [[V]] {offsets = [8], sizes = [1], strides = [1]} : vector<9xf8E4M3FN> to vector<1xf8E4M3FN>
// CHECK: [[FLOAT5:%.+]] = amdgpu.ext_packed_fp8 [[IN3]][0] : vector<1xf8E4M3FN> to f32
// CHECK: [[W5:%.+]] = vector.insert [[FLOAT5]], [[W4]] [8] : f32 into vector<9xf32>
// CHECK: return [[W5]]
func.func @vector_ext_long(%v: vector<9xf8E4M3FN>) -> vector<9xf32> {
  %w = arith.extf %v : vector<9xf8E4M3FN> to vector<9xf32>
  return %w : vector<9xf32>
}

// -----

// CHECK-LABEL: func.func @scalar_trunc
// CHECK-SAME: ([[V:%.+]]: f16)
// CHECK: [[FLOAT:%.+]] = arith.extf [[V]] : f16 to f32
// CHECK: [[TRUNCV:%.+]] = amdgpu.packed_trunc_2xfp8 [[FLOAT]], undef into undef[word 0] : f32 to vector<4xf8E5M2>
// CHECK: [[W:%.+]] = vector.extract [[TRUNCV]][0] : f8E5M2 from vector<4xf8E5M2>
// CHECK: return [[W]] : f8E5M2
func.func @scalar_trunc(%v: f16) -> f8E5M2 {
  %w = arith.truncf %v : f16 to f8E5M2
  return %w : f8E5M2
}

// No 0-D test because arith.truncf hasn't been extended to support it.

// -----

// CHECK-LABEL: func.func @vector_trunc_short
// CHECK-SAME: ([[V:%.+]]: vector<2xf64>) -> vector<2xf8E5M2> {
// CHECK: [[V0:%.+]] = vector.extract [[V]][0]
// CHECK: [[F0:%.+]] = arith.truncf [[V0]] : f64 to f32
// CHECK: [[V1:%.+]] = vector.extract [[V]][1]
// CHECK: [[F1:%.+]] = arith.truncf [[V1]] : f64 to f32
// CHECK: [[W0:%.+]] = amdgpu.packed_trunc_2xfp8 [[F0]], [[F1]] into undef[word 0] : f32 to vector<4xf8E5M2>
// CHECK: [[W:%.+]] = vector.extract_strided_slice [[W0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2> to vector<2xf8E5M2>
// CHECK: return [[W]] : vector<2xf8E5M2>
func.func @vector_trunc_short(%v: vector<2xf64>) -> vector<2xf8E5M2> {
  %w = arith.truncf %v : vector<2xf64> to vector<2xf8E5M2>
  return %w : vector<2xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @vector_trunc_long
// CHECK-SAME: ([[V:%.+]]: vector<9xf32>)
// CHECK: [[ZEROES:%.+]] = arith.constant dense<0.000000e+00> : vector<9xf8E4M3FN>
// CHECK: [[T0:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T1:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T0]][word 1]
// CHECK: [[W0:%.+]] = vector.insert_strided_slice [[T1]], [[ZEROES]] {offsets = [0], strides = [1]}

// CHECK: [[T2:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T3:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T2]][word 1]
// CHECK: [[W1:%.+]] = vector.insert_strided_slice [[T3]], [[W0]] {offsets = [4], strides = [1]}

// CHECK: [[T4:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, undef into undef[word 0]
// CHECK: [[T4_SHORT:%.+]] = vector.extract_strided_slice [[T4]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK: [[W:%.+]] = vector.insert_strided_slice [[T4_SHORT]], [[W1]] {offsets = [8], strides = [1]}
// CHECK: return [[W]]
func.func @vector_trunc_long(%v: vector<9xf32>) -> vector<9xf8E4M3FN> {
  %w = arith.truncf %v : vector<9xf32> to vector<9xf8E4M3FN>
  return %w : vector<9xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @vector_trunc_long_2d
// CHECK-SAME: ([[V:%.+]]: vector<1x9xf32>)
// CHECK: [[ZEROES:%.+]] = arith.constant dense<0.000000e+00> : vector<9xf8E4M3FN>
// CHECK: [[T0:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T1:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T0]][word 1]
// CHECK: [[W0:%.+]] = vector.insert_strided_slice [[T1]], [[ZEROES]] {offsets = [0], strides = [1]}

// CHECK: [[T2:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T3:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T2]][word 1]
// CHECK: [[W1:%.+]] = vector.insert_strided_slice [[T3]], [[W0]] {offsets = [4], strides = [1]}

// CHECK: [[T4:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, undef into undef[word 0]
// CHECK: [[T4_SHORT:%.+]] = vector.extract_strided_slice [[T4]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK: [[W:%.+]] = vector.insert_strided_slice [[T4_SHORT]], [[W1]] {offsets = [8], strides = [1]}
// CHECK: [[RE:%.+]] = vector.shape_cast [[W]] : vector<9xf8E4M3FN> to vector<1x9xf8E4M3FN>
// CHECK: return [[RE]]
func.func @vector_trunc_long_2d(%v: vector<1x9xf32>) -> vector<1x9xf8E4M3FN> {
  %w = arith.truncf %v : vector<1x9xf32> to vector<1x9xf8E4M3FN>
  return %w : vector<1x9xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @vector_ext_long_2d
// CHECK-SAME: ([[V:%.+]]: vector<1x11xf8E4M3FN>)
// CHECK: [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<11xf32>
// CHECK: [[CAST:%.+]] = vector.shape_cast [[V]] : vector<1x11xf8E4M3FN> to vector<11xf8E4M3FN>
// CHECK: [[V0:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [0], sizes = [4], strides = [1]}
// CHECK: [[F0:%.+]] = amdgpu.ext_packed_fp8 [[V0]][0] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W0:%.+]] = vector.insert_strided_slice [[F0]], [[CST]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<11xf32>
// CHECK: [[F1:%.+]] = amdgpu.ext_packed_fp8 [[V0]][1] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W1:%.+]] = vector.insert_strided_slice [[F1]], [[W0]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<11xf32>

// CHECK: [[V1:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<11xf8E4M3FN> to vector<4xf8E4M3FN>
// CHECK: [[F2:%.+]] = amdgpu.ext_packed_fp8 [[V1]][0] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W2:%.+]] = vector.insert_strided_slice [[F2]], [[W1]] {offsets = [4], strides = [1]} : vector<2xf32> into vector<11xf32>
// CHECK: [[F3:%.+]] = amdgpu.ext_packed_fp8 [[V1]][1] : vector<4xf8E4M3FN> to vector<2xf32>
// CHECK: [[W3:%.+]] = vector.insert_strided_slice [[F3]], [[W2]] {offsets = [6], strides = [1]} : vector<2xf32> into vector<11xf32>

// CHECK: [[V2:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [8], sizes = [3], strides = [1]} : vector<11xf8E4M3FN> to vector<3xf8E4M3FN>
// CHECK: [[F4:%.+]] = amdgpu.ext_packed_fp8 [[V2]][0] : vector<3xf8E4M3FN> to vector<2xf32>
// CHECK: [[W4:%.+]] = vector.insert_strided_slice [[F4]], [[W3]] {offsets = [8], strides = [1]} : vector<2xf32> into vector<11xf32>
// CHECK: [[F5:%.+]] = amdgpu.ext_packed_fp8 [[V2]][2] : vector<3xf8E4M3FN> to f32
// CHECK: [[W5:%.+]] = vector.insert [[F5]], [[W4]] [10] : f32 into vector<11xf32>
// CHECK: [[CAST:%.+]] = vector.shape_cast [[W5]] : vector<11xf32> to vector<1x11xf32>
// CHECK: return [[CAST]]
func.func @vector_ext_long_2d(%v: vector<1x11xf8E4M3FN>) -> vector<1x11xf32> {
  %w = arith.extf %v : vector<1x11xf8E4M3FN> to vector<1x11xf32>
  return %w : vector<1x11xf32>
}
