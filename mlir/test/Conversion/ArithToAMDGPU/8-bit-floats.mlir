// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx940" | FileCheck %s

// CHECK-LABEL: func.func @scalar_ext
// CHECK-SAME: ([[V:%.+]]: f8E5M2FNUZ)
// CHECK: [[FLOAT:%.+]] = amdgpu.ext_packed_fp8 [[V]][0] : f8E5M2FNUZ to f32
// CHECK: [[W:%.+]] = arith.truncf [[FLOAT]] : f32 to f16
// CHECK: return [[W]]
func.func @scalar_ext(%v: f8E5M2FNUZ) -> f16 {
  %w = arith.extf %v : f8E5M2FNUZ to f16
  return %w : f16
}

// No 0-D test because arith.extf hasn't been extended to support it.

// -----

// CHECK-LABEL: func.func @vector_ext_short
// CHECK-SAME: ([[V:%.+]]: vector<2xf8E5M2FNUZ>)
// CHECK-DAG: [[ZEROES:%.+]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK: [[FLOAT0:%.+]] = amdgpu.ext_packed_fp8 [[V]][0] : vector<2xf8E5M2FNUZ> to f32
// CHECK: [[EXT0:%.+]] = arith.extf [[FLOAT0]] : f32 to f64
// CHECK: [[W0:%.+]] = vector.insert [[EXT0]], [[ZEROES]] [0]
// CHECK: [[FLOAT1:%.+]] = amdgpu.ext_packed_fp8 [[V]][1] : vector<2xf8E5M2FNUZ> to f32
// CHECK: [[EXT1:%.+]] = arith.extf [[FLOAT1]]
// CHECK: [[W1:%.+]] = vector.insert [[EXT1]], [[W0]] [1]
// CHECK: return [[W1]] : vector<2xf64>

func.func @vector_ext_short(%v: vector<2xf8E5M2FNUZ>) -> vector<2xf64> {
  %w = arith.extf %v : vector<2xf8E5M2FNUZ> to vector<2xf64>
  return %w : vector<2xf64>
}

// -----

// CHECK-LABEL: func.func @vector_ext_long
// CHECK-SAME: ([[V:%.+]]: vector<9xf8E4M3FNUZ>)
// CHECK: [[V0:%.+]] = vector.extract_strided_slice [[V]] {offsets = [0], sizes = [4], strides = [1]}
// CHECK: [[F0:%.+]] = amdgpu.ext_packed_fp8 [[V0]][0]
// CHECK: [[W0:%.+]] = vector.insert [[F0]]
// CHECK: [[F1:%.+]] = amdgpu.ext_packed_fp8 [[V0]][1]
// CHECK: [[W1:%.+]] = vector.insert [[F1]], [[W0]]
// CHECK: [[F2:%.+]] = amdgpu.ext_packed_fp8 [[V0]][2]
// CHECK: [[W2:%.+]] = vector.insert [[F2]], [[W1]]
// CHECK: [[F3:%.+]] = amdgpu.ext_packed_fp8 [[V0]][3]
// CHECK: [[W3:%.+]] = vector.insert [[F3]], [[W2]]

// CHECK: [[V1:%.+]] = vector.extract_strided_slice [[V]] {offsets = [4], sizes = [4], strides = [1]} : vector<9xf8E4M3FNUZ> to vector<4xf8E4M3FNUZ>
// CHECK: [[F4:%.+]] = amdgpu.ext_packed_fp8 [[V1]][0]
// CHECK: [[W4:%.+]] = vector.insert [[F4]], [[W3]]
// CHECK: [[F5:%.+]] = amdgpu.ext_packed_fp8 [[V1]][1]
// CHECK: [[W5:%.+]] = vector.insert [[F5]], [[W4]]
// CHECK: [[F6:%.+]] = amdgpu.ext_packed_fp8 [[V1]][2]
// CHECK: [[W6:%.+]] = vector.insert [[F6]], [[W5]]
// CHECK: [[F7:%.+]] = amdgpu.ext_packed_fp8 [[V1]][3]
// CHECK: [[W7:%.+]] = vector.insert [[F7]], [[W6]]

// CHECK: [[V2:%.+]] = vector.extract_strided_slice [[V]] {offsets = [8], sizes = [1], strides = [1]} : vector<9xf8E4M3FNUZ> to vector<1xf8E4M3FNUZ>
// CHECK: [[F8:%.+]] = amdgpu.ext_packed_fp8 [[V2]][0]
// CHECK: [[W8:%.+]] = vector.insert [[F8]], [[W7]]
// CHECK: return [[W8]]
func.func @vector_ext_long(%v: vector<9xf8E4M3FNUZ>) -> vector<9xf32> {
  %w = arith.extf %v : vector<9xf8E4M3FNUZ> to vector<9xf32>
  return %w : vector<9xf32>
}

// -----

// CHECK-LABEL: func.func @scalar_trunc
// CHECK-SAME: ([[V:%.+]]: f16)
// CHECK: [[FLOAT:%.+]] = arith.extf [[V]] : f16 to f32
// CHECK: [[TRUNCV:%.+]] = amdgpu.packed_trunc_2xfp8 [[FLOAT]], undef into undef[word 0] : f32 to vector<4xf8E5M2FNUZ>
// CHECK: [[W:%.+]] = vector.extract [[TRUNCV]][0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
// CHECK: return [[W]] : f8E5M2FNUZ
func.func @scalar_trunc(%v: f16) -> f8E5M2FNUZ {
  %w = arith.truncf %v : f16 to f8E5M2FNUZ
  return %w : f8E5M2FNUZ
}

// No 0-D test because arith.truncf hasn't been extended to support it.

// -----

// CHECK-LABEL: func.func @vector_trunc_short
// CHECK-SAME: ([[V:%.+]]: vector<2xf64>) -> vector<2xf8E5M2FNUZ> {
// CHECK: [[V0:%.+]] = vector.extract [[V]][0]
// CHECK: [[F0:%.+]] = arith.truncf [[V0]] : f64 to f32
// CHECK: [[V1:%.+]] = vector.extract [[V]][1]
// CHECK: [[F1:%.+]] = arith.truncf [[V1]] : f64 to f32
// CHECK: [[W0:%.+]] = amdgpu.packed_trunc_2xfp8 [[F0]], [[F1]] into undef[word 0] : f32 to vector<4xf8E5M2FNUZ>
// CHECK: [[W:%.+]] = vector.extract_strided_slice [[W0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2FNUZ> to vector<2xf8E5M2FNUZ>
// CHECK: return [[W]] : vector<2xf8E5M2FNUZ>
func.func @vector_trunc_short(%v: vector<2xf64>) -> vector<2xf8E5M2FNUZ> {
  %w = arith.truncf %v : vector<2xf64> to vector<2xf8E5M2FNUZ>
  return %w : vector<2xf8E5M2FNUZ>
}

// -----

// CHECK-LABEL: func.func @vector_trunc_long
// CHECK-SAME: ([[V:%.+]]: vector<9xf32>)
// CHECK: [[ZEROES:%.+]] = arith.constant dense<0.000000e+00> : vector<9xf8E4M3FNUZ>
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
func.func @vector_trunc_long(%v: vector<9xf32>) -> vector<9xf8E4M3FNUZ> {
  %w = arith.truncf %v : vector<9xf32> to vector<9xf8E4M3FNUZ>
  return %w : vector<9xf8E4M3FNUZ>
}

// -----

// CHECK-LABEL: func.func @vector_trunc_long_2d
// CHECK-SAME: ([[V:%.+]]: vector<1x9xf32>)
// CHECK: [[ZEROES:%.+]] = arith.constant dense<0.000000e+00> : vector<9xf8E4M3FNUZ>
// CHECK: [[T0:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T1:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T0]][word 1]
// CHECK: [[W0:%.+]] = vector.insert_strided_slice [[T1]], [[ZEROES]] {offsets = [0], strides = [1]}

// CHECK: [[T2:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into undef[word 0]
// CHECK: [[T3:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, %{{.+}} into [[T2]][word 1]
// CHECK: [[W1:%.+]] = vector.insert_strided_slice [[T3]], [[W0]] {offsets = [4], strides = [1]}

// CHECK: [[T4:%.+]] = amdgpu.packed_trunc_2xfp8 %{{.+}}, undef into undef[word 0]
// CHECK: [[T4_SHORT:%.+]] = vector.extract_strided_slice [[T4]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK: [[W:%.+]] = vector.insert_strided_slice [[T4_SHORT]], [[W1]] {offsets = [8], strides = [1]}
// CHECK: [[RE:%.+]] = vector.shape_cast [[W]] : vector<9xf8E4M3FNUZ> to vector<1x9xf8E4M3FNUZ>
// CHECK: return [[RE]]
func.func @vector_trunc_long_2d(%v: vector<1x9xf32>) -> vector<1x9xf8E4M3FNUZ> {
  %w = arith.truncf %v : vector<1x9xf32> to vector<1x9xf8E4M3FNUZ>
  return %w : vector<1x9xf8E4M3FNUZ>
}

// -----

// CHECK-LABEL: func.func @vector_ext_long_2d
// CHECK-SAME: ([[V:%.+]]: vector<1x9xf8E4M3FNUZ>)
// CHECK: [[CAST:%.+]] = vector.shape_cast [[V]] : vector<1x9xf8E4M3FNUZ> to vector<9xf8E4M3FNUZ>
// CHECK: [[V0:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [0], sizes = [4], strides = [1]}
// CHECK: [[F0:%.+]] = amdgpu.ext_packed_fp8 [[V0]][0]
// CHECK: [[W0:%.+]] = vector.insert [[F0]]
// CHECK: [[F1:%.+]] = amdgpu.ext_packed_fp8 [[V0]][1]
// CHECK: [[W1:%.+]] = vector.insert [[F1]], [[W0]]
// CHECK: [[F2:%.+]] = amdgpu.ext_packed_fp8 [[V0]][2]
// CHECK: [[W2:%.+]] = vector.insert [[F2]], [[W1]]
// CHECK: [[F3:%.+]] = amdgpu.ext_packed_fp8 [[V0]][3]
// CHECK: [[W3:%.+]] = vector.insert [[F3]], [[W2]]

// CHECK: [[V1:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<9xf8E4M3FNUZ> to vector<4xf8E4M3FNUZ>
// CHECK: [[F4:%.+]] = amdgpu.ext_packed_fp8 [[V1]][0]
// CHECK: [[W4:%.+]] = vector.insert [[F4]], [[W3]]
// CHECK: [[F5:%.+]] = amdgpu.ext_packed_fp8 [[V1]][1]
// CHECK: [[W5:%.+]] = vector.insert [[F5]], [[W4]]
// CHECK: [[F6:%.+]] = amdgpu.ext_packed_fp8 [[V1]][2]
// CHECK: [[W6:%.+]] = vector.insert [[F6]], [[W5]]
// CHECK: [[F7:%.+]] = amdgpu.ext_packed_fp8 [[V1]][3]
// CHECK: [[W7:%.+]] = vector.insert [[F7]], [[W6]]

// CHECK: [[V2:%.+]] = vector.extract_strided_slice [[CAST]] {offsets = [8], sizes = [1], strides = [1]} : vector<9xf8E4M3FNUZ> to vector<1xf8E4M3FNUZ>
// CHECK: [[F8:%.+]] = amdgpu.ext_packed_fp8 [[V2]][0]
// CHECK: [[W8:%.+]] = vector.insert [[F8]], [[W7]]
// CHECK: [[CAST:%.+]] = vector.shape_cast [[W8]] : vector<9xf32> to vector<1x9xf32>
// CHECK: return [[CAST]]
func.func @vector_ext_long_2d(%v: vector<1x9xf8E4M3FNUZ>) -> vector<1x9xf32> {
  %w = arith.extf %v : vector<1x9xf8E4M3FNUZ> to vector<1x9xf32>
  return %w : vector<1x9xf32>
}
