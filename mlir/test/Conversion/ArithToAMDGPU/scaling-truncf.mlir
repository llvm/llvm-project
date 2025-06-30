// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx950" | FileCheck %s

// CHECK-LABEL: @conversion_f8_fallback
// CHECK-DAG:     [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf8E5M2>
// CHECK-DAG:     [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK:         [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_00:%.+]] = vector.shape_cast [[IN_SLICE_00]]
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0]
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_00]] into undef[0], [[SCALE_SCALAR_00]]
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]]
// CHECK-NEXT:    [[OUT_SCALAR_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]]
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_00]], [[CST]]
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_01:%.+]] = vector.shape_cast [[IN_SLICE_01]]
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1]
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_01]] into undef[0], [[SCALE_SCALAR_01]]
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]]
// CHECK-NEXT:    [[OUT_SCALAR_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]]
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_01]], [[ACC_A]]
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_10:%.+]] = vector.shape_cast [[IN_SLICE_10]]
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0]
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_10]] into undef[0], [[SCALE_SCALAR_10]]
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]]
// CHECK-NEXT:    [[OUT_SCALAR_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]]
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_10]], [[ACC_B]]
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_11:%.+]] = vector.shape_cast [[IN_SLICE_11]]
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1]
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_11]] into undef[0], [[SCALE_SCALAR_11]]
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]]
// CHECK-NEXT:    [[OUT_SCALAR_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]]
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_11]], [[ACC_A]]
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf8E5M2>
func.func @conversion_f8_fallback(%in: vector<2x2xf32>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf8E5M2> {
    %ext = arith.scaling_truncf %in, %scale : vector<2x2xf32>, vector<2x2xf8E8M0FNU> to vector<2x2xf8E5M2>
    return %ext : vector<2x2xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_f4_fallback
// CHECK-DAG:     [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf4E2M1FN>
// CHECK-DAG:     [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK:         [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_00:%.+]] = vector.shape_cast [[IN_SLICE_00]]
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0]
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_00]] into undef[0], [[SCALE_SCALAR_00]]
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]]
// CHECK-NEXT:    [[OUT_SCALAR_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]]
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_00]], [[CST]]
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_01:%.+]] = vector.shape_cast [[IN_SLICE_01]]
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1]
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_01]] into undef[0], [[SCALE_SCALAR_01]]
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]]
// CHECK-NEXT:    [[OUT_SCALAR_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]]
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_01]], [[ACC_A]]
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_10:%.+]] = vector.shape_cast [[IN_SLICE_10]]
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0]
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_10]] into undef[0], [[SCALE_SCALAR_10]]
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]]
// CHECK-NEXT:    [[OUT_SCALAR_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]]
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_10]], [[ACC_B]]
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    [[IN_SCALAR_11:%.+]] = vector.shape_cast [[IN_SLICE_11]]
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1]
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.packed_scaled_trunc [[IN_SCALAR_11]] into undef[0], [[SCALE_SCALAR_11]]
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]]
// CHECK-NEXT:    [[OUT_SCALAR_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]]
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_SCALAR_11]], [[ACC_A]]
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf4E2M1FN>
func.func @conversion_f4_fallback(%in: vector<2x2xf32>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf4E2M1FN> {
    %ext = arith.scaling_truncf %in, %scale : vector<2x2xf32>, vector<2x2xf8E8M0FNU> to vector<2x2xf4E2M1FN>
    return %ext : vector<2x2xf4E2M1FN>
}

// -----

// CHECK-LABEL: @conversion_broadcast
// CHECK-DAG:     [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<8x2x4xf8E5M2>
// CHECK-DAG:     [[BCAST:%.+]] = vector.broadcast %arg1
// CHECK-DAG:     [[IN_CAST:%.+]] = vector.shape_cast %arg0 : vector<8x8xf32> to vector<8x2x4xf32>
// CHECK-DAG:     [[SCALE_CAST:%.+]] = vector.shape_cast [[BCAST]]
// CHECK-DAG:     [[SCALE_EXT:%.+]] = arith.extf [[SCALE_CAST]] : vector<8x2x4xf8E8M0FNU> to vector<8x2x4xf32>
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[CST]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][0, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][1, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][2, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][3, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][4, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][5, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][6, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 0, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 0, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 0, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 0, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 1, 0]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 1, 1]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 2]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 1, 2]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_B]]
// CHECK-NEXT:    vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 3]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract [[SCALE_EXT]][7, 1, 3]
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice %{{.*}}, [[ACC_A]]
// CHECK-NEXT:    [[FINAL_CAST:%.+]] = vector.shape_cast [[ACC_B]]
// CHECK-NEXT:    return [[FINAL_CAST]] : vector<8x8xf8E5M2>
func.func @conversion_broadcast(%in: vector<8x8xf32>, %scale: vector<8x2xf8E8M0FNU>) -> vector<8x8xf8E5M2> {
    %bc = vector.broadcast %scale : vector<8x2xf8E8M0FNU> to vector<4x8x2xf8E8M0FNU>
    %cast1 = vector.shape_cast %in : vector<8x8xf32> to vector<8x2x4xf32>
    %cast2 = vector.shape_cast %bc : vector<4x8x2xf8E8M0FNU> to vector<8x2x4xf8E8M0FNU>
    %ext = arith.scaling_truncf %cast1, %cast2 : vector<8x2x4xf32>, vector<8x2x4xf8E8M0FNU> to vector<8x2x4xf8E5M2>
    %cast3 = vector.shape_cast %ext : vector<8x2x4xf8E5M2> to vector<8x8xf8E5M2>
    return %cast3 : vector<8x8xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_scalar
// CHECK:         [[SCALE_F32:%.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK-NEXT:    [[SPLAT_IN:%.+]] = vector.splat %arg0 : vector<1xf32>
// CHECK-NEXT:    [[PACKED_TRUNC:%.+]] = amdgpu.packed_scaled_trunc [[SPLAT_IN]] into undef[0], [[SCALE_F32]]
// CHECK-NEXT:    [[RESULT:%.+]] = vector.extract [[PACKED_TRUNC]][0]
// CHECK-NEXT:    return [[RESULT]] : f8E5M2
func.func @conversion_scalar(%in: f32, %scale: f8E8M0FNU) -> f8E5M2 {
    %ext = arith.scaling_truncf %in, %scale : f32, f8E8M0FNU to f8E5M2
    return %ext : f8E5M2
}
