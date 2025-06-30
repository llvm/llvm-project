// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx950" | FileCheck %s

// CHECK-LABEL: @conversion_f8_f32_fallback
// CHECK:         [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK-NEXT:    [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_00:%.+]] = vector.shape_cast [[IN_SLICE_00]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_00]][0], [[SCALE_SCALAR_00]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_00]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_01:%.+]] = vector.shape_cast [[IN_SLICE_01]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_01]][0], [[SCALE_SCALAR_01]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_01]], [[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_10:%.+]] = vector.shape_cast [[IN_SLICE_10]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_10]][0], [[SCALE_SCALAR_10]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_10]], [[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_11:%.+]] = vector.shape_cast [[IN_SLICE_11]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_11]][0], [[SCALE_SCALAR_11]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_11]], [[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf32>
func.func @conversion_f8_f32_fallback(%in: vector<2x2xf8E5M2>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf32> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf8E5M2>, vector<2x2xf8E8M0FNU> to vector<2x2xf32>
    return %ext : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @conversion_f4_f32_fallback
// CHECK:         [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK-NEXT:    [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_00:%.+]] = vector.shape_cast [[IN_SLICE_00]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_00]][0], [[SCALE_SCALAR_00]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_00]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_01:%.+]] = vector.shape_cast [[IN_SLICE_01]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_01]][0], [[SCALE_SCALAR_01]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_01]], [[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_10:%.+]] = vector.shape_cast [[IN_SLICE_10]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_10]][0], [[SCALE_SCALAR_10]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_10]], [[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_11:%.+]] = vector.shape_cast [[IN_SLICE_11]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_11]][0], [[SCALE_SCALAR_11]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_11]], [[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf32>
func.func @conversion_f4_f32_fallback(%in: vector<2x2xf4E2M1FN>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf32> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf4E2M1FN>, vector<2x2xf8E8M0FNU> to vector<2x2xf32>
    return %ext : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @conversion_f8_f16_fallback
// CHECK:         [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK-NEXT:    [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_00:%.+]] = vector.shape_cast [[IN_SLICE_00]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_00]][0], [[SCALE_SCALAR_00]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_00]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_01:%.+]] = vector.shape_cast [[IN_SLICE_01]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_01]][0], [[SCALE_SCALAR_01]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_01]], [[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_10:%.+]] = vector.shape_cast [[IN_SLICE_10]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_10]][0], [[SCALE_SCALAR_10]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_10]], [[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_11:%.+]] = vector.shape_cast [[IN_SLICE_11]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_11]][0], [[SCALE_SCALAR_11]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_11]], [[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf16>
func.func @conversion_f8_f16_fallback(%in: vector<2x2xf8E5M2>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf16> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf8E5M2>, vector<2x2xf8E8M0FNU> to vector<2x2xf16>
    return %ext : vector<2x2xf16>
}

// -----

// CHECK-LABEL: @conversion_f4_f16_fallback
// CHECK:         [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK-NEXT:    [[SCALE_EXT:%.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    [[IN_SLICE_00:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_00:%.+]] = vector.shape_cast [[IN_SLICE_00]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_00:%.+]] = vector.extract [[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_00:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_00]][0], [[SCALE_SCALAR_00]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_00:%.+]] = vector.extract_strided_slice [[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_00:%.+]] = vector.shape_cast [[OUT_SLICE_00]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_00]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_01:%.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_01:%.+]] = vector.shape_cast [[IN_SLICE_01]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_01:%.+]] = vector.extract [[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_01:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_01]][0], [[SCALE_SCALAR_01]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_01:%.+]] = vector.extract_strided_slice [[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_01:%.+]] = vector.shape_cast [[OUT_SLICE_01]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_01]], [[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_10:%.+]] = vector.shape_cast [[IN_SLICE_10]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_10]][0], [[SCALE_SCALAR_10]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_10]], [[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    [[IN_VEC_11:%.+]] = vector.shape_cast [[IN_SLICE_11]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_11]][0], [[SCALE_SCALAR_11]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    [[OUT_VEC_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_11]], [[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    return [[ACC_B]] : vector<2x2xf16>
func.func @conversion_f4_f16_fallback(%in: vector<2x2xf4E2M1FN>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf16> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf4E2M1FN>, vector<2x2xf8E8M0FNU> to vector<2x2xf16>
    return %ext : vector<2x2xf16>
}

// -----

// CHECK-LABEL: @conversion_broadcast
// CHECK:         [[CST:%.+]] = arith.constant dense<0.000000e+00> : vector<8x2x4xf32>
// CHECK-NEXT:    [[BCAST:%.+]] = vector.broadcast %arg1 : vector<8x2xf8E8M0FNU> to vector<4x8x2xf8E8M0FNU>
// CHECK-NEXT:    [[IN_CAST:%.+]] = vector.shape_cast %arg0 : vector<8x8xf8E5M2> to vector<8x2x4xf8E5M2>
// CHECK-NEXT:    [[SCALE_CAST:%.+]] = vector.shape_cast [[BCAST]] : vector<4x8x2xf8E8M0FNU> to vector<8x2x4xf8E8M0FNU>
// CHECK-NEXT:    [[SCALE_EXT:%.+]] = arith.extf [[SCALE_CAST]] : vector<8x2x4xf8E8M0FNU> to vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_0:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_0:%.+]] = vector.shape_cast [[IN_SLICE_0]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_0:%.+]] = vector.extract [[SCALE_EXT]][0, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_0:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_0]][0], [[SCALE_SCALAR_0]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_0:%.+]] = vector.extract_strided_slice [[PACKED_0]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_0:%.+]] = vector.shape_cast [[OUT_SLICE_0]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_0]], [[CST]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_1:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_1:%.+]] = vector.shape_cast [[IN_SLICE_1]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_1:%.+]] = vector.extract [[SCALE_EXT]][0, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_1:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_1]][0], [[SCALE_SCALAR_1]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_1:%.+]] = vector.extract_strided_slice [[PACKED_1]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_1:%.+]] = vector.shape_cast [[OUT_SLICE_1]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_1]], [[ACC_A]] {offsets = [0, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_2:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_2:%.+]] = vector.shape_cast [[IN_SLICE_2]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_2:%.+]] = vector.extract [[SCALE_EXT]][0, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_2:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_2]][0], [[SCALE_SCALAR_2]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_2:%.+]] = vector.extract_strided_slice [[PACKED_2]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_2:%.+]] = vector.shape_cast [[OUT_SLICE_2]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_2]], [[ACC_B]] {offsets = [0, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_3:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_3:%.+]] = vector.shape_cast [[IN_SLICE_3]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_3:%.+]] = vector.extract [[SCALE_EXT]][0, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_3:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_3]][0], [[SCALE_SCALAR_3]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_3:%.+]] = vector.extract_strided_slice [[PACKED_3]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_3:%.+]] = vector.shape_cast [[OUT_SLICE_3]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_3]], [[ACC_A]] {offsets = [0, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_4:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_4:%.+]] = vector.shape_cast [[IN_SLICE_4]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_4:%.+]] = vector.extract [[SCALE_EXT]][0, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_4:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_4]][0], [[SCALE_SCALAR_4]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_4:%.+]] = vector.extract_strided_slice [[PACKED_4]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_4:%.+]] = vector.shape_cast [[OUT_SLICE_4]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_4]], [[ACC_B]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_5:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_5:%.+]] = vector.shape_cast [[IN_SLICE_5]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_5:%.+]] = vector.extract [[SCALE_EXT]][0, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_5:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_5]][0], [[SCALE_SCALAR_5]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_5:%.+]] = vector.extract_strided_slice [[PACKED_5]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_5:%.+]] = vector.shape_cast [[OUT_SLICE_5]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_5]], [[ACC_A]] {offsets = [0, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_6:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_6:%.+]] = vector.shape_cast [[IN_SLICE_6]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_6:%.+]] = vector.extract [[SCALE_EXT]][0, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_6:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_6]][0], [[SCALE_SCALAR_6]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_6:%.+]] = vector.extract_strided_slice [[PACKED_6]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_6:%.+]] = vector.shape_cast [[OUT_SLICE_6]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_6]], [[ACC_B]] {offsets = [0, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_7:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [0, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_7:%.+]] = vector.shape_cast [[IN_SLICE_7]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_7:%.+]] = vector.extract [[SCALE_EXT]][0, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_7:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_7]][0], [[SCALE_SCALAR_7]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_7:%.+]] = vector.extract_strided_slice [[PACKED_7]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_7:%.+]] = vector.shape_cast [[OUT_SLICE_7]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_7]], [[ACC_A]] {offsets = [0, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_8:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_8:%.+]] = vector.shape_cast [[IN_SLICE_8]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_8:%.+]] = vector.extract [[SCALE_EXT]][1, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_8:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_8]][0], [[SCALE_SCALAR_8]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_8:%.+]] = vector.extract_strided_slice [[PACKED_8]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_8:%.+]] = vector.shape_cast [[OUT_SLICE_8]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_8]], [[ACC_B]] {offsets = [1, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_9:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_9:%.+]] = vector.shape_cast [[IN_SLICE_9]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_9:%.+]] = vector.extract [[SCALE_EXT]][1, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_9:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_9]][0], [[SCALE_SCALAR_9]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_9:%.+]] = vector.extract_strided_slice [[PACKED_9]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_9:%.+]] = vector.shape_cast [[OUT_SLICE_9]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_9]], [[ACC_A]] {offsets = [1, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_10:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_10:%.+]] = vector.shape_cast [[IN_SLICE_10]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_10:%.+]] = vector.extract [[SCALE_EXT]][1, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_10:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_10]][0], [[SCALE_SCALAR_10]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_10:%.+]] = vector.extract_strided_slice [[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_10:%.+]] = vector.shape_cast [[OUT_SLICE_10]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_10]], [[ACC_B]] {offsets = [1, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_11:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_11:%.+]] = vector.shape_cast [[IN_SLICE_11]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_11:%.+]] = vector.extract [[SCALE_EXT]][1, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_11:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_11]][0], [[SCALE_SCALAR_11]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_11:%.+]] = vector.extract_strided_slice [[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_11:%.+]] = vector.shape_cast [[OUT_SLICE_11]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_11]], [[ACC_A]] {offsets = [1, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_12:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_12:%.+]] = vector.shape_cast [[IN_SLICE_12]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_12:%.+]] = vector.extract [[SCALE_EXT]][1, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_12:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_12]][0], [[SCALE_SCALAR_12]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_12:%.+]] = vector.extract_strided_slice [[PACKED_12]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_12:%.+]] = vector.shape_cast [[OUT_SLICE_12]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_12]], [[ACC_B]] {offsets = [1, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_13:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_13:%.+]] = vector.shape_cast [[IN_SLICE_13]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_13:%.+]] = vector.extract [[SCALE_EXT]][1, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_13:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_13]][0], [[SCALE_SCALAR_13]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_13:%.+]] = vector.extract_strided_slice [[PACKED_13]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_13:%.+]] = vector.shape_cast [[OUT_SLICE_13]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_13]], [[ACC_A]] {offsets = [1, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_14:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_14:%.+]] = vector.shape_cast [[IN_SLICE_14]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_14:%.+]] = vector.extract [[SCALE_EXT]][1, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_14:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_14]][0], [[SCALE_SCALAR_14]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_14:%.+]] = vector.extract_strided_slice [[PACKED_14]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_14:%.+]] = vector.shape_cast [[OUT_SLICE_14]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_14]], [[ACC_B]] {offsets = [1, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_15:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [1, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_15:%.+]] = vector.shape_cast [[IN_SLICE_15]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_15:%.+]] = vector.extract [[SCALE_EXT]][1, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_15:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_15]][0], [[SCALE_SCALAR_15]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_15:%.+]] = vector.extract_strided_slice [[PACKED_15]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_15:%.+]] = vector.shape_cast [[OUT_SLICE_15]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_15]], [[ACC_A]] {offsets = [1, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_16:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_16:%.+]] = vector.shape_cast [[IN_SLICE_16]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_16:%.+]] = vector.extract [[SCALE_EXT]][2, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_16:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_16]][0], [[SCALE_SCALAR_16]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_16:%.+]] = vector.extract_strided_slice [[PACKED_16]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_16:%.+]] = vector.shape_cast [[OUT_SLICE_16]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_16]], [[ACC_B]] {offsets = [2, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_17:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_17:%.+]] = vector.shape_cast [[IN_SLICE_17]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_17:%.+]] = vector.extract [[SCALE_EXT]][2, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_17:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_17]][0], [[SCALE_SCALAR_17]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_17:%.+]] = vector.extract_strided_slice [[PACKED_17]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_17:%.+]] = vector.shape_cast [[OUT_SLICE_17]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_17]], [[ACC_A]] {offsets = [2, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_18:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_18:%.+]] = vector.shape_cast [[IN_SLICE_18]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_18:%.+]] = vector.extract [[SCALE_EXT]][2, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_18:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_18]][0], [[SCALE_SCALAR_18]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_18:%.+]] = vector.extract_strided_slice [[PACKED_18]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_18:%.+]] = vector.shape_cast [[OUT_SLICE_18]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_18]], [[ACC_B]] {offsets = [2, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_19:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_19:%.+]] = vector.shape_cast [[IN_SLICE_19]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_19:%.+]] = vector.extract [[SCALE_EXT]][2, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_19:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_19]][0], [[SCALE_SCALAR_19]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_19:%.+]] = vector.extract_strided_slice [[PACKED_19]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_19:%.+]] = vector.shape_cast [[OUT_SLICE_19]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_19]], [[ACC_A]] {offsets = [2, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_20:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_20:%.+]] = vector.shape_cast [[IN_SLICE_20]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_20:%.+]] = vector.extract [[SCALE_EXT]][2, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_20:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_20]][0], [[SCALE_SCALAR_20]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_20:%.+]] = vector.extract_strided_slice [[PACKED_20]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_20:%.+]] = vector.shape_cast [[OUT_SLICE_20]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_20]], [[ACC_B]] {offsets = [2, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_21:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_21:%.+]] = vector.shape_cast [[IN_SLICE_21]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_21:%.+]] = vector.extract [[SCALE_EXT]][2, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_21:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_21]][0], [[SCALE_SCALAR_21]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_21:%.+]] = vector.extract_strided_slice [[PACKED_21]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_21:%.+]] = vector.shape_cast [[OUT_SLICE_21]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_21]], [[ACC_A]] {offsets = [2, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_22:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_22:%.+]] = vector.shape_cast [[IN_SLICE_22]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_22:%.+]] = vector.extract [[SCALE_EXT]][2, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_22:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_22]][0], [[SCALE_SCALAR_22]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_22:%.+]] = vector.extract_strided_slice [[PACKED_22]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_22:%.+]] = vector.shape_cast [[OUT_SLICE_22]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_22]], [[ACC_B]] {offsets = [2, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_23:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [2, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_23:%.+]] = vector.shape_cast [[IN_SLICE_23]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_23:%.+]] = vector.extract [[SCALE_EXT]][2, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_23:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_23]][0], [[SCALE_SCALAR_23]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_23:%.+]] = vector.extract_strided_slice [[PACKED_23]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_23:%.+]] = vector.shape_cast [[OUT_SLICE_23]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_23]], [[ACC_A]] {offsets = [2, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_24:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_24:%.+]] = vector.shape_cast [[IN_SLICE_24]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_24:%.+]] = vector.extract [[SCALE_EXT]][3, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_24:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_24]][0], [[SCALE_SCALAR_24]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_24:%.+]] = vector.extract_strided_slice [[PACKED_24]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_24:%.+]] = vector.shape_cast [[OUT_SLICE_24]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_24]], [[ACC_B]] {offsets = [3, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_25:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_25:%.+]] = vector.shape_cast [[IN_SLICE_25]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_25:%.+]] = vector.extract [[SCALE_EXT]][3, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_25:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_25]][0], [[SCALE_SCALAR_25]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_25:%.+]] = vector.extract_strided_slice [[PACKED_25]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_25:%.+]] = vector.shape_cast [[OUT_SLICE_25]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_25]], [[ACC_A]] {offsets = [3, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_26:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_26:%.+]] = vector.shape_cast [[IN_SLICE_26]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_26:%.+]] = vector.extract [[SCALE_EXT]][3, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_26:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_26]][0], [[SCALE_SCALAR_26]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_26:%.+]] = vector.extract_strided_slice [[PACKED_26]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_26:%.+]] = vector.shape_cast [[OUT_SLICE_26]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_26]], [[ACC_B]] {offsets = [3, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_27:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_27:%.+]] = vector.shape_cast [[IN_SLICE_27]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_27:%.+]] = vector.extract [[SCALE_EXT]][3, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_27:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_27]][0], [[SCALE_SCALAR_27]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_27:%.+]] = vector.extract_strided_slice [[PACKED_27]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_27:%.+]] = vector.shape_cast [[OUT_SLICE_27]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_27]], [[ACC_A]] {offsets = [3, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_28:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_28:%.+]] = vector.shape_cast [[IN_SLICE_28]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_28:%.+]] = vector.extract [[SCALE_EXT]][3, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_28:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_28]][0], [[SCALE_SCALAR_28]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_28:%.+]] = vector.extract_strided_slice [[PACKED_28]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_28:%.+]] = vector.shape_cast [[OUT_SLICE_28]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_28]], [[ACC_B]] {offsets = [3, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_29:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_29:%.+]] = vector.shape_cast [[IN_SLICE_29]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_29:%.+]] = vector.extract [[SCALE_EXT]][3, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_29:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_29]][0], [[SCALE_SCALAR_29]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_29:%.+]] = vector.extract_strided_slice [[PACKED_29]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_29:%.+]] = vector.shape_cast [[OUT_SLICE_29]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_29]], [[ACC_A]] {offsets = [3, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_30:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_30:%.+]] = vector.shape_cast [[IN_SLICE_30]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_30:%.+]] = vector.extract [[SCALE_EXT]][3, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_30:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_30]][0], [[SCALE_SCALAR_30]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_30:%.+]] = vector.extract_strided_slice [[PACKED_30]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_30:%.+]] = vector.shape_cast [[OUT_SLICE_30]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_30]], [[ACC_B]] {offsets = [3, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_31:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [3, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_31:%.+]] = vector.shape_cast [[IN_SLICE_31]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_31:%.+]] = vector.extract [[SCALE_EXT]][3, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_31:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_31]][0], [[SCALE_SCALAR_31]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_31:%.+]] = vector.extract_strided_slice [[PACKED_31]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_31:%.+]] = vector.shape_cast [[OUT_SLICE_31]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_31]], [[ACC_A]] {offsets = [3, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_32:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_32:%.+]] = vector.shape_cast [[IN_SLICE_32]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_32:%.+]] = vector.extract [[SCALE_EXT]][4, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_32:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_32]][0], [[SCALE_SCALAR_32]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_32:%.+]] = vector.extract_strided_slice [[PACKED_32]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_32:%.+]] = vector.shape_cast [[OUT_SLICE_32]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_32]], [[ACC_B]] {offsets = [4, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_33:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_33:%.+]] = vector.shape_cast [[IN_SLICE_33]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_33:%.+]] = vector.extract [[SCALE_EXT]][4, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_33:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_33]][0], [[SCALE_SCALAR_33]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_33:%.+]] = vector.extract_strided_slice [[PACKED_33]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_33:%.+]] = vector.shape_cast [[OUT_SLICE_33]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_33]], [[ACC_A]] {offsets = [4, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_34:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_34:%.+]] = vector.shape_cast [[IN_SLICE_34]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_34:%.+]] = vector.extract [[SCALE_EXT]][4, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_34:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_34]][0], [[SCALE_SCALAR_34]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_34:%.+]] = vector.extract_strided_slice [[PACKED_34]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_34:%.+]] = vector.shape_cast [[OUT_SLICE_34]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_34]], [[ACC_B]] {offsets = [4, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_35:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_35:%.+]] = vector.shape_cast [[IN_SLICE_35]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_35:%.+]] = vector.extract [[SCALE_EXT]][4, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_35:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_35]][0], [[SCALE_SCALAR_35]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_35:%.+]] = vector.extract_strided_slice [[PACKED_35]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_35:%.+]] = vector.shape_cast [[OUT_SLICE_35]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_35]], [[ACC_A]] {offsets = [4, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_36:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_36:%.+]] = vector.shape_cast [[IN_SLICE_36]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_36:%.+]] = vector.extract [[SCALE_EXT]][4, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_36:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_36]][0], [[SCALE_SCALAR_36]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_36:%.+]] = vector.extract_strided_slice [[PACKED_36]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_36:%.+]] = vector.shape_cast [[OUT_SLICE_36]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_36]], [[ACC_B]] {offsets = [4, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_37:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_37:%.+]] = vector.shape_cast [[IN_SLICE_37]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_37:%.+]] = vector.extract [[SCALE_EXT]][4, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_37:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_37]][0], [[SCALE_SCALAR_37]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_37:%.+]] = vector.extract_strided_slice [[PACKED_37]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_37:%.+]] = vector.shape_cast [[OUT_SLICE_37]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_37]], [[ACC_A]] {offsets = [4, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_38:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_38:%.+]] = vector.shape_cast [[IN_SLICE_38]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_38:%.+]] = vector.extract [[SCALE_EXT]][4, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_38:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_38]][0], [[SCALE_SCALAR_38]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_38:%.+]] = vector.extract_strided_slice [[PACKED_38]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_38:%.+]] = vector.shape_cast [[OUT_SLICE_38]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_38]], [[ACC_B]] {offsets = [4, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_39:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [4, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_39:%.+]] = vector.shape_cast [[IN_SLICE_39]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_39:%.+]] = vector.extract [[SCALE_EXT]][4, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_39:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_39]][0], [[SCALE_SCALAR_39]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_39:%.+]] = vector.extract_strided_slice [[PACKED_39]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_39:%.+]] = vector.shape_cast [[OUT_SLICE_39]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_39]], [[ACC_A]] {offsets = [4, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_40:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_40:%.+]] = vector.shape_cast [[IN_SLICE_40]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_40:%.+]] = vector.extract [[SCALE_EXT]][5, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_40:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_40]][0], [[SCALE_SCALAR_40]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_40:%.+]] = vector.extract_strided_slice [[PACKED_40]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_40:%.+]] = vector.shape_cast [[OUT_SLICE_40]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_40]], [[ACC_B]] {offsets = [5, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_41:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_41:%.+]] = vector.shape_cast [[IN_SLICE_41]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_41:%.+]] = vector.extract [[SCALE_EXT]][5, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_41:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_41]][0], [[SCALE_SCALAR_41]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_41:%.+]] = vector.extract_strided_slice [[PACKED_41]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_41:%.+]] = vector.shape_cast [[OUT_SLICE_41]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_41]], [[ACC_A]] {offsets = [5, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_42:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_42:%.+]] = vector.shape_cast [[IN_SLICE_42]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_42:%.+]] = vector.extract [[SCALE_EXT]][5, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_42:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_42]][0], [[SCALE_SCALAR_42]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_42:%.+]] = vector.extract_strided_slice [[PACKED_42]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_42:%.+]] = vector.shape_cast [[OUT_SLICE_42]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_42]], [[ACC_B]] {offsets = [5, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_43:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_43:%.+]] = vector.shape_cast [[IN_SLICE_43]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_43:%.+]] = vector.extract [[SCALE_EXT]][5, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_43:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_43]][0], [[SCALE_SCALAR_43]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_43:%.+]] = vector.extract_strided_slice [[PACKED_43]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_43:%.+]] = vector.shape_cast [[OUT_SLICE_43]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_43]], [[ACC_A]] {offsets = [5, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_44:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_44:%.+]] = vector.shape_cast [[IN_SLICE_44]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_44:%.+]] = vector.extract [[SCALE_EXT]][5, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_44:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_44]][0], [[SCALE_SCALAR_44]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_44:%.+]] = vector.extract_strided_slice [[PACKED_44]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_44:%.+]] = vector.shape_cast [[OUT_SLICE_44]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_44]], [[ACC_B]] {offsets = [5, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_45:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_45:%.+]] = vector.shape_cast [[IN_SLICE_45]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_45:%.+]] = vector.extract [[SCALE_EXT]][5, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_45:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_45]][0], [[SCALE_SCALAR_45]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_45:%.+]] = vector.extract_strided_slice [[PACKED_45]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_45:%.+]] = vector.shape_cast [[OUT_SLICE_45]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_45]], [[ACC_A]] {offsets = [5, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_46:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_46:%.+]] = vector.shape_cast [[IN_SLICE_46]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_46:%.+]] = vector.extract [[SCALE_EXT]][5, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_46:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_46]][0], [[SCALE_SCALAR_46]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_46:%.+]] = vector.extract_strided_slice [[PACKED_46]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_46:%.+]] = vector.shape_cast [[OUT_SLICE_46]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_46]], [[ACC_B]] {offsets = [5, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_47:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [5, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_47:%.+]] = vector.shape_cast [[IN_SLICE_47]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_47:%.+]] = vector.extract [[SCALE_EXT]][5, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_47:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_47]][0], [[SCALE_SCALAR_47]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_47:%.+]] = vector.extract_strided_slice [[PACKED_47]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_47:%.+]] = vector.shape_cast [[OUT_SLICE_47]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_47]], [[ACC_A]] {offsets = [5, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_48:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_48:%.+]] = vector.shape_cast [[IN_SLICE_48]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_48:%.+]] = vector.extract [[SCALE_EXT]][6, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_48:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_48]][0], [[SCALE_SCALAR_48]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_48:%.+]] = vector.extract_strided_slice [[PACKED_48]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_48:%.+]] = vector.shape_cast [[OUT_SLICE_48]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_48]], [[ACC_B]] {offsets = [6, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_49:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_49:%.+]] = vector.shape_cast [[IN_SLICE_49]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_49:%.+]] = vector.extract [[SCALE_EXT]][6, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_49:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_49]][0], [[SCALE_SCALAR_49]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_49:%.+]] = vector.extract_strided_slice [[PACKED_49]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_49:%.+]] = vector.shape_cast [[OUT_SLICE_49]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_49]], [[ACC_A]] {offsets = [6, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_50:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_50:%.+]] = vector.shape_cast [[IN_SLICE_50]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_50:%.+]] = vector.extract [[SCALE_EXT]][6, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_50:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_50]][0], [[SCALE_SCALAR_50]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_50:%.+]] = vector.extract_strided_slice [[PACKED_50]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_50:%.+]] = vector.shape_cast [[OUT_SLICE_50]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_50]], [[ACC_B]] {offsets = [6, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_51:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_51:%.+]] = vector.shape_cast [[IN_SLICE_51]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_51:%.+]] = vector.extract [[SCALE_EXT]][6, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_51:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_51]][0], [[SCALE_SCALAR_51]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_51:%.+]] = vector.extract_strided_slice [[PACKED_51]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_51:%.+]] = vector.shape_cast [[OUT_SLICE_51]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_51]], [[ACC_A]] {offsets = [6, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_52:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_52:%.+]] = vector.shape_cast [[IN_SLICE_52]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_52:%.+]] = vector.extract [[SCALE_EXT]][6, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_52:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_52]][0], [[SCALE_SCALAR_52]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_52:%.+]] = vector.extract_strided_slice [[PACKED_52]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_52:%.+]] = vector.shape_cast [[OUT_SLICE_52]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_52]], [[ACC_B]] {offsets = [6, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_53:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_53:%.+]] = vector.shape_cast [[IN_SLICE_53]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_53:%.+]] = vector.extract [[SCALE_EXT]][6, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_53:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_53]][0], [[SCALE_SCALAR_53]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_53:%.+]] = vector.extract_strided_slice [[PACKED_53]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_53:%.+]] = vector.shape_cast [[OUT_SLICE_53]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_53]], [[ACC_A]] {offsets = [6, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_54:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_54:%.+]] = vector.shape_cast [[IN_SLICE_54]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_54:%.+]] = vector.extract [[SCALE_EXT]][6, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_54:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_54]][0], [[SCALE_SCALAR_54]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_54:%.+]] = vector.extract_strided_slice [[PACKED_54]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_54:%.+]] = vector.shape_cast [[OUT_SLICE_54]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_54]], [[ACC_B]] {offsets = [6, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_55:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [6, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_55:%.+]] = vector.shape_cast [[IN_SLICE_55]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_55:%.+]] = vector.extract [[SCALE_EXT]][6, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_55:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_55]][0], [[SCALE_SCALAR_55]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_55:%.+]] = vector.extract_strided_slice [[PACKED_55]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_55:%.+]] = vector.shape_cast [[OUT_SLICE_55]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_55]], [[ACC_A]] {offsets = [6, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_56:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_56:%.+]] = vector.shape_cast [[IN_SLICE_56]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_56:%.+]] = vector.extract [[SCALE_EXT]][7, 0, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_56:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_56]][0], [[SCALE_SCALAR_56]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_56:%.+]] = vector.extract_strided_slice [[PACKED_56]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_56:%.+]] = vector.shape_cast [[OUT_SLICE_56]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_56]], [[ACC_B]] {offsets = [7, 0, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_57:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_57:%.+]] = vector.shape_cast [[IN_SLICE_57]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_57:%.+]] = vector.extract [[SCALE_EXT]][7, 0, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_57:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_57]][0], [[SCALE_SCALAR_57]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_57:%.+]] = vector.extract_strided_slice [[PACKED_57]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_57:%.+]] = vector.shape_cast [[OUT_SLICE_57]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_57]], [[ACC_A]] {offsets = [7, 0, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_58:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_58:%.+]] = vector.shape_cast [[IN_SLICE_58]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_58:%.+]] = vector.extract [[SCALE_EXT]][7, 0, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_58:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_58]][0], [[SCALE_SCALAR_58]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_58:%.+]] = vector.extract_strided_slice [[PACKED_58]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_58:%.+]] = vector.shape_cast [[OUT_SLICE_58]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_58]], [[ACC_B]] {offsets = [7, 0, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_59:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_59:%.+]] = vector.shape_cast [[IN_SLICE_59]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_59:%.+]] = vector.extract [[SCALE_EXT]][7, 0, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_59:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_59]][0], [[SCALE_SCALAR_59]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_59:%.+]] = vector.extract_strided_slice [[PACKED_59]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_59:%.+]] = vector.shape_cast [[OUT_SLICE_59]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_59]], [[ACC_A]] {offsets = [7, 0, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_60:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 0], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_60:%.+]] = vector.shape_cast [[IN_SLICE_60]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_60:%.+]] = vector.extract [[SCALE_EXT]][7, 1, 0] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_60:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_60]][0], [[SCALE_SCALAR_60]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_60:%.+]] = vector.extract_strided_slice [[PACKED_60]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_60:%.+]] = vector.shape_cast [[OUT_SLICE_60]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_60]], [[ACC_B]] {offsets = [7, 1, 0], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_61:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_61:%.+]] = vector.shape_cast [[IN_SLICE_61]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_61:%.+]] = vector.extract [[SCALE_EXT]][7, 1, 1] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_61:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_61]][0], [[SCALE_SCALAR_61]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_61:%.+]] = vector.extract_strided_slice [[PACKED_61]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_61:%.+]] = vector.shape_cast [[OUT_SLICE_61]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_61]], [[ACC_A]] {offsets = [7, 1, 1], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_62:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 2], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_62:%.+]] = vector.shape_cast [[IN_SLICE_62]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_62:%.+]] = vector.extract [[SCALE_EXT]][7, 1, 2] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_62:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_62]][0], [[SCALE_SCALAR_62]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_62:%.+]] = vector.extract_strided_slice [[PACKED_62]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_62:%.+]] = vector.shape_cast [[OUT_SLICE_62]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_A:%.+]] = vector.insert_strided_slice [[OUT_VEC_62]], [[ACC_B]] {offsets = [7, 1, 2], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[IN_SLICE_63:%.+]] = vector.extract_strided_slice [[IN_CAST]] {offsets = [7, 1, 3], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<8x2x4xf8E5M2> to vector<1x1x1xf8E5M2>
// CHECK-NEXT:    [[IN_VEC_63:%.+]] = vector.shape_cast [[IN_SLICE_63]] : vector<1x1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    [[SCALE_SCALAR_63:%.+]] = vector.extract [[SCALE_EXT]][7, 1, 3] : f32 from vector<8x2x4xf32>
// CHECK-NEXT:    [[PACKED_63:%.+]] = amdgpu.scaled_ext_packed [[IN_VEC_63]][0], [[SCALE_SCALAR_63]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[OUT_SLICE_63:%.+]] = vector.extract_strided_slice [[PACKED_63]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    [[OUT_VEC_63:%.+]] = vector.shape_cast [[OUT_SLICE_63]] : vector<1xf32> to vector<1x1x1xf32>
// CHECK-NEXT:    [[ACC_B:%.+]] = vector.insert_strided_slice [[OUT_VEC_63]], [[ACC_A]] {offsets = [7, 1, 3], strides = [1, 1, 1]} : vector<1x1x1xf32> into vector<8x2x4xf32>
// CHECK-NEXT:    [[FINAL_CAST:%.+]] = vector.shape_cast [[ACC_B]] : vector<8x2x4xf32> to vector<8x8xf32>
// CHECK-NEXT:    return [[FINAL_CAST]] : vector<8x8xf32>
func.func @conversion_broadcast(%in: vector<8x8xf8E5M2>, %scale: vector<8x2xf8E8M0FNU>) -> vector<8x8xf32> {
    %bc = vector.broadcast %scale : vector<8x2xf8E8M0FNU> to vector<4x8x2xf8E8M0FNU>
    %cast1 = vector.shape_cast %in : vector<8x8xf8E5M2> to vector<8x2x4xf8E5M2>
    %cast2 = vector.shape_cast %bc : vector<4x8x2xf8E8M0FNU> to vector<8x2x4xf8E8M0FNU>
    %ext = arith.scaling_extf %cast1, %cast2 : vector<8x2x4xf8E5M2>, vector<8x2x4xf8E8M0FNU> to vector<8x2x4xf32>
    %cast3 = vector.shape_cast %ext : vector<8x2x4xf32> to vector<8x8xf32>
    return %cast3 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: @conversion_scalar
// CHECK:         [[SCALE_F32:%.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK-NEXT:    [[SPLAT_IN:%.+]] = vector.splat %arg0 : vector<1xf8E5M2>
// CHECK-NEXT:    [[PACKED_EXT:%.+]] = amdgpu.scaled_ext_packed [[SPLAT_IN]][0], [[SCALE_F32]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    [[RESULT:%.+]] = vector.extract [[PACKED_EXT]][0] : f32 from vector<2xf32>
// CHECK-NEXT:    return [[RESULT]] : f32
func.func @conversion_scalar(%in: f8E5M2, %scale: f8E8M0FNU) -> f32 {
    %ext = arith.scaling_extf %in, %scale : f8E5M2, f8E8M0FNU to f32
    return %ext : f32
}
