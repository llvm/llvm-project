// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx950" | FileCheck %s

// CHECK-LABEL: @conversion_f8_f32_fallback
// CHECK:         %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK-NEXT:    %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_00:.+]] = vector.shape_cast %[[IN_SLICE_00]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_00]][0], %[[SCALE_SCALAR_00]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_00]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_01:.+]] = vector.shape_cast %[[IN_SLICE_01]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_01]][0], %[[SCALE_SCALAR_01]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_01]], %[[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_10:.+]] = vector.shape_cast %[[IN_SLICE_10]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_10]][0], %[[SCALE_SCALAR_10]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_10]], %[[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_11:.+]] = vector.shape_cast %[[IN_SLICE_11]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_11]][0], %[[SCALE_SCALAR_11]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_11]], %[[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf32>
func.func @conversion_f8_f32_fallback(%in: vector<2x2xf8E5M2>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf32> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf8E5M2>, vector<2x2xf8E8M0FNU> to vector<2x2xf32>
    return %ext : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @conversion_f4_f32_fallback
// CHECK:         %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK-NEXT:    %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_00:.+]] = vector.shape_cast %[[IN_SLICE_00]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_00]][0], %[[SCALE_SCALAR_00]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_00]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_01:.+]] = vector.shape_cast %[[IN_SLICE_01]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_01]][0], %[[SCALE_SCALAR_01]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_01]], %[[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_10:.+]] = vector.shape_cast %[[IN_SLICE_10]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_10]][0], %[[SCALE_SCALAR_10]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_10]], %[[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_11:.+]] = vector.shape_cast %[[IN_SLICE_11]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_11]][0], %[[SCALE_SCALAR_11]] : vector<1xf4E2M1FN> to vector<2xf32>
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
// CHECK-NEXT:    %[[OUT_VEC_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]] : vector<1xf32> to vector<1x1xf32>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_11]], %[[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf32> into vector<2x2xf32>
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf32>
func.func @conversion_f4_f32_fallback(%in: vector<2x2xf4E2M1FN>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf32> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf4E2M1FN>, vector<2x2xf8E8M0FNU> to vector<2x2xf32>
    return %ext : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @conversion_f8_f16_fallback
// CHECK:         %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK-NEXT:    %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_00:.+]] = vector.shape_cast %[[IN_SLICE_00]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_00]][0], %[[SCALE_SCALAR_00]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_00]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_01:.+]] = vector.shape_cast %[[IN_SLICE_01]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_01]][0], %[[SCALE_SCALAR_01]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_01]], %[[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_10:.+]] = vector.shape_cast %[[IN_SLICE_10]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_10]][0], %[[SCALE_SCALAR_10]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_10]], %[[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf8E5M2> to vector<1x1xf8E5M2>
// CHECK-NEXT:    %[[IN_VEC_11:.+]] = vector.shape_cast %[[IN_SLICE_11]] : vector<1x1xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_11]][0], %[[SCALE_SCALAR_11]] : vector<1xf8E5M2> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_11]], %[[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf16>
func.func @conversion_f8_f16_fallback(%in: vector<2x2xf8E5M2>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf16> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf8E5M2>, vector<2x2xf8E8M0FNU> to vector<2x2xf16>
    return %ext : vector<2x2xf16>
}

// -----

// CHECK-LABEL: @conversion_f4_f16_fallback
// CHECK:         %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK-NEXT:    %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK-NEXT:    %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_00:.+]] = vector.shape_cast %[[IN_SLICE_00]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_00]][0], %[[SCALE_SCALAR_00]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_00]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_01:.+]] = vector.shape_cast %[[IN_SLICE_01]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_01]][0], %[[SCALE_SCALAR_01]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_01]], %[[ACC_A]] {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_10:.+]] = vector.shape_cast %[[IN_SLICE_10]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_10]][0], %[[SCALE_SCALAR_10]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_VEC_10]], %[[ACC_B]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]} : vector<2x2xf4E2M1FN> to vector<1x1xf4E2M1FN>
// CHECK-NEXT:    %[[IN_VEC_11:.+]] = vector.shape_cast %[[IN_SLICE_11]] : vector<1x1xf4E2M1FN> to vector<1xf4E2M1FN>
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1] : f32 from vector<2x2xf32>
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.scaled_ext_packed %[[IN_VEC_11]][0], %[[SCALE_SCALAR_11]] : vector<1xf4E2M1FN> to vector<2xf16>
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
// CHECK-NEXT:    %[[OUT_VEC_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]] : vector<1xf16> to vector<1x1xf16>
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_VEC_11]], %[[ACC_A]] {offsets = [1, 1], strides = [1, 1]} : vector<1x1xf16> into vector<2x2xf16>
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf16>
func.func @conversion_f4_f16_fallback(%in: vector<2x2xf4E2M1FN>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf16> {
    %ext = arith.scaling_extf %in, %scale : vector<2x2xf4E2M1FN>, vector<2x2xf8E8M0FNU> to vector<2x2xf16>
    return %ext : vector<2x2xf16>
}

// -----

// CHECK-LABEL: @conversion_broadcast
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<8x2x4xf32>
// CHECK-DAG:     %[[BCAST:.+]] = vector.broadcast %arg1
// CHECK-DAG:     %[[IN_CAST:.+]] = vector.shape_cast %arg0
// CHECK-DAG:     %[[SCALE_CAST:.+]] = vector.shape_cast %[[BCAST]]
// CHECK-DAG:     %[[SCALE_EXT:.+]] = arith.extf %[[SCALE_CAST]]
// CHECK-DAG:     vector.extract_strided_slice %[[IN_CAST]] {offsets = [0, 0, 0]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract %[[SCALE_EXT]][0, 0, 0]
// CHECK-NEXT:    amdgpu.scaled_ext_packed
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.insert_strided_slice
// CHECK-DAG:     vector.extract_strided_slice %[[IN_CAST]] {offsets = [0, 0, 1]
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract %[[SCALE_EXT]][0, 0, 1]
// CHECK-NEXT:    amdgpu.scaled_ext_packed
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.insert_strided_slice
// CHECK-DAG:     vector.shape_cast
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
// CHECK:         %[[SCALE_F32:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK-NEXT:    %[[SPLAT_IN:.+]] = vector.splat %arg0 : vector<1xf8E5M2>
// CHECK-NEXT:    %[[PACKED_EXT:.+]] = amdgpu.scaled_ext_packed %[[SPLAT_IN]][0], %[[SCALE_F32]] : vector<1xf8E5M2> to vector<2xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = vector.extract %[[PACKED_EXT]][0] : f32 from vector<2xf32>
// CHECK-NEXT:    return %[[RESULT]] : f32
func.func @conversion_scalar(%in: f8E5M2, %scale: f8E8M0FNU) -> f32 {
    %ext = arith.scaling_extf %in, %scale : f8E5M2, f8E8M0FNU to f32
    return %ext : f32
}
