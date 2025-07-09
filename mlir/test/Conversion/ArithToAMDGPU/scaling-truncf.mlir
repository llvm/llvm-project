// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="chipset=gfx950" | FileCheck %s

// CHECK-LABEL: @conversion_f8_fallback
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf8E5M2>
// CHECK-DAG:     %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK:         %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_00:.+]] = vector.shape_cast %[[IN_SLICE_00]]
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0]
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_00]] into undef[0], %[[SCALE_SCALAR_00]]
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]]
// CHECK-NEXT:    %[[OUT_SCALAR_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]]
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_00]], %[[CST]]
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_01:.+]] = vector.shape_cast %[[IN_SLICE_01]]
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1]
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_01]] into undef[0], %[[SCALE_SCALAR_01]]
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]]
// CHECK-NEXT:    %[[OUT_SCALAR_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]]
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_01]], %[[ACC_A]]
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_10:.+]] = vector.shape_cast %[[IN_SLICE_10]]
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0]
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_10]] into undef[0], %[[SCALE_SCALAR_10]]
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]]
// CHECK-NEXT:    %[[OUT_SCALAR_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]]
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_10]], %[[ACC_B]]
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_11:.+]] = vector.shape_cast %[[IN_SLICE_11]]
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1]
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_11]] into undef[0], %[[SCALE_SCALAR_11]]
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]]
// CHECK-NEXT:    %[[OUT_SCALAR_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]]
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_11]], %[[ACC_A]]
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf8E5M2>
func.func @conversion_f8_fallback(%in: vector<2x2xf32>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf8E5M2> {
    %ext = arith.scaling_truncf %in, %scale : vector<2x2xf32>, vector<2x2xf8E8M0FNU> to vector<2x2xf8E5M2>
    return %ext : vector<2x2xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_f4_fallback
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf4E2M1FN>
// CHECK-DAG:     %[[SCALE_EXT:.+]] = arith.extf %arg1 : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
// CHECK:         %[[IN_SLICE_00:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_00:.+]] = vector.shape_cast %[[IN_SLICE_00]]
// CHECK-NEXT:    %[[SCALE_SCALAR_00:.+]] = vector.extract %[[SCALE_EXT]][0, 0]
// CHECK-NEXT:    %[[PACKED_00:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_00]] into undef[0], %[[SCALE_SCALAR_00]]
// CHECK-NEXT:    %[[OUT_SLICE_00:.+]] = vector.extract_strided_slice %[[PACKED_00]]
// CHECK-NEXT:    %[[OUT_SCALAR_00:.+]] = vector.shape_cast %[[OUT_SLICE_00]]
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_00]], %[[CST]]
// CHECK-NEXT:    %[[IN_SLICE_01:.+]] = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_01:.+]] = vector.shape_cast %[[IN_SLICE_01]]
// CHECK-NEXT:    %[[SCALE_SCALAR_01:.+]] = vector.extract %[[SCALE_EXT]][0, 1]
// CHECK-NEXT:    %[[PACKED_01:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_01]] into undef[0], %[[SCALE_SCALAR_01]]
// CHECK-NEXT:    %[[OUT_SLICE_01:.+]] = vector.extract_strided_slice %[[PACKED_01]]
// CHECK-NEXT:    %[[OUT_SCALAR_01:.+]] = vector.shape_cast %[[OUT_SLICE_01]]
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_01]], %[[ACC_A]]
// CHECK-NEXT:    %[[IN_SLICE_10:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_10:.+]] = vector.shape_cast %[[IN_SLICE_10]]
// CHECK-NEXT:    %[[SCALE_SCALAR_10:.+]] = vector.extract %[[SCALE_EXT]][1, 0]
// CHECK-NEXT:    %[[PACKED_10:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_10]] into undef[0], %[[SCALE_SCALAR_10]]
// CHECK-NEXT:    %[[OUT_SLICE_10:.+]] = vector.extract_strided_slice %[[PACKED_10]]
// CHECK-NEXT:    %[[OUT_SCALAR_10:.+]] = vector.shape_cast %[[OUT_SLICE_10]]
// CHECK-NEXT:    %[[ACC_A:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_10]], %[[ACC_B]]
// CHECK-NEXT:    %[[IN_SLICE_11:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 1], strides = [1, 1]}
// CHECK-NEXT:    %[[IN_SCALAR_11:.+]] = vector.shape_cast %[[IN_SLICE_11]]
// CHECK-NEXT:    %[[SCALE_SCALAR_11:.+]] = vector.extract %[[SCALE_EXT]][1, 1]
// CHECK-NEXT:    %[[PACKED_11:.+]] = amdgpu.packed_scaled_trunc %[[IN_SCALAR_11]] into undef[0], %[[SCALE_SCALAR_11]]
// CHECK-NEXT:    %[[OUT_SLICE_11:.+]] = vector.extract_strided_slice %[[PACKED_11]]
// CHECK-NEXT:    %[[OUT_SCALAR_11:.+]] = vector.shape_cast %[[OUT_SLICE_11]]
// CHECK-NEXT:    %[[ACC_B:.+]] = vector.insert_strided_slice %[[OUT_SCALAR_11]], %[[ACC_A]]
// CHECK-NEXT:    return %[[ACC_B]] : vector<2x2xf4E2M1FN>
func.func @conversion_f4_fallback(%in: vector<2x2xf32>, %scale: vector<2x2xf8E8M0FNU>) -> vector<2x2xf4E2M1FN> {
    %ext = arith.scaling_truncf %in, %scale : vector<2x2xf32>, vector<2x2xf8E8M0FNU> to vector<2x2xf4E2M1FN>
    return %ext : vector<2x2xf4E2M1FN>
}

// -----

// CHECK-LABEL: @conversion_broadcast
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<8x2x4xf8E5M2>
// CHECK-DAG:     %[[BCAST:.+]] = vector.broadcast %arg1
// CHECK-DAG:     %[[IN_CAST:.+]] = vector.shape_cast %arg0
// CHECK-DAG:     %[[SCALE_CAST:.+]] = vector.shape_cast %[[BCAST]]
// CHECK-DAG:     %[[SCALE_EXT:.+]] = arith.extf %[[SCALE_CAST]]
// CHECK-DAG:     vector.extract_strided_slice %[[IN_CAST]] {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]}
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract %[[SCALE_EXT]][0, 0, 0]
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [0], strides = [1]}
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [2], strides = [1]}
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}} {offsets = [0, 0, 0], strides = [1, 1, 1]}
// CHECK-NEXT:    vector.extract_strided_slice %[[IN_CAST]] {offsets = [0, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]}
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.extract %[[SCALE_EXT]][0, 1, 0]
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [0], strides = [1]}
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT:    amdgpu.packed_scaled_trunc
// CHECK-NEXT:    vector.extract_strided_slice %{{.+}} {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [2], strides = [1]}
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [0, 1, 0], strides = [1, 1, 1]} 
func.func @conversion_broadcast(%in: vector<8x8xf32>, %scale: vector<8x2xf8E8M0FNU>) -> vector<8x8xf8E5M2> {
    %bc = vector.broadcast %scale : vector<8x2xf8E8M0FNU> to vector<4x8x2xf8E8M0FNU>
    %cast1 = vector.shape_cast %in : vector<8x8xf32> to vector<8x2x4xf32>
    %cast2 = vector.shape_cast %bc : vector<4x8x2xf8E8M0FNU> to vector<8x2x4xf8E8M0FNU>
    %ext = arith.scaling_truncf %cast1, %cast2 : vector<8x2x4xf32>, vector<8x2x4xf8E8M0FNU> to vector<8x2x4xf8E5M2>
    %cast3 = vector.shape_cast %ext : vector<8x2x4xf8E5M2> to vector<8x8xf8E5M2>
    return %cast3 : vector<8x8xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_broadcast_odd
// CHECK-NEXT:    %[[CST3:.+]] = arith.constant dense<0.000000e+00> : vector<3xf8E5M2>
// CHECK-NEXT:    %[[CST6:.+]] = arith.constant dense<0.000000e+00> : vector<6xf8E5M2>
// CHECK-NEXT:    %[[SCALE_BCAST:.+]] = vector.broadcast %arg1 : vector<2xf8E8M0FNU> to vector<3x2xf8E8M0FNU>
// CHECK-NEXT:    %[[SCALE_FLAT:.+]] = vector.shape_cast %[[SCALE_BCAST]] : vector<3x2xf8E8M0FNU> to vector<6xf8E8M0FNU>
// CHECK-NEXT:    %[[SCALE_EXTF:.+]] = arith.extf %[[SCALE_FLAT]] : vector<6xf8E8M0FNU> to vector<6xf32>
// CHECK-NEXT:    %[[IN_CHUNK0:.+]] = vector.extract_strided_slice %arg0 {offsets = [0], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
// CHECK-NEXT:    %[[SCALE0:.+]] = vector.extract %[[SCALE_EXTF]][0] : f32 from vector<6xf32>
// CHECK-NEXT:    %[[IN_CHUNK0_PART0:.+]] = vector.extract_strided_slice %[[IN_CHUNK0]] {offsets = [0], sizes = [2], strides = [1]} : vector<3xf32> to vector<2xf32>
// CHECK-NEXT:    %[[PACKED0_PART0:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK0_PART0]] into undef[0], %[[SCALE0]] : vector<2xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK0_PART0:.+]] = vector.extract_strided_slice %[[PACKED0_PART0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2> to vector<2xf8E5M2>
// CHECK-NEXT:    %[[ACCUM0_PART0:.+]] = vector.insert_strided_slice %[[OUT_CHUNK0_PART0]], %[[CST3]] {offsets = [0], strides = [1]} : vector<2xf8E5M2> into vector<3xf8E5M2>
// CHECK-NEXT:    %[[IN_CHUNK0_PART1:.+]] = vector.extract_strided_slice %[[IN_CHUNK0]] {offsets = [2], sizes = [1], strides = [1]} : vector<3xf32> to vector<1xf32>
// CHECK-NEXT:    %[[PACKED0_PART1:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK0_PART1]] into undef[0], %[[SCALE0]] : vector<1xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK0_PART1:.+]] = vector.extract_strided_slice %[[PACKED0_PART1]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[CHUNK0_RES:.+]] = vector.insert_strided_slice %[[OUT_CHUNK0_PART1]], %[[ACCUM0_PART0]] {offsets = [2], strides = [1]} : vector<1xf8E5M2> into vector<3xf8E5M2>
// CHECK-NEXT:    %[[FINAL_ACCUM_A:.+]] = vector.insert_strided_slice %[[CHUNK0_RES]], %[[CST6]] {offsets = [0], strides = [1]} : vector<3xf8E5M2> into vector<6xf8E5M2>
// CHECK-NEXT:    %[[IN_CHUNK1:.+]] = vector.extract_strided_slice %arg0 {offsets = [3], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
// CHECK-NEXT:    %[[SCALE1:.+]] = vector.extract %[[SCALE_EXTF]][3] : f32 from vector<6xf32>
// CHECK-NEXT:    %[[IN_CHUNK1_PART0:.+]] = vector.extract_strided_slice %[[IN_CHUNK1]] {offsets = [0], sizes = [2], strides = [1]} : vector<3xf32> to vector<2xf32>
// CHECK-NEXT:    %[[PACKED1_PART0:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK1_PART0]] into undef[0], %[[SCALE1]] : vector<2xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK1_PART0:.+]] = vector.extract_strided_slice %[[PACKED1_PART0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2> to vector<2xf8E5M2>
// CHECK-NEXT:    %[[ACCUM1_PART0:.+]] = vector.insert_strided_slice %[[OUT_CHUNK1_PART0]], %[[CST3]] {offsets = [0], strides = [1]} : vector<2xf8E5M2> into vector<3xf8E5M2>
// CHECK-NEXT:    %[[IN_CHUNK1_PART1:.+]] = vector.extract_strided_slice %[[IN_CHUNK1]] {offsets = [2], sizes = [1], strides = [1]} : vector<3xf32> to vector<1xf32>
// CHECK-NEXT:    %[[PACKED1_PART1:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK1_PART1]] into undef[0], %[[SCALE1]] : vector<1xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK1_PART1:.+]] = vector.extract_strided_slice %[[PACKED1_PART1]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf8E5M2> to vector<1xf8E5M2>
// CHECK-NEXT:    %[[CHUNK1_RES:.+]] = vector.insert_strided_slice %[[OUT_CHUNK1_PART1]], %[[ACCUM1_PART0]] {offsets = [2], strides = [1]} : vector<1xf8E5M2> into vector<3xf8E5M2>
// CHECK-NEXT:    %[[FINAL_RESULT:.+]] = vector.insert_strided_slice %[[CHUNK1_RES]], %[[FINAL_ACCUM_A]] {offsets = [3], strides = [1]} : vector<3xf8E5M2> into vector<6xf8E5M2>
// CHECK-NEXT:    return %[[FINAL_RESULT]] : vector<6xf8E5M2>
func.func @conversion_broadcast_odd(%in: vector<6xf32>, %scale: vector<2xf8E8M0FNU>) -> vector<6xf8E5M2> {
    %bc = vector.broadcast %scale : vector<2xf8E8M0FNU> to vector<3x2xf8E8M0FNU>
    %cast = vector.shape_cast %bc : vector<3x2xf8E8M0FNU> to vector<6xf8E8M0FNU>
    %ext = arith.scaling_truncf %in, %cast : vector<6xf32>, vector<6xf8E8M0FNU> to vector<6xf8E5M2>
    return %ext : vector<6xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_broadcast
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf8E5M2>
// CHECK-DAG:     %[[SCALE_SPLAT:.+]] = vector.broadcast %arg1 : f8E8M0FNU to vector<4xf8E8M0FNU>
// CHECK-DAG:     %[[SCALE_EXTF:.+]] = arith.extf %[[SCALE_SPLAT]] : vector<4xf8E8M0FNU> to vector<4xf32>
// CHECK-DAG:     %[[SCALE_SCALAR:.+]] = vector.extract %[[SCALE_EXTF]][0] : f32 from vector<4xf32>
// CHECK:         %[[IN_CHUNK0:.+]] = vector.extract_strided_slice %arg0 {offsets = [0], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
// CHECK-NEXT:    %[[PACKED0:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK0]] into undef[0], %[[SCALE_SCALAR]] : vector<2xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK0:.+]] = vector.extract_strided_slice %[[PACKED0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2> to vector<2xf8E5M2>
// CHECK-NEXT:    %[[ACCUM_A:.+]] = vector.insert_strided_slice %[[OUT_CHUNK0]], %[[CST]] {offsets = [0], strides = [1]} : vector<2xf8E5M2> into vector<4xf8E5M2>
// CHECK-NEXT:    %[[IN_CHUNK1:.+]] = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
// CHECK-NEXT:    %[[PACKED1:.+]] = amdgpu.packed_scaled_trunc %[[IN_CHUNK1]] into undef[0], %[[SCALE_SCALAR]] : vector<2xf32> to vector<4xf8E5M2>
// CHECK-NEXT:    %[[OUT_CHUNK1:.+]] = vector.extract_strided_slice %[[PACKED1]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf8E5M2> to vector<2xf8E5M2>
// CHECK-NEXT:    %[[FINAL_RESULT:.+]] = vector.insert_strided_slice %[[OUT_CHUNK1]], %[[ACCUM_A]] {offsets = [2], strides = [1]} : vector<2xf8E5M2> into vector<4xf8E5M2>
// CHECK-NEXT:    return %[[FINAL_RESULT]] : vector<4xf8E5M2>
func.func @conversion_broadcast(%in: vector<4xf32>, %scale: f8E8M0FNU) -> vector<4xf8E5M2> {
    %splat = vector.broadcast %scale : f8E8M0FNU to vector<4xf8E8M0FNU>
    %ext = arith.scaling_truncf %in, %splat : vector<4xf32>, vector<4xf8E8M0FNU> to vector<4xf8E5M2>
    return %ext : vector<4xf8E5M2>
}

// -----

// CHECK-LABEL: @conversion_scalar
// CHECK:         %[[SCALE_F32:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK-NEXT:    %[[SPLAT_IN:.+]] = vector.broadcast %arg0 : f32 to vector<1xf32>
// CHECK-NEXT:    %[[PACKED_TRUNC:.+]] = amdgpu.packed_scaled_trunc %[[SPLAT_IN]] into undef[0], %[[SCALE_F32]]
// CHECK-NEXT:    %[[RESULT:.+]] = vector.extract %[[PACKED_TRUNC]][0]
// CHECK-NEXT:    return %[[RESULT]] : f8E5M2
func.func @conversion_scalar(%in: f32, %scale: f8E8M0FNU) -> f8E5M2 {
    %ext = arith.scaling_truncf %in, %scale : f32, f8E8M0FNU to f8E5M2
    return %ext : f8E5M2
}
