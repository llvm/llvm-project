// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 --split-input-file --verify-diagnostics \
// RUN: | FileCheck %s

// CHECK-LABEL: @scaled_ext_packed816_fp4
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf4E2M1FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed816_fp4(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi4:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf4E2M1FN> to vector<8xi4>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.f16.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.f32.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf32>
  func.return %ret0, %ret1, %ret2: vector<8xf16>, vector<8xbf16>, vector<8xf32>
}

// CHECK-LABEL: @scaled_ext_packed816_fp8
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf8E4M3FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed816_fp8(%v: vector<8xf8E4M3FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi8:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf8E4M3FN> to vector<8xi8>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f16.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f32.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xf32>

  func.return %ret0, %ret1, %ret2 : vector<8xf16>, vector<8xbf16>, vector<8xf32>
}

// CHECK-LABEL: @scaled_ext_packed816_bf8
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf8E5M2>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed816_bf8(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi8:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf8E5M2> to vector<8xi8>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[RES:.+]] = rocdl.cvt.scale.pk8.f16.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.bf16.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f32.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf32>
  func.return %ret0, %ret1, %ret2 : vector<8xf16>, vector<8xbf16>, vector<8xf32>
}


// CHECK-LABEL: @scaled_ext_packed816_fp6
// CHECK-SAME: (%[[SOURCE:.+]]: vector<16xf6E2M3FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed816_fp6(%v: vector<16xf6E2M3FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>, vector<16xbf16>, vector<16xf32>) {
  // CHECK-DAG: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK-DAG: %[[SOURCE_16xi6:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<16xf6E2M3FN> to vector<16xi6>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f16.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf16>
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.bf16.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xbf16>
  %ret1 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f32.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf32>
  %ret2 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xf32>
  return %ret0, %ret1, %ret2: vector<16xf16>, vector<16xbf16>, vector<16xf32>
}

// CHECK-LABEL: @scaled_ext_packed816_bf6
// CHECK-SAME: (%[[SOURCE:.+]]: vector<16xf6E3M2FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed816_bf6(%v: vector<16xf6E3M2FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>, vector<16xbf16>, vector<16xf32>) {
  // CHECK-DAG: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK-DAG: %[[SOURCE_16xi6:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<16xf6E3M2FN> to vector<16xi6>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f16.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf16>
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.bf16.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xbf16>
  %ret1 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f32.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf32>
  %ret2 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf32>
  return %ret0, %ret1, %ret2: vector<16xf16>, vector<16xbf16>, vector<16xf32>
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_block_size_and_first_scale_byte_16(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op blockSize of 16 can only have firstScaleByte be 0 or 1 for f4 and f6}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(2) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_block_size_and_first_scale_byte_32(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op blockSize of 32 can only have firstScaleByte be 0 or 2 for f4 and f6.}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(1) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_attributes_for_f8(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op blockSize of 16 can only have (firstScaleLane, firstScaleByte) be (0, 0) or (1, 2) for f8.}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(1) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_input_output_sizes(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op failed to verify that all of {source, res} have same shape}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<16xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_src_elem_type(%v: vector<16xf16>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op operand #0 must be}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf16>, vector<4xf8E8M0FNU> -> vector<16xf16>
  return %ret0: vector<16xf16>
}

// -----

func.func @amdgpu.scaled_ext_packed816_invalid_dst_elem_type(%v: vector<16xf6E3M2FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf64>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed816' op result #0 must be vector}}
  %ret0 = amdgpu.scaled_ext_packed816 %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf64>
  return %ret0: vector<16xf64>
}
