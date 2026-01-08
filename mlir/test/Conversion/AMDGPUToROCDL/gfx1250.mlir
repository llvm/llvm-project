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

// -----

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// CHECK-LABEL: func @make_dma_base
// CHECK-SAME: (%[[IDX:.+]]: index, %[[MEM:.+]]: memref<8xi32, 1>, %[[SMEM:.+]]: memref<8xi32, 3>)
func.func @make_dma_base(%idx: index, %mem: memref<8xi32, #gpu_global_addrspace>, %smem: memref<8xi32,#gpu_lds_addrspace>) -> (!amdgpu.tdm_base<i32>) {
  // CHECK-DAG: %[[INT:.+]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[MEMREF_DESC_MEM:.+]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<8xi32, 1>
  // CHECK-DAG: %[[MEMREF_DESC_SMEM:.+]] = builtin.unrealized_conversion_cast %[[SMEM]] : memref<8xi32, 3>

  // CHECK-DAG: %[[MEM_BASE_PTR:.+]] = llvm.extractvalue %[[MEMREF_DESC_MEM]][1] : !llvm.struct<(ptr<1>
  // CHECK-DAG: %[[SMEM_BASE_PTR:.+]] = llvm.extractvalue %[[MEMREF_DESC_SMEM]][1] : !llvm.struct<(ptr<3>

  // CHECK-DAG: %[[MEM_BASE_OFFSET:.+]] = llvm.getelementptr %[[MEM_BASE_PTR]][%[[INT]]]
  // CHECK-DAG: %[[SMEM_BASE_OFFSET:.+]] = llvm.getelementptr %[[SMEM_BASE_PTR]][%[[INT]]]

  // CHECK-DAG: %[[MEM_INT:.+]] = llvm.ptrtoint %[[MEM_BASE_OFFSET]] : !llvm.ptr<1> to i64
  // CHECK-DAG: %[[SMEM_INT:.+]] = llvm.ptrtoint %[[SMEM_BASE_OFFSET]] : !llvm.ptr<3> to i32

  // CHECK: %[[MEM_INT_LOW:.+]] = llvm.trunc %[[MEM_INT]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64)
  // CHECK: %[[SHIFTED_MEM_INT:.+]] = llvm.lshr %[[MEM_INT]], %[[SHIFT]]
  // CHECK: %[[MEM_INT_HIGH:.+]] = llvm.trunc %[[SHIFTED_MEM_INT]] : i64 to i32
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(33554431 : i32)
  // CHECK: %[[VALID_MEM_INT_HIGH:.+]] = llvm.and %[[MEM_INT_HIGH]], %[[MASK]]

  // CHECK-DAG: %[[TYPE_FIELD:.+]] = llvm.mlir.constant(-2147483648 : i32)
  // CHECK: %[[MEM_INT_HIGH_TYPE:.+]] = llvm.or %[[VALID_MEM_INT_HIGH]], %[[TYPE_FIELD]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32

  // CHECK: %[[V4I32_0_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[V4I32_0_1:.+]] = llvm.insertelement %[[C1]], %[[V4I32_0_0]][%[[C0]] : i32]
  // CHECK: %[[V4I32_0_2:.+]] = llvm.insertelement %[[SMEM_INT]], %[[V4I32_0_1]][%[[C1]] : i32]
  // CHECK: %[[V4I32_0_3:.+]] = llvm.insertelement %[[MEM_INT_LOW]], %[[V4I32_0_2]][%[[C2]] : i32]
  // CHECK: %[[V4I32_0_4:.+]] = llvm.insertelement %[[MEM_INT_HIGH_TYPE]], %[[V4I32_0_3]][%[[C3]] : i32]

  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu_global_addrspace>, memref<8xi32, #gpu_lds_addrspace> -> !amdgpu.tdm_base<i32>

  func.return %0 : !amdgpu.tdm_base<i32>
}
