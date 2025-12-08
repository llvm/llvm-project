// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 --split-input-file --verify-diagnostics \
// RUN: | FileCheck %s

// CHECK-LABEL: @scaled_ext_packed_matrix_fp4
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf4E2M1FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed_matrix_fp4(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi4:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf4E2M1FN> to vector<8xi4>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.f16.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_i32:.+]] = llvm.bitcast %[[SOURCE_8xi4]] : vector<8xi4> to i32
  // CHECK: rocdl.cvt.scale.pk8.f32.fp4 %[[SOURCE_i32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf32>
  func.return %ret0, %ret1, %ret2: vector<8xf16>, vector<8xbf16>, vector<8xf32>
}

// CHECK-LABEL: @scaled_ext_packed_matrix_fp8
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf8E4M3FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed_matrix_fp8(%v: vector<8xf8E4M3FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi8:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf8E4M3FN> to vector<8xi8>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f16.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f32.fp8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xf32>

  func.return %ret0, %ret1, %ret2 : vector<8xf16>, vector<8xbf16>, vector<8xf32>
}

// CHECK-LABEL: @scaled_ext_packed_matrix_bf8
// CHECK-SAME: (%[[SOURCE:.+]]: vector<8xf8E5M2>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed_matrix_bf8(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) -> (vector<8xf16>, vector<8xbf16>, vector<8xf32>) {
  // CHECK: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK: %[[SOURCE_8xi8:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<8xf8E5M2> to vector<8xi8>
  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[RES:.+]] = rocdl.cvt.scale.pk8.f16.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf16>
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.bf16.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xbf16>
  %ret1 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v2xi32:.+]] = llvm.bitcast %[[SOURCE_8xi8]] : vector<8xi8> to vector<2xi32>
  // CHECK: rocdl.cvt.scale.pk8.f32.bf8 %[[SOURCE_v2xi32]], %[[SCALE_i32]][0] : vector<8xf32>
  %ret2 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf32>
  func.return %ret0, %ret1, %ret2 : vector<8xf16>, vector<8xbf16>, vector<8xf32>
}


// CHECK-LABEL: @scaled_ext_packed_matrix_fp6
// CHECK-SAME: (%[[SOURCE:.+]]: vector<16xf6E2M3FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed_matrix_fp6(%v: vector<16xf6E2M3FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>, vector<16xbf16>, vector<16xf32>) {
  // CHECK-DAG: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK-DAG: %[[SOURCE_16xi6:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<16xf6E2M3FN> to vector<16xi6>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f16.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf16>
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.bf16.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xbf16>
  %ret1 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f32.fp6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf32>
  %ret2 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xf32>
  return %ret0, %ret1, %ret2: vector<16xf16>, vector<16xbf16>, vector<16xf32>
}

// CHECK-LABEL: @scaled_ext_packed_matrix_bf6
// CHECK-SAME: (%[[SOURCE:.+]]: vector<16xf6E3M2FN>, %[[SCALE:.+]]: vector<4xf8E8M0FNU>)
func.func @scaled_ext_packed_matrix_bf6(%v: vector<16xf6E3M2FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>, vector<16xbf16>, vector<16xf32>) {
  // CHECK-DAG: %[[SCALE_4xi8:.+]] = builtin.unrealized_conversion_cast %[[SCALE]] : vector<4xf8E8M0FNU> to vector<4xi8>
  // CHECK-DAG: %[[SOURCE_16xi6:.+]] = builtin.unrealized_conversion_cast %[[SOURCE]] : vector<16xf6E3M2FN> to vector<16xi6>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f16.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf16>
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.bf16.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xbf16>
  %ret1 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xbf16>

  // CHECK: %[[SCALE_i32:.+]] = llvm.bitcast %[[SCALE_4xi8]] : vector<4xi8> to i32
  // CHECK: %[[SOURCE_v3xi32:.+]] = llvm.bitcast %[[SOURCE_16xi6]] : vector<16xi6> to vector<3xi32>
  // CHECK: rocdl.cvt.scale.pk16.f32.bf6 %[[SOURCE_v3xi32]], %[[SCALE_i32]][0] : vector<16xf32>
  %ret2 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf32>
  return %ret0, %ret1, %ret2: vector<16xf16>, vector<16xbf16>, vector<16xf32>
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_block_size_and_first_scale_byte_16(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op blockSize of 16 can only have firstScaleByte be 0 or 1 for f4 and f6}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(2) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_block_size_and_first_scale_byte_32(%v: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op blockSize of 32 can only have firstScaleByte be 0 or 2 for f4 and f6.}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(1) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_attributes_for_f8(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op blockSize of 16 can only have (firstScaleLane, firstScaleByte) be (0, 0) or (16, 2) for f8.}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(1) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_input_output_sizes(%v: vector<8xf8E5M2>, %scale: vector<4xf8E8M0FNU>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op failed to verify that all of {source, res} have same shape}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(16) firstScaleLane(0) firstScaleByte(0) : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<16xf16>
  func.return
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_src_elem_type(%v: vector<16xf16>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf16>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op operand #0 must be}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf16>, vector<4xf8E8M0FNU> -> vector<16xf16>
  return %ret0: vector<16xf16>
}

// -----

func.func @amdgpu.scaled_ext_packed_matrix_invalid_dst_elem_type(%v: vector<16xf6E3M2FN>, %scale: vector<4xf8E8M0FNU>) -> (vector<16xf64>) {
  // expected-error@+1 {{'amdgpu.scaled_ext_packed_matrix' op result #0 must be vector}}
  %ret0 = amdgpu.scaled_ext_packed_matrix %v scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf64>
  return %ret0: vector<16xf64>
}

// -----

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

func.func @amdgpu.make_dma_base.invalid_element_types(%idx: index, %mem: memref<8xi32, #gpu_global_addrspace>, %smem: memref<8xf32,#gpu_lds_addrspace>) -> (!amdgpu.tdm_base<i32>) {
  // expected-error@+1 {{'amdgpu.make_dma_base' op failed to verify that all of {global, lds} have same element type}}
  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu_global_addrspace>, memref<8xf32, #gpu_lds_addrspace> -> !amdgpu.tdm_base<i32>
  return %0 : !amdgpu.tdm_base<i32>
}

// -----

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

func.func @amdgpu.make_dma_base.invalid_element_types(%idx: index, %mem: memref<8xi7, #gpu_global_addrspace>, %smem: memref<8xi7,#gpu_lds_addrspace>) -> (!amdgpu.tdm_base<i7>) {
  // expected-error@+1 {{'amdgpu.make_dma_base' op element type must be 1, 2, 4, or 8 bytes long but type was 7 bits long.}}
  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi7, #gpu_global_addrspace>, memref<8xi7, #gpu_lds_addrspace> -> !amdgpu.tdm_base<i7>
  return %0 : !amdgpu.tdm_base<i7>
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

// -----

// CHECK-LABEL: func @make_dma_descriptor
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>)
func.func @make_dma_descriptor(%base: !amdgpu.tdm_base<i32>) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK-DAG: %[[DATA_SIZE:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0:.+]] = llvm.shl %[[DATA_SIZE]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR3_0:.+]] = llvm.lshr %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TENSOR_DIM_1_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK: %[[SGPR2:.+]] = llvm.or %[[SGPR2_0]], %[[TENSOR_DIM_1_SHIFTED]]

  // CHECK-DAG: %[[TILE_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0:.+]], %[[C16]]
  // CHECK: %[[SGPR3:.+]] = llvm.or %[[SGPR3_0]], %[[TILE_DIM_0_SHIFTED]]

  // CHECK-DAG: %[[SGPR4:.+]] = llvm.mlir.constant(128 : i32)

  // CHECK-DAG: %[[TENSOR_DIM_0_STRIDE:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_0_STRIDE]]
  // CHECK-DAG: %[[SGPR5:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_MASKED]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_HIGH_64:.+]] = llvm.lshr %[[TENSOR_DIM_0_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR6_0:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_HIGH_64]] : i64 to i32

  // CHECK-DAG: %[[TENSOR_DIM_1_STRIDE:.+]] = llvm.mlir.constant(64 : i64)
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_1_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_1_STRIDE]]
  // CHECK-DAG: %[[TENSOR_DIM_1_STRIDE_LOW:.+]] = llvm.trunc %[[TENSOR_DIM_1_STRIDE_MASKED]]
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK: %[[TENSOR_DIM_1_STRIDE_SHIFTED:.+]] = llvm.lshr %[[TENSOR_DIM_1_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR7:.+]] = llvm.trunc %[[TENSOR_DIM_1_STRIDE_SHIFTED]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1_STRIDE_LOW]], %[[SHIFT]]
  // CHECK-DAG: %[[SGPR6:.+]] = llvm.or %[[SGPR6_0]], %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED]]

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement %[[SGPR5]], %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement %[[SGPR6]], %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement %[[SGPR7]], %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]] : vector<4xi32>, vector<8xi32> to !amdgpu.tdm_descriptor
  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// CHECK-LABEL: func @make_dma_descriptor_atomic_barrier
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[BARRIER:.+]]: {{.*}}, %[[IDX:.+]]: index)
func.func @make_dma_descriptor_atomic_barrier(%base: !amdgpu.tdm_base<i32>, %barrier : memref<8xi32, #gpu_lds_addrspace>, %idx: index) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[INDEX:.+]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[BARRIER_MEMREF_DESC:.+]] = builtin.unrealized_conversion_cast %[[BARRIER]]
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK-DAG: %[[DATA_SIZE:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0_0:.+]] = llvm.shl %[[DATA_SIZE]], %[[C16]]

  // CHECK-DAG: %[[ATOMIC_BARRIER_ENABLE_OFFSET:.+]] = llvm.mlir.constant(18 : i32)
  // CHECK: %[[ATOMIC_BARRIER_ENABLE_FIELD:.+]] = llvm.shl %[[C1]], %[[ATOMIC_BARRIER_ENABLE_OFFSET]]
  // CHECK: %[[SGPR0:.+]] = llvm.or %[[SGPR0_0]], %[[ATOMIC_BARRIER_ENABLE_FIELD]]

  // CHECK: %[[ATOMIC_BARRIER_ALIGNED_PTR:.+]] = llvm.extractvalue %[[BARRIER_MEMREF_DESC]][1]
  // CHECK: %[[ATOMIC_BARRIER_ADDR:.+]] = llvm.getelementptr %[[ATOMIC_BARRIER_ALIGNED_PTR]][%[[INDEX]]
  // CHECK: %[[ATOMIC_BARRIER_I32:.+]] = llvm.ptrtoint %[[ATOMIC_BARRIER_ADDR]] : !llvm.ptr<3> to i32
  // CHECK: %[[ATOMIC_BARRIER_NO_3_LSB:.+]] = llvm.lshr %[[ATOMIC_BARRIER_I32]], %[[C3]]
  // CHECK: %[[MASK:.+]] = llvm.mlir.constant(65535 : i32)
  // CHECK: %[[ATOMIC_BARRIER:.+]] = llvm.and %[[ATOMIC_BARRIER_NO_3_LSB]], %[[MASK]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1_0:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]
  // CHECK: %[[SGPR1:.+]] = llvm.or %[[ATOMIC_BARRIER]], %[[SGPR1_0]]

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]

  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64]
                globalStride [64, 1]
                sharedSize [128, 64]
                atomicBarrier(%barrier[%idx] : memref<8xi32, #gpu_lds_addrspace>)
                : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: func @make_dma_descriptor_workgroup_mask
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[WG_MASK:.+]]: i16, %[[TIMEOUT:.+]]: i1)
func.func @make_dma_descriptor_workgroup_mask(%base: !amdgpu.tdm_base<i32>, %wg_mask: i16, %timeout: i1) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK-DAG: %[[WG_MASK_EXT:.+]] = llvm.zext %[[WG_MASK]]
  // CHECK-DAG: %[[DATA_SIZE:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[DATA_SIZE_SHIFTED:.+]] = llvm.shl %[[DATA_SIZE]], %[[C16]]
  // CHECK: %[[SGPR0_BASE:.+]] = llvm.or %[[WG_MASK_EXT]], %[[DATA_SIZE_SHIFTED]]
  // CHECK-DAG: %[[C21:.+]] = llvm.mlir.constant(21 : i32)
  // CHECK: %[[TIMEOUT_SHIFTED:.+]] = llvm.shl %[[C1]], %[[C21]]
  // CHECK: %[[SGPR0:.+]] = llvm.or %[[SGPR0_BASE]], %[[TIMEOUT_SHIFTED]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR3_0:.+]] = llvm.lshr %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TENSOR_DIM_1_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK: %[[SGPR2:.+]] = llvm.or %[[SGPR2_0]], %[[TENSOR_DIM_1_SHIFTED]]

  // CHECK-DAG: %[[TILE_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0:.+]], %[[C16]]
  // CHECK: %[[SGPR3:.+]] = llvm.or %[[SGPR3_0]], %[[TILE_DIM_0_SHIFTED]]

  // CHECK-DAG: %[[SGPR4:.+]] = llvm.mlir.constant(128 : i32)

  // CHECK-DAG: %[[TENSOR_DIM_0_STRIDE:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_0_STRIDE]]
  // CHECK-DAG: %[[SGPR5:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_MASKED]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_HIGH_64:.+]] = llvm.lshr %[[TENSOR_DIM_0_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR6_0:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_HIGH_64]] : i64 to i32

  // CHECK-DAG: %[[TENSOR_DIM_1_STRIDE:.+]] = llvm.mlir.constant(64 : i64)
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_1_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_1_STRIDE]]
  // CHECK-DAG: %[[TENSOR_DIM_1_STRIDE_LOW:.+]] = llvm.trunc %[[TENSOR_DIM_1_STRIDE_MASKED]]
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK: %[[TENSOR_DIM_1_STRIDE_SHIFTED:.+]] = llvm.lshr %[[TENSOR_DIM_1_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR7:.+]] = llvm.trunc %[[TENSOR_DIM_1_STRIDE_SHIFTED]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1_STRIDE_LOW]], %[[SHIFT]]
  // CHECK-DAG: %[[SGPR6:.+]] = llvm.or %[[SGPR6_0]], %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED]]

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement %[[SGPR5]], %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement %[[SGPR6]], %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement %[[SGPR7]], %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]] : vector<4xi32>, vector<8xi32> to !amdgpu.tdm_descriptor
  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] workgroupMask %wg_mask earlyTimeout %timeout : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}
