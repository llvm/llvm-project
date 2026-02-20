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

func.func @amdgpu.make_dma_base.invalid_element_types(%idx: index, %mem: memref<8xi32, #gpu.address_space<global>>, %smem: memref<8xf32,#gpu.address_space<workgroup>>) -> (!amdgpu.tdm_base<i32>) {
  // expected-error@+1 {{'amdgpu.make_dma_base' op failed to verify that all of {global, lds} have same element type}}
  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu.address_space<global>>, memref<8xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i32>
  return %0 : !amdgpu.tdm_base<i32>
}

// -----

func.func @amdgpu.make_dma_base.invalid_element_types(%idx: index, %mem: memref<8xi7, #gpu.address_space<global>>, %smem: memref<8xi7,#gpu.address_space<workgroup>>) -> (!amdgpu.tdm_base<i7>) {
  // expected-error@+1 {{'amdgpu.make_dma_base' op element type must be 1, 2, 4, or 8 bytes long but type was 7 bits long.}}
  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi7, #gpu.address_space<global>>, memref<8xi7, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i7>
  return %0 : !amdgpu.tdm_base<i7>
}

// -----

// CHECK-LABEL: func @make_dma_base
// CHECK-SAME: (%[[IDX:.+]]: index, %[[MEM:.+]]: memref<8xi32, #gpu.address_space<global>>, %[[SMEM:.+]]: memref<8xi32, #gpu.address_space<workgroup>>)
func.func @make_dma_base(%idx: index, %mem: memref<8xi32, #gpu.address_space<global>>, %smem: memref<8xi32,#gpu.address_space<workgroup>>) -> (!amdgpu.tdm_base<i32>) {
  // CHECK-DAG: %[[INT:.+]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[MEMREF_DESC_MEM:.+]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<8xi32, #gpu.address_space<global>>
  // CHECK-DAG: %[[MEMREF_DESC_SMEM:.+]] = builtin.unrealized_conversion_cast %[[SMEM]] : memref<8xi32, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32

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

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(30 : i32)
  // CHECK: %[[TYPE_FIELD:.+]] = llvm.shl %[[C2]], %[[SHIFT]]
  // CHECK: %[[MEM_INT_HIGH_TYPE:.+]] = llvm.or disjoint %[[VALID_MEM_INT_HIGH]], %[[TYPE_FIELD]]

  // CHECK: %[[V4I32_0_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[V4I32_0_1:.+]] = llvm.insertelement %[[C1]], %[[V4I32_0_0]][%[[C0]] : i32]
  // CHECK: %[[V4I32_0_2:.+]] = llvm.insertelement %[[SMEM_INT]], %[[V4I32_0_1]][%[[C1]] : i32]
  // CHECK: %[[V4I32_0_3:.+]] = llvm.insertelement %[[MEM_INT_LOW]], %[[V4I32_0_2]][%[[C2]] : i32]
  // CHECK: %[[V4I32_0_4:.+]] = llvm.insertelement %[[MEM_INT_HIGH_TYPE]], %[[V4I32_0_3]][%[[C3]] : i32]

  %0 = amdgpu.make_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu.address_space<global>>, memref<8xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i32>

  func.return %0 : !amdgpu.tdm_base<i32>
}

// -----

// CHECK-LABEL: func @make_gather_dma_base
// CHECK-SAME: (%[[IDX:.+]]: index, %[[MEM:.+]]: memref<8xi32, #gpu.address_space<global>>, %[[SMEM:.+]]: memref<8xi32, #gpu.address_space<workgroup>>)
func.func @make_gather_dma_base(%idx: index, %mem: memref<8xi32, #gpu.address_space<global>>, %smem: memref<8xi32,#gpu.address_space<workgroup>>) -> (!amdgpu.tdm_gather_base<i32, i16>, !amdgpu.tdm_gather_base<i32, i32>) {

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32

  // CHECK-DAG: %[[GATHER_MODE_OFFSET:.+]] = llvm.mlir.constant(30 : i32) : i32
  // CHECK-DAG: %[[GATHER_MODE_BIT:.+]] = llvm.shl %[[C1]], %[[GATHER_MODE_OFFSET]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[C1]], %[[GATHER_MODE_BIT]]

  // CHECK: %[[V4I32_0_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[V4I32_0_1:.+]] = llvm.insertelement %[[SGPR0]], %[[V4I32_0_0]][%[[C0]] : i32]

  %0 = amdgpu.make_gather_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu.address_space<global>>, memref<8xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<i32, i16>

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32

  // CHECK-DAG: %[[GATHER_MODE_OFFSET:.+]] = llvm.mlir.constant(30 : i32) : i32
  // CHECK-DAG: %[[GATHER_MODE_BIT:.+]] = llvm.shl %[[C1]], %[[GATHER_MODE_OFFSET]]
  // CHECK: %[[SGPR0_0:.+]] = llvm.or disjoint %[[C1]], %[[GATHER_MODE_BIT]]

  // CHECK-DAG: %[[INDEX_SIZE_OFFSET:.+]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK-DAG: %[[INDEX_SIZE_BIT:.+]] = llvm.shl %[[C1]], %[[INDEX_SIZE_OFFSET]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_0]], %[[INDEX_SIZE_BIT]]

  // CHECK: %[[V4I32_0_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[V4I32_0_1:.+]] = llvm.insertelement %[[SGPR0]], %[[V4I32_0_0]][%[[C0]] : i32]


  %1 = amdgpu.make_gather_dma_base %mem[%idx], %smem[%idx] : memref<8xi32, #gpu.address_space<global>>, memref<8xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<i32, i32>

  func.return %0, %1 : !amdgpu.tdm_gather_base<i32,i16>, !amdgpu.tdm_gather_base<i32,i32>
}

// -----

// This test exercises the lowering for operations that only require 2-descriptors
// to be fully described.

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

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0:.+]] = llvm.shl %[[C2]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TENSOR_DIM_1_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[SGPR2_0]], %[[TENSOR_DIM_1_SHIFTED]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR3_0:.+]] = llvm.lshr %[[TENSOR_DIM_1]], %[[C16]]

  // CHECK-DAG: %[[TILE_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0:.+]], %[[C16]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[SGPR3_0]], %[[TILE_DIM_0_SHIFTED]]

  // CHECK-DAG: %[[SGPR4:.+]] = llvm.mlir.constant(128 : i32)

  // CHECK-DAG: %[[TENSOR_DIM_0_STRIDE:.+]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_0_STRIDE]]
  // CHECK-DAG: %[[SGPR5:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_MASKED]] : i64 to i32

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_HIGH_64:.+]] = llvm.lshr %[[TENSOR_DIM_0_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR6:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_HIGH_64]] : i64 to i32

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement %[[SGPR5]], %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement %[[SGPR6]], %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement %[[C0]], %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK: %[[DGROUP2:.+]] = llvm.mlir.zero : vector<4xi32>
  // CHECK: %[[DGROUP3:.+]] = llvm.mlir.zero : vector<4xi32>

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]], %[[DGROUP2]], %[[DGROUP3]] : vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> to !amdgpu.tdm_descriptor
  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: func @make_dma_descriptor_atomic_barrier
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[BARRIER:.+]]: {{.*}}, %[[IDX:.+]]: index)
func.func @make_dma_descriptor_atomic_barrier(%base: !amdgpu.tdm_base<i32>, %barrier : memref<2x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %idx: index) -> !amdgpu.tdm_descriptor {
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

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0_0:.+]] = llvm.shl %[[C2]], %[[C16]]

  // CHECK-DAG: %[[ATOMIC_BARRIER_ENABLE_OFFSET:.+]] = llvm.mlir.constant(18 : i32)
  // CHECK: %[[ATOMIC_BARRIER_ENABLE_FIELD:.+]] = llvm.shl %[[C1]], %[[ATOMIC_BARRIER_ENABLE_OFFSET]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_0]], %[[ATOMIC_BARRIER_ENABLE_FIELD]]

  // CHECK: %[[ATOMIC_BARRIER_ALIGNED_PTR:.+]] = llvm.extractvalue %[[BARRIER_MEMREF_DESC]][1]
  // CHECK: %[[ATOMIC_BARRIER_ADDR:.+]] = llvm.getelementptr %[[ATOMIC_BARRIER_ALIGNED_PTR]][%[[INDEX]]
  // CHECK: %[[ATOMIC_BARRIER_I32:.+]] = llvm.ptrtoint %[[ATOMIC_BARRIER_ADDR]] : !llvm.ptr<3> to i32
  // CHECK: %[[ATOMIC_BARRIER_NO_3_LSB:.+]] = llvm.lshr %[[ATOMIC_BARRIER_I32]], %[[C3]]
  // CHECK: %[[MASK:.+]] = llvm.mlir.constant(65535 : i32)
  // CHECK: %[[ATOMIC_BARRIER:.+]] = llvm.and %[[ATOMIC_BARRIER_NO_3_LSB]], %[[MASK]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1_0:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]
  // CHECK: %[[SGPR1:.+]] = llvm.or disjoint %[[ATOMIC_BARRIER]], %[[SGPR1_0]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]

  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64]
                globalStride [64, 1]
                sharedSize [128, 64]
                atomicBarrier(%barrier[%idx] : memref<2x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>)
                : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}


// -----

// CHECK-LABEL: func @make_dma_descriptor_iterate
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[IDX:.+]]: index, %[[I32:.+]]: i32)
func.func @make_dma_descriptor_iterate(%base: !amdgpu.tdm_base<i32>, %idx : index, %i32: i32) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]
  // CHECK-DAG: %[[INDEX:.+]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64

  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK: %[[C3:.+]] = llvm.mlir.constant(3 : i32)

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0_0:.+]] = llvm.shl %[[C2]], %[[C16]]

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(19 : i32)
  // CHECK: %[[ITERATE_ENABLE:.+]] = llvm.shl %[[C1]], %[[SHIFT]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_0]], %[[ITERATE_ENABLE]]

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]

  // CHECK: %[[SGPR2:.+]] = llvm.trunc %[[INDEX]] : i64 to i32

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[GLOBAL_ADDR_INC_HIGH:.+]] = llvm.lshr %[[INDEX]], %[[SHIFT]]
  // CHECK: %[[GLOBAL_ADDR_INC_HIGH_2:.+]] = llvm.trunc %[[GLOBAL_ADDR_INC_HIGH]] : i64 to i32
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant([[FIRST_16_HIGH:65535]] : i32) : i32
  // CHECK: %[[SGPR3_LOW:.+]] = llvm.and %[[GLOBAL_ADDR_INC_HIGH_2]], %[[MASK]]

  // CHECK: %[[ITERATE_COUNT:.+]] = llvm.trunc %[[INDEX]] : i64 to i32
  // CHECK: %[[ITERATE_COUNT_M1:.+]] = llvm.sub %[[ITERATE_COUNT]], %[[C1]]
  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[ITERATE_COUNT_SHIFTED:.+]] = llvm.shl %[[ITERATE_COUNT_M1]], %[[SHIFT]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[SGPR3_LOW]], %[[ITERATE_COUNT_SHIFTED]]

  // CHECK: %[[V4I32:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[DGROUP2_0:.+]] = llvm.insertelement %[[C0]], %[[V4I32]][%[[C0]]
  // CHECK: %[[DGROUP2_1:.+]] = llvm.insertelement %[[I32]], %[[DGROUP2_0]][%[[C1]]
  // CHECK: %[[DGROUP2_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP2_1]][%[[C2]]
  // CHECK: %[[DGROUP2:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP2_2]][%[[C3]]

  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] iterate %idx, %i32, %idx : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: func @make_dma_descriptor_pad_enable
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[PAD_AMOUNT:.+]]: i32, %[[PAD_INTERVAL:.+]]: i32)
func.func @make_dma_descriptor_pad_enable(%base: !amdgpu.tdm_base<i32>, %pad_amount: i32, %pad_interval: i32) -> !amdgpu.tdm_descriptor {

  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(20 : i32)
  // CHECK: %[[PAD_ENABLE:.+]] = llvm.shl %[[C1]], %[[SHIFT]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_BASE:.+]], %[[PAD_ENABLE]]

  // CHECK: %[[PAD_INTERVAL_CTTZ:.+]] = "llvm.intr.cttz"(%[[PAD_INTERVAL]]) <{is_zero_poison = false}> : (i32) -> i32
  // CHECK: %[[PAD_INTERVAL_M1:.+]] = llvm.sub %[[PAD_INTERVAL_CTTZ]], %[[C1]]
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(22 : i32)
  // CHECK: %[[PAD_INTERVAL:.+]] = llvm.shl %[[PAD_INTERVAL_M1]], %[[SHIFT]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_BASE:.+]], %[[PAD_INTERVAL]]

  // CHECK: %[[PAD_AMOUNT_M1:.+]] = llvm.sub %[[PAD_AMOUNT]], %[[C1]]
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(25 : i32)
  // CHECK: %[[PAD_AMOUNT_SHIFTED:.+]] = llvm.shl %[[PAD_AMOUNT_M1]], %[[SHIFT]]
  // CHECK: llvm.or disjoint %[[SGPR0:.+]], %[[PAD_AMOUNT_SHIFTED]]

  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] padShared(%pad_amount every %pad_interval) : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: func @make_dma_descriptor_dynamic
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[GS0:.+]]: index, %[[GS1:.+]]: index, %[[GST1:.+]]: index, %[[SHS0:.+]]: index, %[[SHS1:.+]]: index)
func.func @make_dma_descriptor_dynamic(%base: !amdgpu.tdm_base<i32>, %gs0: index, %gs1: index, %gst1: index, %shs0: index, %shs1: index) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]
  // CHECK-DAG: %[[GS0I:.+]] = builtin.unrealized_conversion_cast %[[GS0]]
  // CHECK-DAG: %[[GS1I:.+]] = builtin.unrealized_conversion_cast %[[GS1]]
  // CHECK-DAG: %[[GST1I:.+]] = builtin.unrealized_conversion_cast %[[GST1]]
  // CHECK-DAG: %[[SHS0I:.+]] = builtin.unrealized_conversion_cast %[[SHS0]]
  // CHECK-DAG: %[[SHS1I:.+]] = builtin.unrealized_conversion_cast %[[SHS1]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK: %[[TENSOR_DIM_0:.+]] = llvm.trunc %[[GS0I]] : i64 to i32
  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[SHIFT]]

  // CHECK: %[[TENSOR_DIM_1:.+]] = llvm.trunc %[[GS1I]] : i64 to i32
  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[SHIFT]]

  // CHECK: %[[TILE_DIM_0:.+]] = llvm.trunc %[[SHS0I]] : i64 to i32
  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0:.+]], %[[SHIFT]]

  // CHECK: %[[TILE_DIM_1:.+]] = llvm.trunc %[[SHS1I]] : i64 to i32

  // CHECK: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TILE_DIM_0_STRIDE:.+]] = llvm.and %[[MASK]], %[[GST1I]]
  // CHECK: %[[TILE_DIM_0_STRIDE_TRUNC:.+]] = llvm.trunc %[[TILE_DIM_0_STRIDE]]

  %descriptor = amdgpu.make_dma_descriptor %base globalSize [%gs1, %gs0] globalStride [%gst1, 1] sharedSize [%shs1, %shs0] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
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


  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR0:.+]] = llvm.shl %[[C2]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TENSOR_DIM_1_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[SGPR2_0]], %[[TENSOR_DIM_1_SHIFTED]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR3_0:.+]] = llvm.lshr %[[TENSOR_DIM_1]], %[[C16]]

  // CHECK-DAG: %[[TILE_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0]], %[[C16]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[SGPR3_0]], %[[TILE_DIM_0_SHIFTED]]

  // CHECK-DAG: %[[TILE_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[TILE_DIM_2:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_2_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_2]], %[[C16]]
  // CHECK: %[[SGPR4:.+]] = llvm.or disjoint %[[TILE_DIM_1]], %[[TILE_DIM_2_SHIFTED]]

  // CHECK-DAG: %[[TENSOR_DIM_0_STRIDE:.+]] = llvm.mlir.constant(64 : i64) : i64
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
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1_STRIDE_LOW]], %[[SHIFT]]
  // CHECK-DAG: %[[SGPR6:.+]] = llvm.or disjoint %[[SGPR6_0]], %[[TENSOR_DIM_1_STRIDE_LOW_SHIFTED]]

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK: %[[TENSOR_DIM_1_STRIDE_SHIFTED:.+]] = llvm.lshr %[[TENSOR_DIM_1_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR7:.+]] = llvm.trunc %[[TENSOR_DIM_1_STRIDE_SHIFTED]] : i64 to i32

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement %[[SGPR5]], %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement %[[SGPR6]], %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement %[[SGPR7]], %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK-DAG: %[[SGPR0:.+]] = llvm.mlir.constant([[TENSOR_DIM_2:64]] : i32)

  // CHECK-DAG: %[[SGPR1:.+]] = llvm.mlir.constant([[TENSOR_DIM_3:64]] : i32)

  // CHECK-DAG: %[[TENSOR_DIM_1_STRIDE:.+]] = llvm.mlir.constant(64 : i64)
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_2_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_1_STRIDE]]
  // CHECK-DAG: %[[SGPR2:.+]] = llvm.trunc %[[TENSOR_DIM_2_STRIDE_MASKED]]

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[TENSOR_DIM_2_STRIDE_HIGH_64:.+]] = llvm.lshr %[[TENSOR_DIM_2_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR3_0:.+]] = llvm.trunc %[[TENSOR_DIM_2_STRIDE_HIGH_64]] : i64 to i32

  // CHECK-DAG: %[[TILE_DIM_3:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TILE_DIM_3_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_3]], %[[SHIFT]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[SGPR3_0]], %[[TILE_DIM_3_SHIFTED]]

  // CHECK-DAG: %[[V4I32:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[DGROUP2_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V4I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP2_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP2_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP2_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP2_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP2:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP2_2]][%[[C3]] : i32]

  // CHECK-DAG: %[[TENSOR_DIM3_STRIDE:.+]] = llvm.mlir.constant(64 : i64)
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM3_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM3_STRIDE]]
  // CHECK: %[[TENSOR_DIM3_STRIDE_LOW:.+]] = llvm.trunc %[[TENSOR_DIM3_STRIDE_MASKED]] : i64 to i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64)
  // CHECK: %[[TENSOR_DIM3_STRIDE_SHIFTED:.+]] = llvm.lshr %[[TENSOR_DIM3_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[TENSOR_DIM3_STRIDE_HIGH:.+]] = llvm.trunc %[[TENSOR_DIM3_STRIDE_SHIFTED]] : i64 to i32

  // CHECK-DAG: %[[TENSOR_DIM_4:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK-DAG: %[[TENSOR_DIM_4_LOW:.+]] = llvm.shl %[[TENSOR_DIM_4]], %[[SHIFT]]
  // CHECK: %[[SGPR1:.+]] = llvm.or disjoint %[[TENSOR_DIM3_STRIDE_HIGH]], %[[TENSOR_DIM_4_LOW]]

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_4]], %[[SHIFT]]

  // CHECK-DAG: %[[TILE_DIM_4:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TILE_DIM_4_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_4]], %[[SHIFT]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[SGPR2_0]], %[[TILE_DIM_4_SHIFTED]]

  // CHECK: %[[V4I32:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[DGROUP3_0:.+]] = llvm.insertelement %[[TENSOR_DIM3_STRIDE_LOW]], %[[V4I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP3_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP3_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP3_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP3_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP3:.+]] = llvm.insertelement %[[C0]], %[[DGROUP3_2]][%[[C3]] : i32]

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]], %[[DGROUP2]], %[[DGROUP3]] : vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> to !amdgpu.tdm_descriptor
  %descriptor = amdgpu.make_dma_descriptor %base globalSize [64, 64, 64, 128, 64] globalStride [64, 64, 64, 64, 1] sharedSize [64, 64, 64, 128, 64] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: func @make_dma_descriptor_workgroup_mask
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[WG_MASK:.+]]: vector<16xi1>, %[[TIMEOUT:.+]]: i1)
func.func @make_dma_descriptor_workgroup_mask(%base: !amdgpu.tdm_base<i32>, %wg_mask: vector<16xi1>, %timeout: i1) -> !amdgpu.tdm_descriptor {
  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK: %[[WG_MASK_CAST:.+]] = llvm.bitcast %[[WG_MASK]] : vector<16xi1> to i16
  // CHECK-DAG: %[[WG_MASK_EXT:.+]] = llvm.zext %[[WG_MASK_CAST]]
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[DATA_SIZE_SHIFTED:.+]] = llvm.shl %[[C2]], %[[C16]]
  // CHECK: %[[SGPR0_BASE:.+]] = llvm.or disjoint %[[WG_MASK_EXT]], %[[DATA_SIZE_SHIFTED]]

  // CHECK-DAG: %[[C21:.+]] = llvm.mlir.constant(21 : i32)
  // CHECK: %[[TIMEOUT_SHIFTED:.+]] = llvm.shl %[[C1]], %[[C21]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[SGPR0_BASE]], %[[TIMEOUT_SHIFTED]]

  // CHECK-DAG: %[[TENSOR_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR1:.+]] = llvm.shl %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR2_0:.+]] = llvm.lshr %[[TENSOR_DIM_0]], %[[C16]]

  // CHECK-DAG: %[[TENSOR_DIM_1:.+]] = llvm.mlir.constant(128 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TENSOR_DIM_1_SHIFTED:.+]] = llvm.shl %[[TENSOR_DIM_1]], %[[C16]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[SGPR2_0]], %[[TENSOR_DIM_1_SHIFTED]]

  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[SGPR3_0:.+]] = llvm.lshr %[[TENSOR_DIM_1]], %[[C16]]

  // CHECK-DAG: %[[TILE_DIM_0:.+]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK: %[[TILE_DIM_0_SHIFTED:.+]] = llvm.shl %[[TILE_DIM_0:.+]], %[[C16]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[SGPR3_0]], %[[TILE_DIM_0_SHIFTED]]

  // CHECK-DAG: %[[SGPR4:.+]] = llvm.mlir.constant(128 : i32)

  // CHECK-DAG: %[[TENSOR_DIM_0_STRIDE:.+]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-DAG: %[[MASK:.+]] = llvm.mlir.constant(281474976710655 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_MASKED:.+]] = llvm.and %[[MASK]], %[[TENSOR_DIM_0_STRIDE]]
  // CHECK-DAG: %[[SGPR5:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_MASKED]] : i64 to i32

  // CHECK-DAG: %[[SHIFT:.+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[TENSOR_DIM_0_STRIDE_HIGH_64:.+]] = llvm.lshr %[[TENSOR_DIM_0_STRIDE_MASKED]], %[[SHIFT]]
  // CHECK: %[[SGPR6:.+]] = llvm.trunc %[[TENSOR_DIM_0_STRIDE_HIGH_64]] : i64 to i32

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement %[[SGPR0]], %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement %[[SGPR5]], %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement %[[SGPR6]], %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement %[[C0]], %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK: %[[DGROUP2:.+]] = llvm.mlir.zero : vector<4xi32>
  // CHECK: %[[DGROUP3:.+]] = llvm.mlir.zero : vector<4xi32>

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]], %[[DGROUP2]], %[[DGROUP3]] : vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> to !amdgpu.tdm_descriptor
  %descriptor = amdgpu.make_dma_descriptor %base globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] workgroupMask %wg_mask earlyTimeout %timeout : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

// CHECK-LABEL: func @tensor_load_to_lds
// CHECK-SAME: (%[[DESC:.+]]: !amdgpu.tdm_descriptor)
func.func @tensor_load_to_lds(%desc: !amdgpu.tdm_descriptor) {
  // CHECK: %[[DGROUPS:.+]]:4 = builtin.unrealized_conversion_cast %[[DESC]]
  // CHECK: %[[DGROUP4:.*]] = llvm.mlir.zero : vector<8xi32>
  // CHECK: rocdl.tensor.load.to.lds %[[DGROUPS]]#0, %[[DGROUPS]]#1, %[[DGROUPS]]#2, %[[DGROUPS]]#3 %[[DGROUP4]] cachepolicy 0 : vector<4xi32>, vector<8xi32>
  amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor
  func.return
}

// CHECK-LABEL: func @tensor_store_from_lds
// CHECK-SAME: (%[[DESC:.+]]: !amdgpu.tdm_descriptor)
func.func @tensor_store_from_lds(%desc: !amdgpu.tdm_descriptor) {
  // CHECK: %[[DGROUPS:.+]]:4 = builtin.unrealized_conversion_cast %[[DESC]]
  // CHECK: %[[DGROUP4:.*]] = llvm.mlir.zero : vector<8xi32>
  // CHECK: rocdl.tensor.store.from.lds %[[DGROUPS]]#0, %[[DGROUPS]]#1, %[[DGROUPS]]#2, %[[DGROUPS]]#3, %[[DGROUP4]] cachepolicy 0 : vector<4xi32>, vector<8xi32>
  amdgpu.tensor_store_from_lds %desc : !amdgpu.tdm_descriptor
  func.return
}

// CHECK-LABEL: func @make_gather_dma_descriptor
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_gather_base<i32, i16>, %[[INDICES:.+]]: vector<13xi16>)
func.func @make_gather_dma_descriptor(%base: !amdgpu.tdm_gather_base<i32, i16>, %indices: vector<13xi16>) -> !amdgpu.tdm_descriptor {

  // CHECK-DAG: %[[DGROUP0:.+]] = builtin.unrealized_conversion_cast %[[BASE]]

  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32)
  // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32)
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32)

  // CHECK: %[[SGPR4:.+]] = llvm.mlir.constant([[VALID_INDICES:13]] : i32) : i32

  // CHECK: %[[V8I32:.+]] = llvm.mlir.poison : vector<8xi32>
  // CHECK: %[[DGROUP1_0:.+]] = llvm.insertelement {{.*}}, %[[V8I32]][%[[C0]] : i32]
  // CHECK: %[[DGROUP1_1:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_0]][%[[C1]] : i32]
  // CHECK: %[[DGROUP1_2:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_1]][%[[C2]] : i32]
  // CHECK: %[[DGROUP1_3:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_2]][%[[C3]] : i32]
  // CHECK: %[[DGROUP1_4:.+]] = llvm.insertelement %[[SGPR4]], %[[DGROUP1_3]][%[[C4]] : i32]
  // CHECK: %[[DGROUP1_5:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_4]][%[[C5]] : i32]
  // CHECK: %[[DGROUP1_6:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_5]][%[[C6]] : i32]
  // CHECK: %[[DGROUP1:.+]] = llvm.insertelement {{.*}}, %[[DGROUP1_6]][%[[C7]] : i32]

  // CHECK-DAG: %[[IDX0:.+]] = llvm.extractelement %[[INDICES]][%[[C0]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX1:.+]] = llvm.extractelement %[[INDICES]][%[[C1]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX2:.+]] = llvm.extractelement %[[INDICES]][%[[C2]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX3:.+]] = llvm.extractelement %[[INDICES]][%[[C3]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX4:.+]] = llvm.extractelement %[[INDICES]][%[[C4]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX5:.+]] = llvm.extractelement %[[INDICES]][%[[C5]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX6:.+]] = llvm.extractelement %[[INDICES]][%[[C6]] : i32] : vector<13xi16>
  // CHECK-DAG: %[[IDX7:.+]] = llvm.extractelement %[[INDICES]][%[[C7]] : i32] : vector<13xi16>

  // CHECK: %[[IDX0_32:.+]] = llvm.zext %[[IDX0]] : i16 to i32
  // CHECK: %[[IDX1_32:.+]] = llvm.zext %[[IDX1]] : i16 to i32
  // CHECK: %[[IDX2_32:.+]] = llvm.zext %[[IDX2]] : i16 to i32
  // CHECK: %[[IDX3_32:.+]] = llvm.zext %[[IDX3]] : i16 to i32
  // CHECK: %[[IDX4_32:.+]] = llvm.zext %[[IDX4]] : i16 to i32
  // CHECK: %[[IDX5_32:.+]] = llvm.zext %[[IDX5]] : i16 to i32
  // CHECK: %[[IDX6_32:.+]] = llvm.zext %[[IDX6]] : i16 to i32
  // CHECK: %[[IDX7_32:.+]] = llvm.zext %[[IDX7]] : i16 to i32

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX1:.+]] = llvm.shl %[[IDX1_32]], %[[SHIFT]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[IDX0_32]], %[[IDX1]]

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX3:.+]] = llvm.shl %[[IDX3_32]], %[[SHIFT]]
  // CHECK: %[[SGPR1:.+]] = llvm.or disjoint %[[IDX2_32]], %[[IDX3]]

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX5:.+]] = llvm.shl %[[IDX5_32]], %[[SHIFT]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[IDX4_32]], %[[IDX5]]

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX7:.+]] = llvm.shl %[[IDX7_32]], %[[SHIFT]]
  // CHECK: %[[SGPR3:.+]] = llvm.or disjoint %[[IDX6_32]], %[[IDX7]]

  // CHECK: %[[DGROUP2_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[DGROUP2_1:.+]] = llvm.insertelement %[[SGPR0]], %[[DGROUP2_0]][%[[C0]] : i32]
  // CHECK: %[[DGROUP2_2:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP2_1]][%[[C1]] : i32]
  // CHECK: %[[DGROUP2_3:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP2_2]][%[[C2]] : i32]
  // CHECK: %[[DGROUP2:.+]] = llvm.insertelement %[[SGPR3]], %[[DGROUP2_3]][%[[C3]] : i32]

  // CHECK: %[[C8:.+]] = llvm.mlir.constant(8 : i32)
  // CHECK: %[[IDX8:.+]] = llvm.extractelement %[[INDICES]][%[[C8]] : i32] : vector<13xi16>
  // CHECK: %[[C9:.+]] = llvm.mlir.constant(9 : i32)
  // CHECK: %[[IDX9:.+]] = llvm.extractelement %[[INDICES]][%[[C9]] : i32] : vector<13xi16>
  // CHECK: %[[C10:.+]] = llvm.mlir.constant(10 : i32)
  // CHECK: %[[IDX10:.+]] = llvm.extractelement %[[INDICES]][%[[C10]] : i32] : vector<13xi16>
  // CHECK: %[[C11:.+]] = llvm.mlir.constant(11 : i32)
  // CHECK: %[[IDX11:.+]] = llvm.extractelement %[[INDICES]][%[[C11]] : i32] : vector<13xi16>
  // CHECK: %[[C12:.+]] = llvm.mlir.constant(12 : i32)
  // CHECK: %[[IDX12:.+]] = llvm.extractelement %[[INDICES]][%[[C12]] : i32] : vector<13xi16>

  // CHECK: %[[IDX8_32:.+]] = llvm.zext %[[IDX8]] : i16 to i32
  // CHECK: %[[IDX9_32:.+]] = llvm.zext %[[IDX9]] : i16 to i32
  // CHECK: %[[IDX10_32:.+]] = llvm.zext %[[IDX10]] : i16 to i32
  // CHECK: %[[IDX11_32:.+]] = llvm.zext %[[IDX11]] : i16 to i32
  // CHECK: %[[IDX12_32:.+]] = llvm.zext %[[IDX12]] : i16 to i32

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX9:.+]] = llvm.shl %[[IDX9_32]], %[[SHIFT]]
  // CHECK: %[[SGPR0:.+]] = llvm.or disjoint %[[IDX8_32]], %[[IDX9]]

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX11:.+]] = llvm.shl %[[IDX11_32]], %[[SHIFT]]
  // CHECK: %[[SGPR1:.+]] = llvm.or disjoint %[[IDX10_32]], %[[IDX11]]

  // CHECK: %[[SHIFT:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[IDX13:.+]] = llvm.shl %[[C0]], %[[SHIFT]]
  // CHECK: %[[SGPR2:.+]] = llvm.or disjoint %[[IDX12_32]], %[[IDX13]]

  // CHECK-DAG: %[[DGROUP3_0:.+]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: %[[DGROUP3_1:.+]] = llvm.insertelement %[[SGPR0]], %[[DGROUP3_0]][%[[C0]] : i32]
  // CHECK: %[[DGROUP3_2:.+]] = llvm.insertelement %[[SGPR1]], %[[DGROUP3_1]][%[[C1]] : i32]
  // CHECK: %[[DGROUP3:.+]] = llvm.insertelement %[[SGPR2]], %[[DGROUP3_2]][%[[C2]] : i32]

  // CHECK: %[[DGROUPS:.+]] = builtin.unrealized_conversion_cast %[[DGROUP0]], %[[DGROUP1]], %[[DGROUP2]], %[[DGROUP3]]
  %descriptor = amdgpu.make_gather_dma_descriptor %base[%indices] globalSize [128, 64] globalStride [64, 1] sharedSize [128, 64] : !amdgpu.tdm_gather_base<i32, i16>, vector<13xi16> -> !amdgpu.tdm_descriptor
  func.return %descriptor : !amdgpu.tdm_descriptor
}

/// LDS barriers

// CHECK-LABEL: func @ds_barrier_init
func.func @ds_barrier_init(%barrier: memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %participants: i32) {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[PTR:%.*]] = llvm.extractvalue [[CAST]][1]
  // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: [[SUB:%.*]] = llvm.sub %arg1, [[C1]]
  // CHECK: [[MASK:%.*]] = llvm.mlir.constant(536870911 : i32)
  // CHECK: [[MASKED:%.*]] = llvm.and [[SUB]], [[MASK]]
  // CHECK: [[ZEXT:%.*]] = llvm.zext [[MASKED]] : i32 to i64
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i64)
  // CHECK: [[INIT_SHIFT:%.*]] = llvm.shl [[ZEXT]], [[C32]]
  // CHECK: [[STATE:%.*]] = llvm.or [[INIT_SHIFT]], [[ZEXT]]
  // CHECK: llvm.store [[STATE]], [[PTR]] atomic syncscope("workgroup") release
  amdgpu.ds_barrier_init %barrier[], %participants : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i32
  func.return
}

// CHECK-LABEL: func @ds_barrier_poll_state
func.func @ds_barrier_poll_state(%barrier: memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>) -> !amdgpu.ds_barrier_state {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[PTR:%.*]] = llvm.extractvalue [[CAST]][1]
  // CHECK: [[LOADED:%.*]] = llvm.load [[PTR]] atomic syncscope("workgroup") acquire
  // CHECK: builtin.unrealized_conversion_cast [[LOADED]]
  %state = amdgpu.ds_barrier_poll_state %barrier[] : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>> -> !amdgpu.ds_barrier_state
  func.return %state : !amdgpu.ds_barrier_state
}

// CHECK-LABEL: func @ds_async_barrier_arrive
func.func @ds_async_barrier_arrive(%barrier: memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>) {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[PTR:%.*]] = llvm.extractvalue [[CAST]][1]
  // CHECK: rocdl.ds.atomic.async.barrier.arrive.b64 [[PTR]] : !llvm.ptr<3>
  amdgpu.ds_async_barrier_arrive %barrier[] : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @ds_barrier_arrive
func.func @ds_barrier_arrive(%barrier: memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %count: i64) -> !amdgpu.ds_barrier_state {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[PTR:%.*]] = llvm.extractvalue [[CAST]][1]
  // CHECK: [[OLD:%.*]] = rocdl.ds.atomic.barrier.arrive.rtn.b64 [[PTR]], %arg1 : !llvm.ptr<3>, i64 -> i64
  // CHECK: builtin.unrealized_conversion_cast [[OLD]]
  %old_state = amdgpu.ds_barrier_arrive %barrier[], %count : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i64 -> !amdgpu.ds_barrier_state
  func.return %old_state : !amdgpu.ds_barrier_state
}

// CHECK-LABEL: func @ds_barrier_state_phase
func.func @ds_barrier_state_phase(%state: !amdgpu.ds_barrier_state) -> i32 {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[TRUNC:%.*]] = llvm.trunc [[CAST]] : i64 to i32
  // CHECK: [[C29:%.*]] = llvm.mlir.constant(29 : i32)
  // CHECK: [[PHASE:%.*]] = llvm.lshr [[TRUNC]], [[C29]]
  // CHECK: return [[PHASE]]
  %phase = amdgpu.ds_barrier_state_phase %state : !amdgpu.ds_barrier_state -> i32
  func.return %phase : i32
}

// CHECK-LABEL: func @ds_barrier_state_pending_count
func.func @ds_barrier_state_pending_count(%state: !amdgpu.ds_barrier_state) -> i32 {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[TRUNC:%.*]] = llvm.trunc [[CAST]] : i64 to i32
  // CHECK: [[MASK:%.*]] = llvm.mlir.constant(536870911 : i32)
  // CHECK: [[COUNT:%.*]] = llvm.and [[TRUNC]], [[MASK]]
  // CHECK: return [[COUNT]]
  %pending = amdgpu.ds_barrier_state_pending_count %state : !amdgpu.ds_barrier_state -> i32
  func.return %pending : i32
}

// CHECK-LABEL: func @ds_barrier_state_init_count
func.func @ds_barrier_state_init_count(%state: !amdgpu.ds_barrier_state) -> i32 {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i64)
  // CHECK: [[SHIFTED:%.*]] = llvm.lshr [[CAST]], [[C32]]
  // CHECK: [[COUNT:%.*]] = llvm.trunc [[SHIFTED]] : i64 to i32
  // CHECK: return [[COUNT]]
  %init = amdgpu.ds_barrier_state_init_count %state : !amdgpu.ds_barrier_state -> i32
  func.return %init : i32
}

// CHECK-LABEL: func @ds_barrier_state_phase_parity
func.func @ds_barrier_state_phase_parity(%state: !amdgpu.ds_barrier_state) -> i1 {
  // CHECK: [[CAST:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: [[TRUNC:%.*]] = llvm.trunc [[CAST]] : i64 to i32
  // CHECK: [[C29:%.*]] = llvm.mlir.constant(29 : i32)
  // CHECK: [[SHIFTED:%.*]] = llvm.lshr [[TRUNC]], [[C29]]
  // CHECK: [[PARITY:%.*]] = llvm.trunc [[SHIFTED]] : i32 to i1
  // CHECK: return [[PARITY]]
  %parity = amdgpu.ds_barrier_state_phase_parity %state : !amdgpu.ds_barrier_state -> i1
  func.return %parity : i1
}
