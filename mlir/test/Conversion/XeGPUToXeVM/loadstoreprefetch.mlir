// RUN: mlir-opt %s --split-input-file -convert-xegpu-to-xevm -canonicalize | FileCheck %s

gpu.module @test {
// CHECK-LABEL: @load_gather_i64_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: vector<1xindex>, %[[ARG2:.*]]: memref<1xf16>
// CHECK-SAME: %[[ARG3:.*]]: vector<1xi1>
gpu.func @load_gather_i64_src_value_offset(%src: i64, %offset: vector<1xindex>, %dst: memref<1xf16>, %mask: vector<1xi1>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[C2_I64:.*]] = arith.constant 2 : i64
  // CHECK: %[[VAR2:.*]] = vector.extract %[[ARG3]][0] : i1 from vector<1xi1>
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[C2_I64]] : i64
  // CHECK: %[[VAR4:.*]] = arith.addi %[[ARG0]], %[[VAR3]] : i64
  // CHECK: %[[VAR5:.*]] = llvm.inttoptr %[[VAR4]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR6:.*]] = scf.if %[[VAR2]] -> (f16) {
  // CHECK:   %[[VAR7:.*]] = llvm.load %[[VAR5]] {cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>} : !llvm.ptr<1> -> f16
  // CHECK:   scf.yield %[[VAR7]] : f16
  // CHECK: } else {
  // CHECK:   scf.yield %[[CST_0]] : f16
  // CHECK: }
  %0 = xegpu.load %src[%offset], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<1xindex>, vector<1xi1> -> vector<1xf16>
  %c0 = arith.constant 0 : index
  vector.store %0, %dst[%c0] : memref<1xf16>, vector<1xf16>
  gpu.return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: @source_materialize_single_elem_vec
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: vector<1xindex>, %[[ARG2:.*]]: memref<1xf16>
// CHECK-SAME: %[[ARG3:.*]]: vector<1xi1>
gpu.func @source_materialize_single_elem_vec(%src: i64, %offset: vector<1xindex>, %dst: memref<1xf16>, %mask: vector<1xi1>) {
  %0 = xegpu.load %src[%offset], %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<1xindex>, vector<1xi1> -> vector<1xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAR_IF:.*]] = scf.if
  // CHECK: %[[VAR_RET:.*]] = vector.broadcast %[[VAR_IF]] : f16 to vector<1xf16>
  // CHECK: vector.store %[[VAR_RET]], %[[ARG2]][%[[C0]]] : memref<1xf16>, vector<1xf16>
  %c0 = arith.constant 0 : index
  vector.store %0, %dst[%c0] : memref<1xf16>, vector<1xf16>
  gpu.return
}
}

// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_i64_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: vector<1xindex>, %[[ARG2:.*]]: vector<1xi1>
gpu.func @store_scatter_i64_src_value_offset(%src: i64, %offset: vector<1xindex>, %mask: vector<1xi1>) {
  // CHECK: %[[CST_0:.*]] = arith.constant 2.900000e+00 : f32
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR2:.*]] = vector.extract %[[ARG2]][0] : i1 from vector<1xi1>
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  %0 = arith.constant dense<2.9>: vector<1xf32>
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[ARG0]], %[[VAR4]] : i64
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: scf.if %[[VAR2]] {
  // CHECK:   llvm.store %[[CST_0]], %[[VAR6]] {cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>} : f32, !llvm.ptr<1>
  // CHECK: }
  xegpu.store %0, %src[%offset], %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
      : vector<1xf32>, i64, vector<1xindex>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_i64_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: vector<1xindex>
gpu.func @prefetch_i64_src_value_offset(%src: i64, %offset: vector<1xindex>) {
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[VAR2:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR3:.*]] = arith.addi %[[ARG0]], %[[VAR2]] : i64
  // CHECK: %[[VAR4:.*]] = llvm.inttoptr %[[VAR3]] : i64 to !llvm.ptr<1>
  // CHECK: xevm.prefetch %[[VAR4]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xegpu.prefetch %src[%offset] <{offset_align_byte=4, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : i64, vector<1xindex>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>, %[[ARG1:.*]]: vector<1xindex>
gpu.func @prefetch_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>) {
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR4:.*]] = arith.addi %[[VAR2]], %[[VAR3]] : i64
  // CHECK: %[[VAR5:.*]] = llvm.inttoptr %[[VAR4]] : i64 to !llvm.ptr<1>
  // CHECK: xevm.prefetch %[[VAR5]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xegpu.prefetch %src[%offset] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : memref<256xf32>, vector<1xindex>
  gpu.return
}
}
