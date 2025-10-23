// RUN: mlir-opt %s --split-input-file -convert-xegpu-to-xevm | FileCheck %s

gpu.module @test {
// CHECK-LABEL: @load_gather_ui64_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: ui64
gpu.func @load_gather_ui64_src_constant_offset(%src: ui64) {
  // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[VAR2]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR4:.*]] = vector.extract %[[CST_0]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR5:.*]] = arith.muli %[[VAR3]], %[[C4_I64]] : i64
  // CHECK: %[[VAR6:.*]] = arith.addi %[[VAR1]], %[[VAR5]] : i64
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR7:.*]] = llvm.inttoptr %[[VAR6]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR8:.*]] = scf.if %[[VAR4]] -> (vector<2xf32>) {
  // CHECK:      %[[VAR9:.*]] = llvm.load %[[VAR7]] {cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}
  // CHECK-SAME:     : !llvm.ptr<1> -> vector<2xf32>
  // CHECK:      scf.yield %[[VAR9]] : vector<2xf32>
  // CHECK:    } else {
  // CHECK:      %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
  // CHECK:      scf.yield %[[CST_1]] : vector<2xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1> -> vector<2xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @load_gather_memref_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>
gpu.func @load_gather_memref_src_constant_offset(%src: memref<256xf32>) {
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR0:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST_0]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR3]], %[[VAR4]] : i64
  %2 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex>
      -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR7:.*]] = scf.if %[[VAR2]] -> (f32) {
  // CHECK:      %[[VAR8:.*]] = llvm.load %[[VAR6]] {cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}
  // CHECK-SAME:     : !llvm.ptr<1> -> vector<1xf32>
  // CHECK:      %[[VAR9:.*]] = vector.extract %[[VAR8]][0] : f32 from vector<1xf32>
  // CHECK:      scf.yield %[[VAR9]] : f32
  // CHECK:    } else {
  // CHECK:      %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  // CHECK:      %[[VAR8:.*]] = vector.extract %[[CST_1:.*]][0] : f32 from vector<1xf32>
  // CHECK:      scf.yield %[[VAR8]] : f32
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>, vector<1xi1> -> vector<1xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @load_gather_memref_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf16>, %[[ARG1:.*]]: vector<1xindex>
gpu.func @load_gather_memref_src_value_offset(%src: memref<256xf16>, %offset: vector<1xindex>) {
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf16> -> index
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[C2_I64:.*]] = arith.constant 2 : i64
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[C2_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR3]], %[[VAR4]] : i64
  %2 = xegpu.create_tdesc %src, %offset : memref<256xf16>, vector<1xindex>
      -> !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR7:.*]] = scf.if %[[VAR2]] -> (vector<8xf16>) {
  // CHECK:      %[[VAR8:.*]] = llvm.load %[[VAR6]] {cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}
  // CHECK-SAME:     : !llvm.ptr<1> -> vector<8xf16>
  // CHECK:      scf.yield %[[VAR8]] : vector<8xf16>
  // CHECK:    } else {
  // CHECK:      %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<8xf16>
  // CHECK:      scf.yield %[[CST_0]] : vector<8xf16>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<1xi1> -> vector<8xf16>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_ui64_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: ui64
gpu.func @store_scatter_ui64_src_constant_offset(%src: ui64) {
  // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[VAR2]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR4:.*]] = vector.extract %[[CST_0]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[CST_1:.*]] = arith.constant dense<2.900000e+00> : vector<2xf32>
  %2 = arith.constant dense<2.9>: vector<2xf32>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR5:.*]] = arith.muli %[[VAR3]], %[[C4_I64]] : i64
  // CHECK: %[[VAR6:.*]] = arith.addi %[[VAR1]], %[[VAR5]] : i64
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR7:.*]] = llvm.inttoptr %[[VAR6]] : i64 to !llvm.ptr<1>
  // CHECK:    scf.if %[[VAR4]] {
  // CHECK:      llvm.store %[[CST_1]], %[[VAR7]] {cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>}
  // CHECK-SAME:     : vector<2xf32>, !llvm.ptr<1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
      : vector<2xf32>, !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_memref_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>
gpu.func @store_scatter_memref_src_constant_offset(%src: memref<256xf32>) {
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR0:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST_0]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[CST_1:.*]] = arith.constant dense<2.900390e+00> : vector<2xf16>
  %2 = arith.constant dense<2.9>: vector<2xf16>
  // CHECK: %[[C2_I64:.*]] = arith.constant 2 : i64
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[C2_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR3]], %[[VAR4]] : i64
  %3 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: scf.if %[[VAR2]] {
  // CHECK:      llvm.store %[[CST_1]], %[[VAR6]] {cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>}
  // CHECK-SAME:     : vector<2xf16>, !llvm.ptr<1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
      : vector<2xf16>, !xegpu.tensor_desc<1x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_memref_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>, %[[ARG1:.*]]: vector<1xindex>
gpu.func @store_scatter_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>) {
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<1xi1>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST]][0] : i1 from vector<1xi1>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<2.900000e+00> : vector<1xf32>
  // CHECK: %[[VAR7:.*]] = vector.extract %[[CST_0]][0] : f32 from vector<1xf32>
  %2 = arith.constant dense<2.9>: vector<1xf32>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR3]], %[[VAR4]] : i64
  %3 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex>
      -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: scf.if %[[VAR2]] {
  // CHECK:      llvm.store %[[VAR7]], %[[VAR6]] {cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>}
  // CHECK-SAME:     : f32, !llvm.ptr<1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
      : vector<1xf32>, !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_ui64_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: ui64
gpu.func @prefetch_ui64_src_constant_offset(%src: ui64) {
  // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR2:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR3:.*]] = arith.index_castui %[[VAR2]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR3]], %[[C4_I64]] : i64
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR1]], %[[VAR4]] : i64
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR6:.*]] = llvm.inttoptr %[[VAR5]] : i64 to !llvm.ptr<1>
  // CHECK: xevm.prefetch %[[VAR6]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>
gpu.func @prefetch_memref_src_constant_offset(%src: memref<256xf32>) {
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[VAR0:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  %0 = arith.constant dense<0> : vector<1xindex>
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR4:.*]] = arith.addi %[[VAR2]], %[[VAR3]] : i64
  %1 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR5:.*]] = llvm.inttoptr %[[VAR4]] : i64 to !llvm.ptr<1>
  // CHECK: xevm.prefetch %[[VAR5]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_value_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>, %[[ARG1:.*]]: vector<1xindex>
gpu.func @prefetch_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>) {
  // CHECK: %[[VAR0:.*]] = vector.extract %[[ARG1]][0] : index from vector<1xindex>
  // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
  // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
  // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
  // CHECK: %[[VAR4:.*]] = arith.addi %[[VAR2]], %[[VAR3]] : i64
  %1 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex>
      -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR5:.*]] = llvm.inttoptr %[[VAR4]] : i64 to !llvm.ptr<1>
  // CHECK: xevm.prefetch %[[VAR5]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
