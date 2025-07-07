// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm --split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
// CHECK: llvm.func @blockload2d(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32) -> vector<8xi16> {
llvm.func @blockload2d(%a: !llvm.ptr<1>, %base_width_a: i32, %base_height_a: i32, %base_pitch_a: i32, %x: i32, %y: i32) -> vector<8xi16> {
  // CHECK: %[[VAR0:.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[VAR1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VAR2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VAR3:.*]] = llvm.insertelement %[[ARG4]], %[[VAR0]][%[[VAR1]] : i32] : vector<2xi32>
  // CHECK: %[[VAR4:.*]] = llvm.insertelement %[[ARG5]], %[[VAR3]][%[[VAR2]] : i32] : vector<2xi32>
  // CHECK: %[[VAR5:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[VAR6:.*]] = llvm.alloca %[[VAR5]] x i16 : (i32) -> !llvm.ptr
  // CHECK: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[VAR4]], %[[VAR6]]) {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>, ptr)>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt", visibility_ = 0 : i64, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
  // CHECK: %[[VAR7:.*]] = llvm.load %[[VAR6]] : !llvm.ptr -> vector<8xi16>
  %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32, transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return %loaded_a : vector<8xi16>
}

// -----
// CHECK: llvm.func spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {no_unwind, will_return}
// CHECK: llvm.func @blockstore2d(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: vector<8xi32>) {
llvm.func @blockstore2d(%c: !llvm.ptr<1>, %base_width_c: i32, %base_height_c: i32, %base_pitch_c: i32, %x: i32, %y: i32, %c_result_casted: vector<8xi32>) {
  // CHECK: %[[VAR0:.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[VAR1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VAR2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VAR3:.*]] = llvm.insertelement %[[ARG4]], %[[VAR0]][%[[VAR1]] : i32] : vector<2xi32>
  // CHECK: %[[VAR4:.*]] = llvm.insertelement %[[ARG5]], %[[VAR3]][%[[VAR2]] : i32] : vector<2xi32>
  // CHECK: %[[VAR5:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[VAR6:.*]] = llvm.alloca %[[VAR5]] x i32 : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG6]], %[[VAR6]] : vector<8xi32>, !llvm.ptr
  // CHECK: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[VAR4]], %[[VAR6]]) {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>, ptr)>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj", visibility_ = 0 : i64, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) -> ()
  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  llvm.return
}

// -----
// CHECK: llvm.func spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
// CHECK: llvm.func @blockprefetch2d(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32) {
llvm.func @blockprefetch2d(%ptr: !llvm.ptr<1>, %base_width: i32, %base_height: i32, %base_pitch: i32, %x: i32, %y: i32) {
  // CHECK: %[[VAR0:.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[VAR1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VAR2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VAR3:.*]] = llvm.insertelement %[[ARG4]], %[[VAR0]][%[[VAR1]] : i32] : vector<2xi32>
  // CHECK: %[[VAR4:.*]] = llvm.insertelement %[[ARG5]], %[[VAR3]][%[[VAR2]] : i32] : vector<2xi32>
  // CHECK: llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[VAR4]]) {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>)>, linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, sym_name = "_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i", visibility_ = 0 : i64
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y <{elem_size_in_bits=8 : i32, tile_width=32 : i32, tile_height=8 : i32, v_blocks=1 : i32, cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----
// CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
// CHECK: llvm.func @mma(%[[ARG0:.*]]: vector<8xf32>, %[[ARG1:.*]]: vector<8xi16>, %[[ARG2:.*]]: vector<8xi32>) -> vector<8xf32> {
llvm.func @mma(%loaded_c_casted: vector<8xf32>, %loaded_a: vector<8xi16>, %loaded_b_casted: vector<8xi32>) -> vector<8xf32> {
  // CHECK: %[[VAR0:.*]] = llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%[[ARG1]], %[[ARG2]], %[[ARG0]]) {convergent, function_type = !llvm.func<vector<8xf32> (vector<8xi16>, vector<8xi32>, vector<8xf32>)>, linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, sym_name = "_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f", visibility_ = 0 : i64, will_return} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %c_result = xevm.mma %loaded_a, %loaded_b_casted, %loaded_c_casted { shape=<m=8, n=16, k=16>, types=<d=f32, a=f16, b=f16, c=f32> } : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  llvm.return %c_result : vector<8xf32>
}

// -----
// CHECK: llvm.func spir_funccc @_Z22atomic_work_item_fenceiii(i32, i32, i32) attributes {no_unwind}
llvm.func @memfence() {
  // CHECK: %[[VAR0:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[VAR1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VAR2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call spir_funccc @_Z22atomic_work_item_fenceiii(%[[VAR2]], %[[VAR0]], %[[VAR1]]) {function_type = !llvm.func<void (i32, i32, i32)>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "_Z22atomic_work_item_fenceiii", visibility_ = 0 : i64} : (i32, i32, i32) -> ()
  xevm.memfence <{addrspace=#xevm.addr_space<global>, scope=#xevm.mem_scope<workgroup>}>
  llvm.return
}

// -----
// CHECK: llvm.func spir_funccc @_Z8prefetchPU3AS1Kcm(!llvm.ptr<1>, i64) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
// CHECK: llvm.func @prefetch(%[[ARG0:.*]]: !llvm.ptr<1>) {
llvm.func @prefetch(%ptr: !llvm.ptr<1>) {
  // CHECK: %[[VAR0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.call spir_funccc @_Z8prefetchPU3AS1Kcm(%[[ARG0]], %[[VAR0]]) {function_type = !llvm.func<void (ptr<1>, i64)>, linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, sym_name = "_Z8prefetchPU3AS1Kcm", visibility_ = 0 : i64
  xevm.prefetch %ptr <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>)
  llvm.return
}

