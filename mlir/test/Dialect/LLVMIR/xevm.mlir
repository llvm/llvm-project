// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @blockload2d(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32)
func.func @blockload2d(%a: !llvm.ptr<1>, %base_width_a: i32, %base_height_a: i32,
  %base_pitch_a: i32, %x: i32, %y: i32) -> vector<8xi16> {
  // CHECK: %[[VAR0:.*]] = xevm.blockload2d %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]]
  // CHECK-DAG: elem_size_in_bits = 16 : i32
  // CHECK-DAG: tile_width = 16 : i32
  // CHECK-DAG: tile_height = 8 : i32
  // CHECK-DAG: v_blocks = 1 : i32
  // CHECK-DAG: transpose = false
  // CHECK-DAG: pack_register = false
  // CHECK-DAG: cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>
  // CHECK: (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y
    <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32,
    transpose=false, pack_register=false, cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}>
    : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  return %loaded_a : vector<8xi16>
}

// -----
// CHECK-LABEL: func.func @blockstore2d(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME: %[[ARG6:.*]]: vector<8xi32>)
func.func @blockstore2d(%c: !llvm.ptr<1>, %base_width_c: i32, %base_height_c: i32,
  %base_pitch_c: i32, %x: i32, %y: i32, %c_result_casted: vector<8xi32>) {
  // CHECK: xevm.blockstore2d %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]
  // CHECK-DAG: elem_size_in_bits = 32 : i32
  // CHECK-DAG: tile_width = 16 : i32
  // CHECK-DAG: tile_height = 8 : i32
  // CHECK: (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted
    <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
    : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  return
}

// -----
// CHECK-LABEL: func.func @blockprefetch2d(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32)
func.func @blockprefetch2d(%ptr: !llvm.ptr<1>, %base_width: i32, %base_height: i32,
  %base_pitch: i32, %x: i32, %y: i32) {
  // CHECK: xevm.blockprefetch2d %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]]
  // CHECK-DAG: elem_size_in_bits = 8 : i32
  // CHECK-DAG: tile_width = 32 : i32
  // CHECK-DAG: tile_height = 8 : i32
  // CHECK-DAG: v_blocks = 1 : i32
  // CHECK-DAG: cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>
  // CHECK:  (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y <{elem_size_in_bits=8 : i32,
    tile_width=32 : i32, tile_height=8 : i32, v_blocks=1 : i32,
    cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  return
}

// -----
// CHECK-LABEL: func.func @blockload(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>)
func.func @blockload(%ptr: !llvm.ptr<1>) -> vector<4xi16> {
  // CHECK: %[[VAR0:.*]] = xevm.blockload %[[ARG0]]
  // CHECK-SAME: cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>
  // CHECK-SAME: (!llvm.ptr<1>) -> vector<4xi16>
  %loaded = xevm.blockload %ptr <{cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}>
              : (!llvm.ptr<1>) -> vector<4xi16>
  return %loaded : vector<4xi16>
}

// -----
// CHECK-LABEL: func.func @blockstore(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[ARG1:.*]]: vector<4xi32>)
func.func @blockstore(%ptr: !llvm.ptr<1>, %value: vector<4xi32>) {
  // CHECK: xevm.blockstore %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: (!llvm.ptr<1>, vector<4xi32>)
  xevm.blockstore %ptr, %value : (!llvm.ptr<1>, vector<4xi32>)
  return
}

// -----
// CHECK-LABEL: func.func @mma(
// CHECK-SAME: %[[ARG0:.*]]: vector<8xf32>, %[[ARG1:.*]]: vector<8xi16>, %[[ARG2:.*]]: vector<8xi32>)
func.func @mma(%loaded_c_casted: vector<8xf32>, %loaded_a: vector<8xi16>, %loaded_b_casted: vector<8xi32>) -> vector<8xf32> {
  // CHECK: %0 = xevm.mma %[[ARG1]], %[[ARG2]], %[[ARG0]] {shape = <m = 8, n = 16, k = 16>,
  // CHECK-SAME: types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %c_result = xevm.mma %loaded_a, %loaded_b_casted, %loaded_c_casted { shape=<m=8, n=16, k=16>,
    types=<d=f32, a=f16, b=f16, c=f32> } : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  return %c_result : vector<8xf32>
}

// -----
// CHECK-LABEL: func.func @memfence()
func.func @memfence() {
  // CHECK: xevm.memfence
  // CHECK-DAG: addrspace = #xevm.addr_space<global>
  // CHECK-DAG: scope = #xevm.mem_scope<workgroup>
  xevm.memfence <{addrspace=#xevm.addr_space<global>, scope=#xevm.mem_scope<workgroup>}>
  return
}

// -----
// CHECK-LABEL: func.func @prefetch(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>)
func.func @prefetch(%ptr: !llvm.ptr<1>) {
  // CHECK: xevm.prefetch %[[ARG0]]
  // CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>)
  xevm.prefetch %ptr <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>)
  return
}

// -----
// CHECK-LABEL: @xevm_module [#xevm.target<O = 3, chip = "pvc">] {
gpu.module @xevm_module [#xevm.target<O = 3, chip = "pvc">]{
}

// -----
// CHECK-LABEL: @xevm_special_ids
llvm.func @xevm_special_ids() -> i32 {
  // CHECK: xevm.local_id.x : i32
  %1 = xevm.local_id.x : i32
  // CHECK: xevm.local_id.y : i32
  %2 = xevm.local_id.y : i32
  // CHECK: xevm.local_id.z : i32
  %3 = xevm.local_id.z : i32
  // CHECK: xevm.local_size.x : i32
  %4 = xevm.local_size.x : i32
  // CHECK: xevm.local_size.y : i32
  %5 = xevm.local_size.y : i32
  // CHECK: xevm.local_size.z : i32
  %6 = xevm.local_size.z : i32
  // CHECK: xevm.group_id.x : i32
  %7 = xevm.group_id.x : i32
  // CHECK: xevm.group_id.y : i32
  %8 = xevm.group_id.y : i32
  // CHECK: xevm.group_id.z : i32
  %9 = xevm.group_id.z : i32
  // CHECK: xevm.group_count.x : i32
  %10 = xevm.group_count.x : i32
  // CHECK: xevm.group_count.y : i32
  %11 = xevm.group_count.y : i32
  // CHECK: xevm.group_count.z : i32
  %12 = xevm.group_count.z : i32
  // CHECK: xevm.lane_id : i32
  %14 = xevm.lane_id : i32
  // CHECK: xevm.subgroup_size : i32
  %39 = xevm.subgroup_size : i32
  // CHECK: xevm.subgroup_id : i32
  %40 = xevm.subgroup_id : i32
  llvm.return %1 : i32
}
