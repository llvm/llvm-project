// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test verifies that a privatized pointer map (a parent map carrying the
// target_param | private | attach map type combination) is lowered such that:
//   * the privatized parent is emitted as an individual map entry rather than
//     undergoing the standard parent-with-members mapping, and
//   * the parent which now has the attach map type is still passed as a kernel
//     argument (OMP_MAP_TARGET_PARAM), unlike normal attach maps.

module attributes {omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 52>} {
  llvm.func @assumed_shape_array_priv_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %member = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) name("") -> !llvm.ptr
    %parent = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(target_param, private, attach) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) members(%member : [0] : !llvm.ptr) name("arr_read_write") -> !llvm.ptr
    %attach = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) name("arr_read_write") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%parent -> %arg2, %attach -> %arg3, %member -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [4 x i64] [i64 16544, i64 3, i64 16384, i64 288]

// CHECK: define void @assumed_shape_array_priv_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[MEMBER_PTR:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[ATTACH_PTR:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[PARENT_PTR:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[BASEPTRS0:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS0]], align 8
// CHECK:  %[[OFFPTRS0:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[MEMBER_PTR]], ptr %[[OFFPTRS0]], align 8
// CHECK:  %[[BASEPTRS1:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS1]], align 8
// CHECK:  %[[OFFPTRS1:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[PARENT_PTR]], ptr %[[OFFPTRS1]], align 8
// CHECK:  %[[BASEPTRS2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS2]], align 8
// CHECK:  %[[OFFPTRS2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[ATTACH_PTR]], ptr %[[OFFPTRS2]], align 8
