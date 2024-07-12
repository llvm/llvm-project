// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering
// to LLVM-IR from MLIR when a fortran common block is lowered alongside
// the omp.map.info.

module attributes {omp.is_target_device = false} {
  llvm.func @omp_map_common_block_using_common_block_members() {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.addressof @var_common_ : !llvm.ptr
    %3 = llvm.getelementptr %2[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %4 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "var1"}
    %6 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "var2"}
    omp.target map_entries(%5 -> %arg0, %6 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
      omp.terminator
    }
    llvm.return
  }

  llvm.func @omp_map_common_block_using_common_block_symbol() {
    %0 = llvm.mlir.addressof @var_common_ : !llvm.ptr
    %1 = omp.map.info var_ptr(%0 : !llvm.ptr, !llvm.array<8 x i8>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "var_common"}
    omp.target map_entries(%1 -> %arg0 : !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr):
      omp.terminator
    }
    llvm.return
  }

  llvm.mlir.global common @var_common_(dense<0> : vector<8xi8>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<8 x i8>
}

// CHECK: @[[GLOBAL_BYTE_ARRAY:.*]] = common global [8 x i8] zeroinitializer, align 4

// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 4, i64 4]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 35]

// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 8]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]

// CHECK: define void @omp_map_common_block_using_common_block_members()
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr @[[GLOBAL_BYTE_ARRAY]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFLOADPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr @[[GLOBAL_BYTE_ARRAY]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr getelementptr (i8, ptr @[[GLOBAL_BYTE_ARRAY]], i64 4), ptr %[[BASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr getelementptr (i8, ptr @[[GLOBAL_BYTE_ARRAY]], i64 4), ptr %[[OFFLOADPTRS]], align 8

// CHECK: define void @omp_map_common_block_using_common_block_symbol()
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr @[[GLOBAL_BYTE_ARRAY]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFLOADPTRS:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr @[[GLOBAL_BYTE_ARRAY]], ptr %[[OFFLOADPTRS]], align 8