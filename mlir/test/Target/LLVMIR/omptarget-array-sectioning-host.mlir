// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes provided to the OpenMP kernel argument
// structure are correct when lowering to LLVM-IR from MLIR with 3-D bounds
// provided for a 3-D array. One with full default size, and the other with
// a user specified OpenMP array sectioning. We expect the default sized
// array bounds to lower to the full size of the array and the sectioned
// array to be the size of 3*3*1*element-byte-size (36 bytes in this case).

module attributes {omp.is_target_device = false} {
  llvm.func @_3d_target_array_section() {
    %0 = llvm.mlir.addressof @_QFEinarray : !llvm.ptr
    %1 = llvm.mlir.addressof @_QFEoutarray : !llvm.ptr
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = omp.bounds   lower_bound(%3 : i64) upper_bound(%4 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %6 = omp.bounds   lower_bound(%2 : i64) upper_bound(%2 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %7 = omp.map.info var_ptr(%0 : !llvm.ptr, !llvm.array<3 x array<3 x array<3 x i32>>>)   map_clauses(tofrom) capture(ByRef) bounds(%5, %5, %6) -> !llvm.ptr {name = "inarray(1:3,1:3,2:2)"}
    %8 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.array<3 x array<3 x array<3 x i32>>>)   map_clauses(tofrom) capture(ByRef) bounds(%5, %5, %5) -> !llvm.ptr {name = "outarray(1:3,1:3,1:3)"}
    omp.target   map_entries(%7 -> %arg0, %8 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
      %9 = llvm.mlir.constant(0 : i64) : i64
      %10 = llvm.mlir.constant(1 : i64) : i64
      %11 = llvm.getelementptr %arg0[0, %10, %9, %9] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<3 x array<3 x i32>>>
      %12 = llvm.load %11 : !llvm.ptr -> i32
      %13 = llvm.getelementptr %arg1[0, %10, %9, %9] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<3 x array<3 x i32>>>
      llvm.store %12, %13 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEinarray() {addr_space = 0 : i32} : !llvm.array<3 x array<3 x array<3 x i32>>> {
    %0 = llvm.mlir.zero : !llvm.array<3 x array<3 x array<3 x i32>>>
    llvm.return %0 : !llvm.array<3 x array<3 x array<3 x i32>>>
  }
  llvm.mlir.global internal @_QFEoutarray() {addr_space = 0 : i32} : !llvm.array<3 x array<3 x array<3 x i32>>> {
    %0 = llvm.mlir.zero : !llvm.array<3 x array<3 x array<3 x i32>>>
    llvm.return %0 : !llvm.array<3 x array<3 x array<3 x i32>>>
  }
}

// CHECK: @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 36, i64 108]
// CHECK: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 35, i64 35]
// CHECKL: @.offload_mapnames = private constant [2 x ptr] [ptr @0, ptr @1]

// CHECK: define void @_3d_target_array_section()

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr @_QFEinarray, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr getelementptr inbounds ([3 x [3 x [3 x i32]]], ptr @_QFEinarray, i64 0, i64 1, i64 0, i64 0), ptr %[[OFFLOADPTRS]], align 8

// CHECK: %[[OFFLOADBASEPTRS2:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr @_QFEoutarray, ptr %[[OFFLOADBASEPTRS2]], align 8
// CHECK: %[[OFFLOADPTRS2:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr @_QFEoutarray, ptr %[[OFFLOADPTRS2]], align 8
