// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering
// to LLVM-IR from MLIR when performing explicit member mapping of a record type
// that includes another nested record type (C++/C class/structure, Fortran
// derived type) where members of both the nested and outer record type have
// members mapped.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(10 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(f32, array<10 x i32>, struct<(f32, i32)>, i32)> : (i64) -> !llvm.ptr
    %5 = llvm.getelementptr %4[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, array<10 x i32>, struct<(f32, i32)>, i32)>
    %6 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %7 = llvm.getelementptr %4[0, 2, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, array<10 x i32>, struct<(f32, i32)>, i32)>
    %8 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %9 = llvm.getelementptr %4[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, array<10 x i32>, struct<(f32, i32)>, i32)>
    %10 = omp.map.bounds lower_bound(%2 : i64) upper_bound(%1 : i64) extent(%0 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %11 = omp.map.info var_ptr(%9 : !llvm.ptr, !llvm.array<10 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%10) -> !llvm.ptr
    %12 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.struct<(f32, array<10 x i32>, struct<(f32, i32)>, i32)>) map_clauses(tofrom) capture(ByRef) members(%6, %8, %11 : [3], [2, 1], [1] : !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr {partial_map = true}
    omp.target map_entries(%6 -> %arg0, %8 -> %arg1, %11 -> %arg2, %12 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_sizes = private unnamed_addr constant [4 x i64] [i64 0, i64 4, i64 4, i64 16]
// CHECK: @.offload_maptypes = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710659]

// CHECK: define void @_QQmain()
// CHECK: %[[ALLOCA:.*]] = alloca { float, [10 x i32], { float, i32 }, i32 }, i64 1, align 8
// CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr { float, [10 x i32], { float, i32 }, i32 }, ptr %[[ALLOCA]], i32 0, i32 3 
// CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr { float, [10 x i32], { float, i32 }, i32 }, ptr %[[ALLOCA]], i32 0, i32 2, i32 1
// CHECK: %[[MEMBER_ACCESS_3:.*]] = getelementptr { float, [10 x i32], { float, i32 }, i32 }, ptr %[[ALLOCA]], i32 0, i32 1

// CHECK: %[[LAST_MEMBER:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS_3]], i64 0, i64 1
// CHECK: %[[FIRST_MEMBER:.*]] = getelementptr i32, ptr %[[MEMBER_ACCESS_1]], i64 1
// CHECK: %[[FIRST_MEMBER_OFF:.*]] = ptrtoint ptr %[[FIRST_MEMBER]] to i64
// CHECK: %[[SECOND_MEMBER_OFF:.*]] = ptrtoint ptr %[[MEMBER_ACCESS_3]] to i64
// CHECK: %[[MEMBER_DIFF:.*]] = sub i64 %[[FIRST_MEMBER_OFF]], %[[SECOND_MEMBER_OFF]]
// CHECK: %[[OFFLOAD_SIZE:.*]] = sdiv exact i64 %[[MEMBER_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
// CHECK: %[[PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr %[[MEMBER_ACCESS_3]], ptr %[[PTR_ARR]], align 8
// CHECK: %[[SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK: store i64 %[[OFFLOAD_SIZE]], ptr %[[SIZE_ARR]], align 8

// CHECK: %[[BASE_PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_2]], align 8
// CHECK: %[[PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[PTR_ARR_2]], align 8

// CHECK: %[[BASE_PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_3]], align 8
// CHECK: %[[PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[PTR_ARR_3]], align 8

// CHECK: %[[BASE_PTR_ARR_4:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_4]], align 8
// CHECK: %[[PTR_ARR_4:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK: store ptr %[[LAST_MEMBER]], ptr %[[PTR_ARR_4]], align 8
