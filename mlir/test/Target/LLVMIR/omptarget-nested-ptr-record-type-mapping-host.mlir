// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering
// to LLVM-IR from MLIR when performing explicit member mapping of a record type
// that includes records with pointer members in various locations of the record
// types hierarchy.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @omp_nested_derived_type_alloca_map(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(6 : index) : i64
    %5 = omp.map.bounds lower_bound(%3 : i64) upper_bound(%0 : i64) extent(%0 : i64) stride(%1 : i64) start_idx(%3 : i64) {stride_in_bytes = true}
    %6 = llvm.getelementptr %arg0[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<(f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>
    %7 = llvm.getelementptr %6[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %9 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) var_ptr_ptr(%8 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%5) -> !llvm.ptr {name = ""}
    %10 = omp.map.info var_ptr(%7 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%nest%array_k"}
    %11 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<(f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>) map_clauses(tofrom) capture(ByRef) members(%10, %9 : [6,2], [6,2,0] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "one_l", partial_map = true}
    omp.target map_entries(%10 -> %arg1, %9 -> %arg2, %11 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 48, i64 8, i64 20]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710675]

// CHECK: define void @omp_nested_derived_type_alloca_map(ptr %[[ARG:.*]]) {

// CHECK: %[[NESTED_DTYPE_MEMBER_GEP:.*]] = getelementptr { float, { ptr, i64, i32, i8, i8, i8, i8 }, [10 x i32], float, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i32, { float, [10 x i32], { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i32 } }, ptr %[[ARG]], i32 0, i32 6
// CHECK: %[[NESTED_STRUCT_PTR_MEMBER_GEP:.*]] = getelementptr { float, [10 x i32], { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i32 }, ptr %[[NESTED_DTYPE_MEMBER_GEP]], i32 0, i32 2
// CHECK: %[[NESTED_STRUCT_PTR_MEMBER_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[NESTED_STRUCT_PTR_MEMBER_GEP]], i32 0, i32 0
// CHECK: %[[NESTED_STRUCT_PTR_MEMBER_BADDR_LOAD:.*]] = load ptr, ptr %[[NESTED_STRUCT_PTR_MEMBER_BADDR_GEP]], align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[NESTED_STRUCT_PTR_MEMBER_BADDR_LOAD]], i64 0
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[NESTED_STRUCT_PTR_MEMBER_GEP]], i64 1
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE_SEGMENT_CALC_1]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[NESTED_STRUCT_PTR_MEMBER_GEP]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE_SEGMENT_CALC_2]], %[[DTYPE_SIZE_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[NESTED_STRUCT_PTR_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[DTYPE_SIZE_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[NESTED_STRUCT_PTR_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[NESTED_STRUCT_PTR_MEMBER_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[NESTED_STRUCT_PTR_MEMBER_BADDR_GEP]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARR_OFFSET]], ptr %[[OFFLOAD_PTRS]], align 8
