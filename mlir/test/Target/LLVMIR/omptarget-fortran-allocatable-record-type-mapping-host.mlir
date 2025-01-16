// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering
// to LLVM-IR from MLIR when performing explicit member mapping of a record type
// that includes fortran allocatables in various locations of the record types
// hierarchy.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @omp_map_derived_type_allocatable_member(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = omp.map.bounds lower_bound(%2 : i64) upper_bound(%0 : i64) extent(%0 : i64) stride(%1 : i64) start_idx(%2 : i64) {stride_in_bytes = true}
    %4 = llvm.getelementptr %arg0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %6 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) var_ptr_ptr(%5 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%3) -> !llvm.ptr {name = ""}
    %7 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%array_j"}
    %8 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>) map_clauses(tofrom) capture(ByRef) members(%7, %6 : [4,-1], [4,0] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "one_l", partial_map = true}
    omp.target map_entries(%7 -> %arg1, %6 -> %arg2, %8 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @omp_allocatable_derived_type_member_map(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = omp.map.bounds lower_bound(%7 : i64) upper_bound(%5 : i64) extent(%5 : i64) stride(%6 : i64) start_idx(%7 : i64) {stride_in_bytes = true}
    %9 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    llvm.store %9, %3 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>, !llvm.ptr
    %10 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
    %12 = llvm.getelementptr %11[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %14 = omp.map.info var_ptr(%12 : !llvm.ptr, i32) var_ptr_ptr(%13 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%8) -> !llvm.ptr {name = ""}
    %15 = omp.map.info var_ptr(%12 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%array_j"}
    %16 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    llvm.store %16, %1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>, !llvm.ptr
    %17 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %18 = llvm.load %17 : !llvm.ptr -> !llvm.ptr
    %19 = llvm.getelementptr %18[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %20 = omp.map.info var_ptr(%19 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%k"}
    %21 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %22 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>) var_ptr_ptr(%21 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %23 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>) map_clauses(tofrom) capture(ByRef) members(%22, %15, %14, %20 : [0,-1,-1], [0,4,-1], [0,4,0], [0,5,-1] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "one_l"}
    omp.target map_entries(%22 -> %arg1, %15 -> %arg2, %14 -> %arg3, %20 -> %arg4, %23 -> %arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @omp_alloca_nested_derived_type_map(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.mlir.constant(3 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(6 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = omp.map.bounds lower_bound(%9 : i64) upper_bound(%5 : i64) extent(%5 : i64) stride(%7 : i64) start_idx(%9 : i64) {stride_in_bytes = true}
    %11 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    llvm.store %11, %3 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>, !llvm.ptr
    %12 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %13 = llvm.load %12 : !llvm.ptr -> !llvm.ptr
    %14 = llvm.getelementptr %13[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>
    %15 = llvm.getelementptr %14[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %17 = omp.map.info var_ptr(%15 : !llvm.ptr, i32) var_ptr_ptr(%16 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%10) -> !llvm.ptr {name = ""}
    %18 = omp.map.info var_ptr(%15 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%nest%array_k"}
    %19 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    llvm.store %19, %1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>, !llvm.ptr
    %20 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %21 = llvm.load %20 : !llvm.ptr -> !llvm.ptr
    %22 = llvm.getelementptr %21[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>
    %23 = llvm.getelementptr %22[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %24 = omp.map.info var_ptr(%23 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%nest%k"}
    %25 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>
    %26 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<"_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>) var_ptr_ptr(%25 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %27 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)>) map_clauses(tofrom) capture(ByRef) members(%26, %18, %17, %24 : [0,-1,-1,-1], [0,6,2,-1], [0,6,2,0], [0,6,3,-1] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "one_l"}
    omp.target map_entries(%26 -> %arg1, %18 -> %arg2, %17 -> %arg3, %24 -> %arg4, %27 -> %arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @omp_nested_derived_type_alloca_map(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(6 : index) : i64
    %5 = omp.map.bounds lower_bound(%3 : i64) upper_bound(%0 : i64) extent(%0 : i64) stride(%1 : i64) start_idx(%3 : i64) {stride_in_bytes = true}
    %6 = llvm.getelementptr %arg0[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<"_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>
    %7 = llvm.getelementptr %6[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %9 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) var_ptr_ptr(%8 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%5) -> !llvm.ptr {name = ""}
    %10 = omp.map.info var_ptr(%7 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "one_l%nest%array_k"}
    %11 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer", (f32, struct<(ptr, i64, i32, i8, i8, i8, i8)>, array<10 x i32>, f32, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32, struct<"_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer", (f32, array<10 x i32>, struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, i32)>)>) map_clauses(tofrom) capture(ByRef) members(%10, %9 : [6,2,-1], [6,2,0] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "one_l", partial_map = true}
    omp.target map_entries(%10 -> %arg1, %9 -> %arg2, %11 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 48, i64 8, i64 20]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710675]
// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [12 x i64] [i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 8, i64 136, i64 48, i64 8, i64 20, i64 4]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [12 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710659]
// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [12 x i64] [i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 8, i64 240, i64 48, i64 8, i64 20, i64 4]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [12 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710659]
// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 48, i64 8, i64 20]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710675]

// CHECK: define void @omp_map_derived_type_allocatable_member(ptr %[[ARG:.*]]) {

// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_GEP:.*]] = getelementptr %_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer, ptr %[[ARG]], i32 0, i32 4
// CHECK: %[[ALLOCATABLE_MEMBER_BADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_ALLOCATABLE_MEMBER_GEP]], i32 0, i32 0

// CHECK: %[[LOAD_ALLOCATABLE_MEMBER_BADDR:.*]] = load ptr, ptr %[[ALLOCATABLE_MEMBER_BADDR]], align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ALLOCATABLE_MEMBER_BADDR]], i64 0
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_1:.*]] = getelementptr i32, ptr %[[ARR_OFFSET]], i64 1
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE_SEGMENT_CALC_1]] to i64
// CHECK:  %[[DTYPE_SIZE_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_ALLOCATABLE_MEMBER_GEP]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE_SEGMENT_CALC_2]], %[[DTYPE_SIZE_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[DTYPE_SIZE_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[ALLOCATABLE_MEMBER_BADDR]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ALLOCATABLE_MEMBER_BADDR]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARR_OFFSET]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK: define void @omp_allocatable_derived_type_member_map(ptr %[[ARG:.*]]) {

// CHECK: %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
// CHECK: %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
// CHECK: %[[LOAD_DTYPE_ALLOCATABLE_ARG:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], align 8
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_DTYPE_ALLOCATABLE_ARG]], ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA]], align 8
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA]], i32 0, i32 0
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], align 8
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_ACCESS:.*]] = getelementptr %_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer, ptr %[[DTYPE_ALLOCATABLE_BADDR_LOAD]], i32 0, i32 4
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_ALLOCATABLE_MEMBER_ACCESS]], i32 0, i32 0
// CHECK: %[[LOAD_DTYPE_ALLOCATABLE_ARG:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], align 8
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_DTYPE_ALLOCATABLE_ARG]], ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2]], align 8
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2]], i32 0, i32 0
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], align 8
// CHECK: %[[DTYPE_REGULAR_MEMBER_ACCESS:.*]] = getelementptr %_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer, ptr %[[DTYPE_ALLOCATABLE_BADDR_LOAD]], i32 0, i32 5
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 0, i32 0
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2_LOAD:.*]] = load ptr, ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2]], align 8
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR]], align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_LOAD]], i64 0
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 1
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE_SEGMENT_CALC_1]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[ARG]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE_SEGMENT_CALC_2]], %[[DTYPE_SIZE_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 1
// CHECK: %[[DTYPE_OFFLOAD_PTR_1:.*]] = getelementptr ptr, ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2]], i32 1
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC:.*]] = ptrtoint ptr %[[DTYPE_ALLOCATABLE_MEMBER_ACCESS]] to i64
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_1]] to i64
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_3:.*]] = sub i64 %[[DTYPE_SIZE2_SEGMENT_CALC]], %[[DTYPE_SIZE2_SEGMENT_CALC_2]]
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_4:.*]] = sdiv exact i64 %[[DTYPE_SIZE2_SEGMENT_CALC_3]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_OFFLOAD_PTR_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_ALLOCATABLE_MEMBER_ACCESS]], i32 1
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC:.*]] = ptrtoint ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR]] to i64
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_2]] to i64
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_3:.*]] = sub i64 %[[DTYPE_SIZE3_SEGMENT_CALC]], %[[DTYPE_SIZE3_SEGMENT_CALC_2]]
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_4:.*]] = sdiv exact i64 %[[DTYPE_SIZE3_SEGMENT_CALC_3]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP:.*]] = getelementptr ptr, ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR]], i32 1
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_REGULAR_MEMBER_ACCESS]] to i64
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP]] to i64
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE4_SEGMENT_CALC_2]], %[[DTYPE_SIZE4_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE4_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_OFFLOAD_PTR_3:.*]] = getelementptr i32, ptr %[[DTYPE_REGULAR_MEMBER_ACCESS]], i32 1
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE5_SEGMENT_CALC]] to i64
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_3]] to i64
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE5_SEGMENT_CALC_2]], %[[DTYPE_SIZE5_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE5_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[DTYPE_SIZE_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_1]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[DTYPE_SIZE2_SEGMENT_CALC_4]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_2]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK:  store i64 %[[DTYPE_SIZE3_SEGMENT_CALC_4]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 4
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 4
// CHECK:  store i64 %[[DTYPE_SIZE4_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 5
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_3]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 5
// CHECK:  store i64 %[[DTYPE_SIZE5_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 6
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 7
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_2_LOAD]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 8
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 8
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 9
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 9
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 10
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 10
// CHECK:  store ptr %[[ARR_OFFSET]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 11
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 11
// CHECK:  store ptr %[[DTYPE_REGULAR_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTRS]], align 8


// CHECK: define void @omp_alloca_nested_derived_type_map(ptr %[[ARG:.*]]) {

// CHECK: %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
// CHECK: %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
// CHECK: %[[LOAD_DTYPE_ALLOCATABLE_ARG:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], align 8
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_DTYPE_ALLOCATABLE_ARG]], ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA]], align 8
// CHECK: %[[DTYPE_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA]], i32 0, i32 0
// CHECK: %[[DTYPE_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_BADDR_GEP]], align 8
// CHECK: %[[DTYPE_NESTED_DTYPE_MEMBER_GEP:.*]] = getelementptr %_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer, ptr %[[DTYPE_BADDR_LOAD]], i32 0, i32 6
// CHECK: %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_GEP:.*]] = getelementptr %_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer, ptr %[[DTYPE_NESTED_DTYPE_MEMBER_GEP]], i32 0, i32 2
// CHECK: %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_GEP]], i32 0, i32 0
// CHECK: %[[LOAD_DTYPE_ALLOCATABLE_ARG:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], align 8
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_DTYPE_ALLOCATABLE_ARG]], ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2]], align 8
// CHECK: %[[DTYPE_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[LOCAL_ALLOCATABLE_DTYPE_ALLOCA_2]], i32 0, i32 0
// CHECK: %[[DTYPE_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_BADDR_GEP]], align 8
// CHECK: %[[DTYPE_NESTED_DTYPE_MEMBER_GEP:.*]] = getelementptr %_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer, ptr %[[DTYPE_BADDR_LOAD]], i32 0, i32 6
// CHECK: %[[DTYPE_NESTED_REGULAR_MEMBER_GEP:.*]] = getelementptr %_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer, ptr %[[DTYPE_NESTED_DTYPE_MEMBER_GEP]], i32 0, i32 3
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 0, i32 0
// CHECK: %[[DTYPE_ALLOCATABLE_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], align 8
// CHECK: %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_LOAD]], i64 0
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 1
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE_SEGMENT_CALC_1]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[ARG]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE_SEGMENT_CALC_2]], %[[DTYPE_SIZE_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG]], i32 1
// CHECK: %[[DTYPE_OFFLOAD_PTR_1:.*]] = getelementptr ptr, ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], i32 1
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC:.*]] = ptrtoint ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_GEP]] to i64
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_1]] to i64
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_3:.*]] = sub i64 %[[DTYPE_SIZE2_SEGMENT_CALC]], %[[DTYPE_SIZE2_SEGMENT_CALC_2]]
// CHECK: %[[DTYPE_SIZE2_SEGMENT_CALC_4:.*]] = sdiv exact i64 %[[DTYPE_SIZE2_SEGMENT_CALC_3]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_OFFLOAD_PTR_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_GEP]], i32 1
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC:.*]] = ptrtoint ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]] to i64
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_2]] to i64
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_3:.*]] = sub i64 %[[DTYPE_SIZE3_SEGMENT_CALC]], %[[DTYPE_SIZE3_SEGMENT_CALC_2]]
// CHECK: %[[DTYPE_SIZE3_SEGMENT_CALC_4:.*]] = sdiv exact i64 %[[DTYPE_SIZE3_SEGMENT_CALC_3]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP:.*]] = getelementptr ptr, ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], i32 1
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_NESTED_REGULAR_MEMBER_GEP]] to i64
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP]] to i64
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE4_SEGMENT_CALC_2]], %[[DTYPE_SIZE4_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE4_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE4_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[DTYPE_OFFLOAD_PTR_3:.*]] = getelementptr i32, ptr %[[DTYPE_NESTED_REGULAR_MEMBER_GEP]], i32 1
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE5_SEGMENT_CALC_1]] to i64
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_OFFLOAD_PTR_3]] to i64
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE5_SEGMENT_CALC_2]], %[[DTYPE_SIZE5_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE5_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE5_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[DTYPE_SIZE_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_1]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[DTYPE_SIZE2_SEGMENT_CALC_4]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_2]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK:  store i64 %[[DTYPE_SIZE3_SEGMENT_CALC_4]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 4
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 4
// CHECK:  store i64 %[[DTYPE_SIZE4_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 5
// CHECK:  store ptr %[[DTYPE_OFFLOAD_PTR_3]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [12 x i64], ptr %.offload_sizes, i32 0, i32 5
// CHECK:  store i64 %[[DTYPE_SIZE5_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 6
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_BADDR_GEP]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 7
// CHECK:  store ptr %[[DTYPE_ALLOCATABLE_BADDR_LOAD]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 8
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 8
// CHECK:  store ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 9
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 9
// CHECK:  store ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 10
// CHECK:  store ptr %[[DTYPE_NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 10
// CHECK:  store ptr %[[ARR_OFFSET]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 11
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_ptrs, i32 0, i32 11
// CHECK:  store ptr %[[DTYPE_NESTED_REGULAR_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK: define void @omp_nested_derived_type_alloca_map(ptr %[[ARG:.*]]) {

// CHECK: %[[NESTED_DTYPE_MEMBER_GEP:.*]] = getelementptr %_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer, ptr %[[ARG]], i32 0, i32 6
// CHECK: %[[NESTED_ALLOCATABLE_MEMBER_GEP:.*]] = getelementptr %_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer, ptr %[[NESTED_DTYPE_MEMBER_GEP]], i32 0, i32 2
// CHECK: %[[NESTED_ALLOCATABLE_MEMBER_BADDR_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[NESTED_ALLOCATABLE_MEMBER_GEP]], i32 0, i32 0
// CHECK: %[[NESTED_ALLOCATABLE_MEMBER_BADDR_LOAD:.*]] = load ptr, ptr %[[NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[NESTED_ALLOCATABLE_MEMBER_BADDR_LOAD]], i64 0
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_1:.*]] = getelementptr i32, ptr %[[ARR_OFFSET]], i64 1
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_SIZE_SEGMENT_CALC_1]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_3:.*]] = ptrtoint ptr %[[NESTED_ALLOCATABLE_MEMBER_GEP]] to i64
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_4:.*]] = sub i64 %[[DTYPE_SIZE_SEGMENT_CALC_2]], %[[DTYPE_SIZE_SEGMENT_CALC_3]]
// CHECK: %[[DTYPE_SIZE_SEGMENT_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_SIZE_SEGMENT_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[NESTED_ALLOCATABLE_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8
// CHECK:  %[[OFFLOAD_SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[DTYPE_SIZE_SEGMENT_CALC_5]], ptr %[[OFFLOAD_SIZES]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[NESTED_ALLOCATABLE_MEMBER_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  %[[BASE_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[NESTED_ALLOCATABLE_MEMBER_BADDR_GEP]], ptr %[[BASE_PTRS]], align 8
// CHECK:  %[[OFFLOAD_PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARR_OFFSET]], ptr %[[OFFLOAD_PTRS]], align 8
