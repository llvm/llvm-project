// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// The intent of these tests are to check that re-ordering the arguments of use_device_addr/ptr do
// not negatively impact the code generation. It's important to note that this test is missing
// components that'd generate a fully funcitoning executeable, as the IR was reduced to keep the
// primary components for the tests.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 50>} {
  llvm.func @mix_use_device_ptr_and_addr_and_map_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg11: !llvm.ptr, %arg12: !llvm.ptr) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = omp.map.bounds lower_bound(%0 : i64) upper_bound(%1 : i64) extent(%1 : i64) stride(%2 : i64) start_idx(%0 : i64) {stride_in_bytes = true}
    %4 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(i64)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %5 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %6 = omp.map.info var_ptr(%arg2 : !llvm.ptr, i32) var_ptr_ptr(%arg3 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%3) -> !llvm.ptr
    %7 = omp.map.info var_ptr(%arg2 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) members(%6 : [0] : !llvm.ptr) -> !llvm.ptr
    %8 = omp.map.info var_ptr(%arg4 : !llvm.ptr, f32) var_ptr_ptr(%arg5 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %9 = omp.map.info var_ptr(%arg4 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) members(%8 : [0] : !llvm.ptr) -> !llvm.ptr
    %10 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(i64)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data map_entries(%4, %5 : !llvm.ptr, !llvm.ptr) use_device_addr(%7 -> %arg6, %9 -> %arg7, %6 -> %arg8, %8 -> %arg9 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) use_device_ptr(%10 -> %arg10 : !llvm.ptr) {
      %11 = llvm.getelementptr %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
      %12 = llvm.getelementptr %arg12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
      %13 = llvm.load %11 : !llvm.ptr -> i64
      llvm.store %13, %12 : i64, !llvm.ptr
      %14 = llvm.mlir.constant(48 : i32) : i32
      "llvm.intr.memcpy"(%arg11, %arg6, %14) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
      %15 = llvm.getelementptr %arg11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
      %17 = llvm.getelementptr %16[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %18 = llvm.load %17 : !llvm.ptr -> i32
      llvm.store %18, %arg1 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }

  llvm.func @mix_use_device_ptr_and_addr_and_map_2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg11: !llvm.ptr, %arg12: !llvm.ptr) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = omp.map.bounds lower_bound(%0 : i64) upper_bound(%1 : i64) extent(%1 : i64) stride(%2 : i64) start_idx(%0 : i64) {stride_in_bytes = true}
    %4 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(i64)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %5 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %6 = omp.map.info var_ptr(%arg2 : !llvm.ptr, i32) var_ptr_ptr(%arg3 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%3) -> !llvm.ptr
    %7 = omp.map.info var_ptr(%arg2 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) members(%6 : [0] : !llvm.ptr) -> !llvm.ptr
    %8 = omp.map.info var_ptr(%arg4 : !llvm.ptr, f32) var_ptr_ptr(%arg5 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %9 = omp.map.info var_ptr(%arg4 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) members(%8 : [0] : !llvm.ptr) -> !llvm.ptr
    %10 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(i64)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data map_entries(%5, %4 : !llvm.ptr, !llvm.ptr) use_device_addr(%8 -> %arg6, %6 -> %arg7, %7 -> %arg8, %9 -> %arg9 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) use_device_ptr(%10 -> %arg10 : !llvm.ptr) {
      %11 = llvm.getelementptr %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
      %12 = llvm.getelementptr %arg12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
      %13 = llvm.load %11 : !llvm.ptr -> i64
      llvm.store %13, %12 : i64, !llvm.ptr
      %14 = llvm.mlir.constant(48 : i32) : i32
      "llvm.intr.memcpy"(%arg11, %arg8, %14) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
      %15 = llvm.getelementptr %arg11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
      %17 = llvm.getelementptr %16[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %18 = llvm.load %17 : !llvm.ptr -> i32
      llvm.store %18, %arg1 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @mix_use_device_ptr_and_addr_and_map_(ptr %[[ARG_0:.*]], ptr %[[ARG_1:.*]], ptr %[[ARG_2:.*]], ptr %[[ARG_3:.*]], ptr %[[ARG_4:.*]], ptr %[[ARG_5:.*]], ptr %[[ARG_6:.*]], ptr %[[ARG_7:.*]]) {
// CHECK: %[[ALLOCA:.*]] = alloca ptr, align 8
// CHECK: %[[BASEPTR_0_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr %[[ARG_0]], ptr %[[BASEPTR_0_GEP]], align 8
// CHECK: %[[BASEPTR_2_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK: store ptr %[[ARG_2]], ptr %[[BASEPTR_2_GEP]], align 8
// CHECK: %[[BASEPTR_3_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 9
// CHECK: store ptr %[[ARG_4]], ptr %[[BASEPTR_3_GEP]], align 8

// CHECK: call void @__tgt_target_data_begin_mapper({{.*}})
// CHECK: %[[LOAD_BASEPTR_0:.*]] = load ptr, ptr %[[BASEPTR_0_GEP]], align 8
// store ptr %[[LOAD_BASEPTR_0]], ptr %[[ALLOCA]], align 8
// CHECK: %[[LOAD_BASEPTR_2:.*]] = load ptr, ptr %[[BASEPTR_2_GEP]], align 8
// CHECK: %[[LOAD_BASEPTR_3:.*]] = load ptr, ptr %[[BASEPTR_3_GEP]], align 8
// CHECK: %[[GEP_A4:.*]] = getelementptr { i64 }, ptr %[[ARG_4]], i32 0, i32 0
// CHECK: %[[GEP_A7:.*]] = getelementptr { i64 }, ptr %[[ARG_7]], i32 0, i32 0
// CHECK: %[[LOAD_A4:.*]] = load i64, ptr %[[GEP_A4]], align 4
// CHECK: store i64 %[[LOAD_A4]], ptr %[[GEP_A7]], align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[ARG_6]], ptr %[[LOAD_BASEPTR_2]], i32 48, i1 false)
// CHECK: %[[GEP_A6:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARG_6]], i32 0, i32 0
// CHECK: %[[LOAD_A6:.*]] = load ptr, ptr %[[GEP_A6]], align 8
// CHECK: %[[GEP_A6_2:.*]] = getelementptr i8, ptr %[[LOAD_A6]], i64 2
// CHECK: %[[LOAD_A6_2:.*]] = load i32, ptr %[[GEP_A6_2]], align 4
// CHECK: store i32 %[[LOAD_A6_2]], ptr %[[ARG_1]], align 4
// CHECK: call void @__tgt_target_data_end_mapper({{.*}})

// CHECK: define void @mix_use_device_ptr_and_addr_and_map_2(ptr %[[ARG_0:.*]], ptr %[[ARG_1:.*]], ptr %[[ARG_2:.*]], ptr %[[ARG_3:.*]], ptr %[[ARG_4:.*]], ptr %[[ARG_5:.*]], ptr %[[ARG_6:.*]], ptr %[[ARG_7:.*]]) {
// CHECK: %[[ALLOCA:.*]] = alloca ptr, align 8
// CHECK: %[[BASEPTR_1_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr %[[ARG_0]], ptr %[[BASEPTR_1_GEP]], align 8
// CHECK: %[[BASEPTR_2_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK: store ptr %[[ARG_2]], ptr %[[BASEPTR_2_GEP]], align 8
// CHECK: %[[BASEPTR_3_GEP:.*]] = getelementptr inbounds [12 x ptr], ptr %.offload_baseptrs, i32 0, i32 9
// CHECK: store ptr %[[ARG_4]], ptr %[[BASEPTR_3_GEP]], align 8
// CHECK: call void @__tgt_target_data_begin_mapper({{.*}})
// CHECK: %[[LOAD_BASEPTR_1:.*]] = load ptr, ptr %[[BASEPTR_1_GEP]], align 8
// store ptr %[[LOAD_BASEPTR_1]], ptr %[[ALLOCA]], align 8
// CHECK: %[[LOAD_BASEPTR_2:.*]] = load ptr, ptr %[[BASEPTR_2_GEP]], align 8
// CHECK: %[[LOAD_BASEPTR_3:.*]] = load ptr, ptr %[[BASEPTR_3_GEP]], align 8
// CHECK: %[[GEP_A4:.*]] = getelementptr { i64 }, ptr %[[ARG_4]], i32 0, i32 0
// CHECK: %[[GEP_A7:.*]] = getelementptr { i64 }, ptr %[[ARG_7]], i32 0, i32 0
// CHECK: %[[LOAD_A4:.*]] = load i64, ptr %[[GEP_A4]], align 4
// CHECK: store i64 %[[LOAD_A4]], ptr %[[GEP_A7]], align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[ARG_6]], ptr %[[LOAD_BASEPTR_2]], i32 48, i1 false)
// CHECK: %[[GEP_A6:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARG_6]], i32 0, i32 0
// CHECK: %[[LOAD_A6:.*]] = load ptr, ptr %[[GEP_A6]], align 8
// CHECK: %[[GEP_A6_2:.*]] = getelementptr i8, ptr %[[LOAD_A6]], i64 2
// CHECK: %[[LOAD_A6_2:.*]] = load i32, ptr %[[GEP_A6_2]], align 4
// CHECK: store i32 %[[LOAD_A6_2]], ptr %[[ARG_1]], align 4
// CHECK: call void @__tgt_target_data_end_mapper({{.*}})
