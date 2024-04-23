// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks the offload sizes, map types and base pointers and pointers
// provided to the OpenMP kernel argument structure are correct when lowering 
// to LLVM-IR from MLIR when the fortran allocatables flag is switched on and 
// a fortran allocatable descriptor type is provided alongside the omp.map.info,
// the test utilises mapping of array sections, full arrays and individual 
// allocated scalars.

module attributes {omp.is_target_device = false} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(5 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.addressof @_QFEfull_arr : !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "scalar"} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.addressof @_QFEsect_arr : !llvm.ptr
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.getelementptr %3[0, 7, %7, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %9 = llvm.load %8 : !llvm.ptr -> i64
    %10 = llvm.getelementptr %3[0, 7, %7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %11 = llvm.load %10 : !llvm.ptr -> i64
    %12 = llvm.getelementptr %3[0, 7, %7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %13 = llvm.load %12 : !llvm.ptr -> i64
    %14 = llvm.sub %11, %2  : i64
    %15 = omp.map.bounds lower_bound(%7 : i64) upper_bound(%14 : i64) extent(%11 : i64) stride(%13 : i64) start_idx(%9 : i64) {stride_in_bytes = true}
    %16 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %17 = omp.map.info var_ptr(%3 : !llvm.ptr, f32) var_ptr_ptr(%16 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%15) -> !llvm.ptr {name = "full_arr"}
    %18 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) members(%17 : !llvm.ptr) -> !llvm.ptr {name = "full_arr"}
    %19 = llvm.getelementptr %6[0, 7, %7, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %20 = llvm.load %19 : !llvm.ptr -> i64
    %21 = llvm.getelementptr %6[0, 7, %7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.getelementptr %6[0, 7, %7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %24 = llvm.load %23 : !llvm.ptr -> i64
    %25 = llvm.sub %1, %20  : i64
    %26 = llvm.sub %0, %20  : i64
    %27 = omp.map.bounds lower_bound(%25 : i64) upper_bound(%26 : i64) extent(%22 : i64) stride(%24 : i64) start_idx(%20 : i64) {stride_in_bytes = true}
    %28 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %29 = omp.map.info var_ptr(%6 : !llvm.ptr, i32) var_ptr_ptr(%28 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) bounds(%27) -> !llvm.ptr {name = "sect_arr(2:5)"}
    %30 = omp.map.info var_ptr(%6 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) members(%29 : !llvm.ptr) -> !llvm.ptr {name = "sect_arr(2:5)"}
    %31 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %32 = omp.map.info var_ptr(%5 : !llvm.ptr, f32) var_ptr_ptr(%31 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "scalar"}
    %33 = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) members(%32 : !llvm.ptr) -> !llvm.ptr {name = "scalar"}
    omp.target map_entries(%17 -> %arg0, %18 -> %arg1, %29 -> %arg2, %30 -> %arg3, %32 -> %arg4, %33 -> %arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr):
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEfull_arr() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.return %0 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
  llvm.mlir.global internal @_QFEsect_arr() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.return %0 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
}

// CHECK: @[[FULL_ARR_GLOB:.*]] = internal global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } undef
// CHECK: @[[ARR_SECT_GLOB:.*]] = internal global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } undef
// CHECK: @.offload_sizes = private unnamed_addr constant [9 x i64] [i64 0, i64 48, i64 0, i64 0, i64 48, i64 0, i64 0, i64 24, i64 4]
// CHECK: @.offload_maptypes = private unnamed_addr constant [9 x i64] [i64 32, i64 281474976710657, i64 281474976710675, i64 32, i64 1125899906842625, i64 1125899906842643, i64 32, i64 1970324836974593, i64 1970324836974611]
// CHECK: @.offload_mapnames = private constant [9 x ptr] [ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}]

// CHECK: define void @_QQmain()
// CHECK: %[[SCALAR_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
// CHECK: %[[FULL_ARR_SIZE5:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @[[FULL_ARR_GLOB]], i32 0, i32 7, i64 0, i32 1), align 4
// CHECK: %[[FULL_ARR_SIZE4:.*]] = sub i64 %[[FULL_ARR_SIZE5]], 1
// CHECK: %[[ARR_SECT_OFFSET3:.*]] = load i64, ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @[[ARR_SECT_GLOB]], i32 0, i32 7, i64 0, i32 0), align 4
// CHECK: %[[ARR_SECT_OFFSET2:.*]] = sub i64 2, %[[ARR_SECT_OFFSET3]]
// CHECK: %[[ARR_SECT_SIZE4:.*]] = sub i64 5, %[[ARR_SECT_OFFSET3]]
// CHECK: %[[SCALAR_BASE:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[SCALAR_ALLOCA]], i32 0, i32 0
// CHECK: %[[FULL_ARR_SIZE3:.*]] = sub i64 %[[FULL_ARR_SIZE4]], 0
// CHECK: %[[FULL_ARR_SIZE2:.*]] = add i64 %[[FULL_ARR_SIZE3]], 1
// CHECK: %[[FULL_ARR_SIZE1:.*]] = mul i64 1, %[[FULL_ARR_SIZE2]]
// CHECK: %[[FULL_ARR_SIZE:.*]] = mul i64 %[[FULL_ARR_SIZE1]], 4
// CHECK: %[[ARR_SECT_SIZE3:.*]] = sub i64 %[[ARR_SECT_SIZE4]], %[[ARR_SECT_OFFSET2]]
// CHECK: %[[ARR_SECT_SIZE2:.*]] = add i64 %[[ARR_SECT_SIZE3]], 1
// CHECK: %[[ARR_SECT_SIZE1:.*]] = mul i64 1, %[[ARR_SECT_SIZE2]]
// CHECK: %[[ARR_SECT_SIZE:.*]] = mul i64 %[[ARR_SECT_SIZE1]], 4
// CHECK: %[[LFULL_ARR:.*]] = load ptr, ptr @_QFEfull_arr, align 8
// CHECK: %[[FULL_ARR_PTR:.*]] = getelementptr inbounds float, ptr %[[LFULL_ARR]], i64 0
// CHECK: %[[ARR_SECT_OFFSET1:.*]] = mul i64 %[[ARR_SECT_OFFSET2]], 1
// CHECK: %[[LARR_SECT:.*]] = load ptr, ptr @_QFEsect_arr, align 8
// CHECK: %[[ARR_SECT_PTR:.*]] = getelementptr inbounds i32, ptr %[[LARR_SECT]], i64 %[[ARR_SECT_OFFSET1]]
// CHECK: %[[SCALAR_PTR_LOAD:.*]] = load ptr, ptr %[[SCALAR_BASE]], align 8
// CHECK: %[[FULL_ARR_DESC_SIZE:.*]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEfull_arr, i32 1) to i64), i64 ptrtoint (ptr @_QFEfull_arr to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[ARR_SECT_DESC_SIZE:.*]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr @_QFEsect_arr, i32 1) to i64), i64 ptrtoint (ptr @_QFEsect_arr to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: %[[SCALAR_DESC_SZ4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[SCALAR_ALLOCA]], i32 1
// CHECK: %[[SCALAR_DESC_SZ3:.*]] = ptrtoint ptr %[[SCALAR_DESC_SZ4]] to i64
// CHECK: %[[SCALAR_DESC_SZ2:.*]] = ptrtoint ptr %[[SCALAR_ALLOCA]] to i64
// CHECK: %[[SCALAR_DESC_SZ1:.*]] = sub i64 %[[SCALAR_DESC_SZ3]], %[[SCALAR_DESC_SZ2]]
// CHECK: %[[SCALAR_DESC_SZ:.*]] = sdiv exact i64 %[[SCALAR_DESC_SZ1]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADPTRS]], align 8

// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK: store i64 %[[FULL_ARR_DESC_SIZE]], ptr %[[OFFLOADSIZES]], align 8

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK: store ptr @_QFEfull_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK: store ptr %[[FULL_ARR_PTR]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK: store i64 %[[FULL_ARR_SIZE]], ptr %[[OFFLOADSIZES]], align 8

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK: store i64 %[[ARR_SECT_DESC_SIZE]], ptr %[[OFFLOADSIZES]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 4
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
// CHECK: store ptr @_QFEsect_arr, ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 5
// CHECK: store ptr %[[ARR_SECT_PTR]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 5
// CHECK: store i64 %[[ARR_SECT_SIZE]], ptr %[[OFFLOADSIZES]], align 8

// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 6
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADSIZES:.*]] = getelementptr inbounds [9 x i64], ptr %.offload_sizes, i32 0, i32 6
// CHECK: store i64 %[[SCALAR_DESC_SZ]], ptr %[[OFFLOADSIZES]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 7
// CHECK: store ptr %[[SCALAR_ALLOCA]], ptr %[[OFFLOADPTRS]], align 8
// CHECK: %[[OFFLOADBASEPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 8
// CHECK: store ptr %[[SCALAR_BASE]], ptr %[[OFFLOADBASEPTRS]], align 8
// CHECK: %[[OFFLOADPTRS:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 8
// CHECK: store ptr %[[SCALAR_PTR_LOAD]], ptr %[[OFFLOADPTRS]], align 8
