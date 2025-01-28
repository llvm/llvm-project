// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @_QPopenmp_target_data() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data map_entries(%2 : !llvm.ptr) {
    %3 = llvm.mlir.constant(99 : i32) : i32
    llvm.store %3, %1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-LABEL: define void @_QPopenmp_target_data() {
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_5:.*]]
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         store i32 99, ptr %[[VAL_3]], align 4
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_11]], ptr %[[VAL_12]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_data_region(%0 : !llvm.ptr) {
  %1 = llvm.mlir.constant(1023 : index) : i64
  %2 = llvm.mlir.constant(0 : index) : i64
  %3 = llvm.mlir.constant(1024 : index) : i64
  %4 = llvm.mlir.constant(1 : index) : i64
  %5 = omp.map.bounds   lower_bound(%2 : i64) upper_bound(%1 : i64) extent(%3 : i64) stride(%4 : i64) start_idx(%4 : i64)
  %6 = omp.map.info var_ptr(%0 : !llvm.ptr, !llvm.array<1024 x i32>)   map_clauses(from) capture(ByRef) bounds(%5)  -> !llvm.ptr {name = ""}
  omp.target_data map_entries(%6 : !llvm.ptr) {
    %7 = llvm.mlir.constant(99 : i32) : i32
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.getelementptr %0[0, %10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x i32>
    llvm.store %7, %11 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4096]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK-LABEL: define void @_QPopenmp_target_data_region
// CHECK:         (ptr %[[ARG_0:.*]]) {
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_4:.*]]
// CHECK:         %[[ARR_OFFSET:.*]] = getelementptr inbounds [1024 x i32], ptr %[[ARR_DATA:.*]], i64 0, i64 0
// CHECK:         %[[VAL_5:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[ARR_DATA]], ptr %[[VAL_5]], align 8
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[ARR_OFFSET]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_8]], ptr %[[VAL_9]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_10:.*]] = getelementptr [1024 x i32], ptr %[[ARR_DATA]], i32 0, i64 0
// CHECK:         store i32 99, ptr %[[VAL_10]], align 4
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_11]], ptr %[[VAL_12]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPomp_target_enter_exit(%1 : !llvm.ptr, %3 : !llvm.ptr) {
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x i32 {bindc_name = "dvc", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEdvc"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "i", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEi"} : (i64) -> !llvm.ptr
  %8 = llvm.mlir.constant(5 : i32) : i32
  llvm.store %8, %7 : i32, !llvm.ptr
  %9 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %9, %5 : i32, !llvm.ptr
  %10 = llvm.load %7 : !llvm.ptr -> i32
  %11 = llvm.mlir.constant(10 : i32) : i32
  %12 = llvm.icmp "slt" %10, %11 : i32
  %13 = llvm.load %5 : !llvm.ptr -> i32
  %14 = llvm.mlir.constant(1023 : index) : i64
  %15 = llvm.mlir.constant(0 : index) : i64
  %16 = llvm.mlir.constant(1024 : index) : i64
  %17 = llvm.mlir.constant(1 : index) : i64
  %18 = omp.map.bounds   lower_bound(%15 : i64) upper_bound(%14 : i64) extent(%16 : i64) stride(%17 : i64) start_idx(%17 : i64)
  %map1 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.array<1024 x i32>)   map_clauses(to) capture(ByRef) bounds(%18) -> !llvm.ptr {name = ""}
  %19 = llvm.mlir.constant(511 : index) : i64
  %20 = llvm.mlir.constant(0 : index) : i64
  %21 = llvm.mlir.constant(512 : index) : i64
  %22 = llvm.mlir.constant(1 : index) : i64
  %23 = omp.map.bounds   lower_bound(%20 : i64) upper_bound(%19 : i64) extent(%21 : i64) stride(%22 : i64) start_idx(%22 : i64)
  %map2 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.array<512 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%23) -> !llvm.ptr {name = ""}
  omp.target_enter_data   if(%12) device(%13 : i32) map_entries(%map1, %map2 : !llvm.ptr, !llvm.ptr)
  %24 = llvm.load %7 : !llvm.ptr -> i32
  %25 = llvm.mlir.constant(10 : i32) : i32
  %26 = llvm.icmp "sgt" %24, %25 : i32
  %27 = llvm.load %5 : !llvm.ptr -> i32
  %28 = llvm.mlir.constant(1023 : index) : i64
  %29 = llvm.mlir.constant(0 : index) : i64
  %30 = llvm.mlir.constant(1024 : index) : i64
  %31 = llvm.mlir.constant(1 : index) : i64
  %32 = omp.map.bounds   lower_bound(%29 : i64) upper_bound(%28 : i64) extent(%30 : i64) stride(%31 : i64) start_idx(%31 : i64)
  %map3 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.array<1024 x i32>)   map_clauses(from) capture(ByRef) bounds(%32) -> !llvm.ptr {name = ""}
  %33 = llvm.mlir.constant(511 : index) : i64
  %34 = llvm.mlir.constant(0 : index) : i64
  %35 = llvm.mlir.constant(512 : index) : i64
  %36 = llvm.mlir.constant(1 : index) : i64
  %37 = omp.map.bounds   lower_bound(%34 : i64) upper_bound(%33 : i64) extent(%35 : i64) stride(%36 : i64) start_idx(%36 : i64)
  %map4 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.array<512 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%37) -> !llvm.ptr {name = ""}
  omp.target_exit_data   if(%26) device(%27 : i32) map_entries(%map3, %map4 : !llvm.ptr, !llvm.ptr)
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 4096, i64 2048]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 1, i64 0]
// CHECK:         @.offload_sizes.1 = private unnamed_addr constant [2 x i64] [i64 4096, i64 2048]
// CHECK:         @.offload_maptypes.2 = private unnamed_addr constant [2 x i64] [i64 2, i64 0]
// CHECK-LABEL: define void @_QPomp_target_enter_exit
// CHECK:         (ptr %[[ARG_0:.*]], ptr %[[ARG_1:.*]]) {
// CHECK:         %[[VAL_0:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_4:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_5:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_6:.*]] = alloca i32, i64 1, align 4
// CHECK:         %[[VAL_7:.*]] = alloca i32, i64 1, align 4
// CHECK:         store i32 5, ptr %[[VAL_7]], align 4
// CHECK:         store i32 2, ptr %[[VAL_6]], align 4
// CHECK:         %[[VAL_8:.*]] = load i32, ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_9:.*]] = icmp slt i32 %[[VAL_8]], 10
// CHECK:         %[[VAL_10:.*]] = load i32, ptr %[[VAL_6]], align 4
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_12:.*]]
// CHECK:         br i1 %[[VAL_9]], label %[[VAL_13:.*]], label %[[VAL_14:.*]]
// CHECK:       omp_if.then:                                      ; preds = %[[VAL_11]]
// CHECK:         %[[ARR_OFFSET1:.*]] = getelementptr inbounds [1024 x i32], ptr %[[VAL_16:.*]], i64 0, i64 0
// CHECK:         %[[ARR_OFFSET2:.*]] = getelementptr inbounds [512 x i32], ptr %[[VAL_20:.*]], i64 0, i64 0
// CHECK:         %[[VAL_15:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_15]], align 8
// CHECK:         %[[VAL_17:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 0
// CHECK:         store ptr %[[ARR_OFFSET1]], ptr %[[VAL_17]], align 8
// CHECK:         %[[VAL_18:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_5]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_18]], align 8
// CHECK:         %[[VAL_19:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_19]], align 8
// CHECK:         %[[VAL_21:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 1
// CHECK:         store ptr %[[ARR_OFFSET2]], ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_22:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_5]], i64 0, i64 1
// CHECK:         store ptr null, ptr %[[VAL_22]], align 8
// CHECK:         %[[VAL_23:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 0
// CHECK:         %[[VAL_24:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_23]], ptr %[[VAL_24]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp_if.else:                                      ; preds = %[[VAL_11]]
// CHECK:         br label %[[VAL_25]]
// CHECK:       omp_if.end:                                       ; preds = %[[VAL_14]], %[[VAL_13]]
// CHECK:         %[[VAL_26:.*]] = load i32, ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_27:.*]] = icmp sgt i32 %[[VAL_26]], 10
// CHECK:         %[[VAL_28:.*]] = load i32, ptr %[[VAL_6]], align 4
// CHECK:         br i1 %[[VAL_27]], label %[[VAL_29:.*]], label %[[VAL_30:.*]]
// CHECK:       omp_if.then2:                                     ; preds = %[[VAL_25]]
// CHECK:         %[[ARR_OFFSET3:.*]] = getelementptr inbounds [1024 x i32], ptr %[[VAL_16]], i64 0, i64 0
// CHECK:         %[[ARR_OFFSET4:.*]] = getelementptr inbounds [512 x i32], ptr %[[VAL_20]], i64 0, i64 0
// CHECK:         %[[VAL_31:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_31]], align 8
// CHECK:         %[[VAL_32:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[ARR_OFFSET3]], ptr %[[VAL_32]], align 8
// CHECK:         %[[VAL_33:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_33]], align 8
// CHECK:         %[[VAL_34:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_34]], align 8
// CHECK:         %[[VAL_35:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 1
// CHECK:         store ptr %[[ARR_OFFSET4]], ptr %[[VAL_35]], align 8
// CHECK:         %[[VAL_36:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 1
// CHECK:         store ptr null, ptr %[[VAL_36]], align 8
// CHECK:         %[[VAL_37:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_38:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_37]], ptr %[[VAL_38]], ptr @.offload_sizes.1, ptr @.offload_maptypes.2, ptr @.offload_mapnames.3, ptr null)
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp_if.else8:                                     ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_39]]
// CHECK:       omp_if.end9:                                      ; preds = %[[VAL_30]], %[[VAL_29]]
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_ptr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %map1 = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  %map2 = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data  map_entries(%map1 : !llvm.ptr) use_device_ptr(%map2 -> %arg0 : !llvm.ptr)  {
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %2 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 8]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 66]
// CHECK-LABEL: define void @_QPopenmp_target_use_dev_ptr
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_4:.*]] = alloca ptr, i64 1, align 8
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_6:.*]]
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_10]], ptr %[[VAL_11]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_12:.*]] = load ptr, ptr %[[VAL_7]], align 8
// CHECK:         store ptr %[[VAL_12]], ptr %[[VAL_3]], align 8
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_3]], align 8
// CHECK:         store i32 10, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_15:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_14]], ptr %[[VAL_15]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %map = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  %map2 = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr) use_device_addr(%map2 -> %arg0 : !llvm.ptr)  {
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %2 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 8]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 66]
// CHECK-LABEL: define void @_QPopenmp_target_use_dev_addr
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca ptr, i64 1, align 8
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_5:.*]]
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_12:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         store i32 10, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_13]], ptr %[[VAL_14]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr_no_ptr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  %map = omp.map.info var_ptr(%a : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %map2 = omp.map.info var_ptr(%a : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr) use_device_addr(%map2 -> %arg0 : !llvm.ptr)  {
    %1 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %1, %arg0 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 67]
// CHECK-LABEL: define void @_QPopenmp_target_use_dev_addr_no_ptr
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_5:.*]]
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_6]], align 8
// CHECK:         store i32 10, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %[[VAL_12]], ptr %[[VAL_13]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr_nomap() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %1 = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %map = omp.map.info var_ptr(%b : !llvm.ptr, !llvm.ptr)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  %map2 = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr) use_device_addr(%map2 -> %arg0 : !llvm.ptr)  {
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    llvm.store %2, %3 : i32, !llvm.ptr
    %4 = llvm.mlir.constant(20 : i32) : i32
    %5 = llvm.load %b : !llvm.ptr -> !llvm.ptr
    llvm.store %4, %5 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 8, i64 0]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 2, i64 64]
// CHECK-LABEL: define void @_QPopenmp_target_use_dev_addr_nomap
// CHECK:         %[[VAL_0:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca ptr, i64 1, align 8
// CHECK:         %[[VAL_4:.*]] = alloca ptr, i64 1, align 8
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_6:.*]]
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_10]], align 8
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 1
// CHECK:         store ptr null, ptr %[[VAL_12]], align 8
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_13]], ptr %[[VAL_14]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_15:.*]] = load ptr, ptr %[[VAL_10]], align 8
// CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_15]], align 8
// CHECK:         store i32 10, ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_4]], align 8
// CHECK:         store i32 20, ptr %[[VAL_17]], align 4
// CHECK:         %[[VAL_18:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_19:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_18]], ptr %[[VAL_19]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_both() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %1 = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
  %map = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %map1 = omp.map.info var_ptr(%b : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %map2 = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %map3 = omp.map.info var_ptr(%b : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data  map_entries(%map, %map1 : !llvm.ptr, !llvm.ptr) use_device_addr(%map3 -> %arg0 : !llvm.ptr) use_device_ptr(%map2 -> %arg1 : !llvm.ptr)  {
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    llvm.store %2, %3 : i32, !llvm.ptr
    %4 = llvm.mlir.constant(20 : i32) : i32
    %5 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    llvm.store %4, %5 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 8, i64 8]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 67, i64 67]
// CHECK-LABEL: define void @_QPopenmp_target_use_dev_both
// CHECK:         %[[VAL_0:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_4:.*]] = alloca ptr, i64 1, align 8
// CHECK:         %[[VAL_5:.*]] = alloca ptr, i64 1, align 8
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_7:.*]]
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_10]], align 8
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_5]], ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_5]], ptr %[[VAL_12]], align 8
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 1
// CHECK:         store ptr null, ptr %[[VAL_13]], align 8
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_15:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 2, ptr %[[VAL_14]], ptr %[[VAL_15]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_8]], align 8
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_3]], align 8
// CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_18:.*]] = load ptr, ptr %[[VAL_3]], align 8
// CHECK:         store i32 10, ptr %[[VAL_18]], align 4
// CHECK:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_17]], align 8
// CHECK:         store i32 20, ptr %[[VAL_19]], align 4
// CHECK:         %[[VAL_20:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_21:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @{{.*}}, i64 -1, i32 2, ptr %[[VAL_20]], ptr %[[VAL_21]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_data_update() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}
  omp.target_data map_entries(%2 : !llvm.ptr) {
    %3 = llvm.mlir.constant(99 : i32) : i32
    llvm.store %3, %1 : i32, !llvm.ptr
    omp.terminator
  }

  omp.target_update map_entries(%2 : !llvm.ptr)

  llvm.return
}

// CHECK-LABEL: define void @_QPopenmp_target_data_update

// CHECK-DAG:     %[[OFFLOAD_BASEPTRS:.*]] = alloca [1 x ptr], align 8
// CHECK-DAG:     %[[OFFLOAD_PTRS:.*]] = alloca [1 x ptr], align 8
// CHECK-DAG:     %[[INT_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK-DAG:     %[[OFFLOAD_MAPPERS:.*]] = alloca [1 x ptr], align 8

// CHECK:         call void @__tgt_target_data_begin_mapper
// CHECK:         store i32 99, ptr %[[INT_ALLOCA]], align 4
// CHECK:         call void @__tgt_target_data_end_mapper

// CHECK:         %[[BASEPTRS_VAL:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:         store ptr %[[INT_ALLOCA]], ptr %[[BASEPTRS_VAL]], align 8
// CHECK:         %[[PTRS_VAL:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK:         store ptr %[[INT_ALLOCA]], ptr %[[PTRS_VAL]], align 8
// CHECK:         %[[MAPPERS_VAL:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_MAPPERS]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[MAPPERS_VAL]], align 8
// CHECK:         %[[BASEPTRS_VAL_2:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:         %[[PTRS_VAL_2:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_update_mapper(ptr @2, i64 -1, i32 1, ptr %[[BASEPTRS_VAL_2]], ptr %[[PTRS_VAL_2]], ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr null)

// CHECK:         ret void

// -----

omp.declare_mapper @_QQFmy_testmy_mapper : !llvm.struct<"_QFmy_testTmy_type", (i32)> {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "var%data"}
  %3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>) map_clauses(tofrom) capture(ByRef) members(%2 : [0] : !llvm.ptr) -> !llvm.ptr {name = "var", partial_map = true}
  omp.declare_mapper_info map_entries(%3, %2 : !llvm.ptr, !llvm.ptr)
}

llvm.func @_QPopenmp_target_data_mapper() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.struct<"_QFmy_testTmy_type", (i32)> {bindc_name = "a"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>) mapper(@_QQFmy_testmy_mapper) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target_data map_entries(%2 : !llvm.ptr) {
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>
    llvm.store %3, %4 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-LABEL: define void @_QPopenmp_target_data_mapper
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_3:.*]] = alloca %[[VAL_4:.*]], i64 1, align 8
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_6:.*]]
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr @.omp_mapper._QQFmy_testmy_mapper, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @4, i64 -1, i32 1, ptr %[[VAL_10]], ptr %[[VAL_11]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr %[[VAL_2]])
// CHECK:         %[[VAL_12:.*]] = getelementptr %[[VAL_4]], ptr %[[VAL_3]], i32 0, i32 0
// CHECK:         store i32 10, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @4, i64 -1, i32 1, ptr %[[VAL_13]], ptr %[[VAL_14]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr %[[VAL_2]])
// CHECK:         ret void

// CHECK-LABEL: define internal void @.omp_mapper._QQFmy_testmy_mapper
// CHECK:       entry:
// CHECK:         %[[VAL_15:.*]] = udiv exact i64 %[[VAL_16:.*]], 4
// CHECK:         %[[VAL_17:.*]] = getelementptr %[[VAL_18:.*]], ptr %[[VAL_19:.*]], i64 %[[VAL_15]]
// CHECK:         %[[VAL_20:.*]] = icmp sgt i64 %[[VAL_15]], 1
// CHECK:         %[[VAL_21:.*]] = and i64 %[[VAL_22:.*]], 8
// CHECK:         %[[VAL_23:.*]] = icmp ne ptr %[[VAL_24:.*]], %[[VAL_19]]
// CHECK:         %[[VAL_25:.*]] = and i64 %[[VAL_22]], 16
// CHECK:         %[[VAL_26:.*]] = icmp ne i64 %[[VAL_25]], 0
// CHECK:         %[[VAL_27:.*]] = and i1 %[[VAL_23]], %[[VAL_26]]
// CHECK:         %[[VAL_28:.*]] = or i1 %[[VAL_20]], %[[VAL_27]]
// CHECK:         %[[VAL_29:.*]] = icmp eq i64 %[[VAL_21]], 0
// CHECK:         %[[VAL_30:.*]] = and i1 %[[VAL_28]], %[[VAL_29]]
// CHECK:         br i1 %[[VAL_30]], label %[[VAL_31:.*]], label %[[VAL_32:.*]]
// CHECK:       .omp.array..init:                                 ; preds = %[[VAL_33:.*]]
// CHECK:         %[[VAL_34:.*]] = mul nuw i64 %[[VAL_15]], 4
// CHECK:         %[[VAL_35:.*]] = and i64 %[[VAL_22]], -4
// CHECK:         %[[VAL_36:.*]] = or i64 %[[VAL_35]], 512
// CHECK:         call void @__tgt_push_mapper_component(ptr %[[VAL_37:.*]], ptr %[[VAL_24]], ptr %[[VAL_19]], i64 %[[VAL_34]], i64 %[[VAL_36]], ptr %[[VAL_38:.*]])
// CHECK:         br label %[[VAL_32]]
// CHECK:       omp.arraymap.head:                                ; preds = %[[VAL_31]], %[[VAL_33]]
// CHECK:         %[[VAL_39:.*]] = icmp eq ptr %[[VAL_19]], %[[VAL_17]]
// CHECK:         br i1 %[[VAL_39]], label %[[VAL_40:.*]], label %[[VAL_41:.*]]
// CHECK:       omp.arraymap.body:                                ; preds = %[[VAL_42:.*]], %[[VAL_32]]
// CHECK:         %[[VAL_43:.*]] = phi ptr [ %[[VAL_19]], %[[VAL_32]] ], [ %[[VAL_44:.*]], %[[VAL_42]] ]
// CHECK:         %[[VAL_45:.*]] = getelementptr %[[VAL_18]], ptr %[[VAL_43]], i32 0, i32 0
// CHECK:         %[[VAL_46:.*]] = call i64 @__tgt_mapper_num_components(ptr %[[VAL_37]])
// CHECK:         %[[VAL_47:.*]] = shl i64 %[[VAL_46]], 48
// CHECK:         %[[VAL_48:.*]] = add nuw i64 3, %[[VAL_47]]
// CHECK:         %[[VAL_49:.*]] = and i64 %[[VAL_22]], 3
// CHECK:         %[[VAL_50:.*]] = icmp eq i64 %[[VAL_49]], 0
// CHECK:         br i1 %[[VAL_50]], label %[[VAL_51:.*]], label %[[VAL_52:.*]]
// CHECK:       omp.type.alloc:                                   ; preds = %[[VAL_41]]
// CHECK:         %[[VAL_53:.*]] = and i64 %[[VAL_48]], -4
// CHECK:         br label %[[VAL_42]]
// CHECK:       omp.type.alloc.else:                              ; preds = %[[VAL_41]]
// CHECK:         %[[VAL_54:.*]] = icmp eq i64 %[[VAL_49]], 1
// CHECK:         br i1 %[[VAL_54]], label %[[VAL_55:.*]], label %[[VAL_56:.*]]
// CHECK:       omp.type.to:                                      ; preds = %[[VAL_52]]
// CHECK:         %[[VAL_57:.*]] = and i64 %[[VAL_48]], -3
// CHECK:         br label %[[VAL_42]]
// CHECK:       omp.type.to.else:                                 ; preds = %[[VAL_52]]
// CHECK:         %[[VAL_58:.*]] = icmp eq i64 %[[VAL_49]], 2
// CHECK:         br i1 %[[VAL_58]], label %[[VAL_59:.*]], label %[[VAL_42]]
// CHECK:       omp.type.from:                                    ; preds = %[[VAL_56]]
// CHECK:         %[[VAL_60:.*]] = and i64 %[[VAL_48]], -2
// CHECK:         br label %[[VAL_42]]
// CHECK:       omp.type.end:                                     ; preds = %[[VAL_59]], %[[VAL_56]], %[[VAL_55]], %[[VAL_51]]
// CHECK:         %[[VAL_61:.*]] = phi i64 [ %[[VAL_53]], %[[VAL_51]] ], [ %[[VAL_57]], %[[VAL_55]] ], [ %[[VAL_60]], %[[VAL_59]] ], [ %[[VAL_48]], %[[VAL_56]] ]
// CHECK:         call void @__tgt_push_mapper_component(ptr %[[VAL_37]], ptr %[[VAL_43]], ptr %[[VAL_45]], i64 4, i64 %[[VAL_61]], ptr @2)
// CHECK:         %[[VAL_44]] = getelementptr %[[VAL_18]], ptr %[[VAL_43]], i32 1
// CHECK:         %[[VAL_62:.*]] = icmp eq ptr %[[VAL_44]], %[[VAL_17]]
// CHECK:         br i1 %[[VAL_62]], label %[[VAL_63:.*]], label %[[VAL_41]]
// CHECK:       omp.arraymap.exit:                                ; preds = %[[VAL_42]]
// CHECK:         %[[VAL_64:.*]] = icmp sgt i64 %[[VAL_15]], 1
// CHECK:         %[[VAL_65:.*]] = and i64 %[[VAL_22]], 8
// CHECK:         %[[VAL_66:.*]] = icmp ne i64 %[[VAL_65]], 0
// CHECK:         %[[VAL_67:.*]] = and i1 %[[VAL_64]], %[[VAL_66]]
// CHECK:         br i1 %[[VAL_67]], label %[[VAL_68:.*]], label %[[VAL_40]]
// CHECK:       .omp.array..del:                                  ; preds = %[[VAL_63]]
// CHECK:         %[[VAL_69:.*]] = mul nuw i64 %[[VAL_15]], 4
// CHECK:         %[[VAL_70:.*]] = and i64 %[[VAL_22]], -4
// CHECK:         %[[VAL_71:.*]] = or i64 %[[VAL_70]], 512
// CHECK:         call void @__tgt_push_mapper_component(ptr %[[VAL_37]], ptr %[[VAL_24]], ptr %[[VAL_19]], i64 %[[VAL_69]], i64 %[[VAL_71]], ptr %[[VAL_38]])
// CHECK:         br label %[[VAL_40]]
// CHECK:       omp.done:                                         ; preds = %[[VAL_68]], %[[VAL_63]], %[[VAL_32]]
// CHECK:         ret void
