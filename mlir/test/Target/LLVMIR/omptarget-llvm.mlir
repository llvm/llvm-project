// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @_QPopenmp_target_data() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr<i32>
  %2 = omp.map_info var_ptr(%1 : !llvm.ptr<i32>)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr<i32> {name = ""}
  omp.target_data map_entries(%2 : !llvm.ptr<i32>) {
    %3 = llvm.mlir.constant(99 : i32) : i32
    llvm.store %3, %1 : !llvm.ptr<i32>
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

llvm.func @_QPopenmp_target_data_region(%1 : !llvm.ptr<array<1024 x i32>>) {
  %2 = omp.map_info var_ptr(%1 : !llvm.ptr<array<1024 x i32>>)   map_clauses(from) capture(ByRef) -> !llvm.ptr<array<1024 x i32>> {name = ""}
  omp.target_data map_entries(%2 : !llvm.ptr<array<1024 x i32>>) {
    %3 = llvm.mlir.constant(99 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.getelementptr %1[0, %6] : (!llvm.ptr<array<1024 x i32>>, i64) -> !llvm.ptr<i32>
    llvm.store %3, %7 : !llvm.ptr<i32>
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
// CHECK:         %[[VAL_5:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_6:.*]], ptr %[[VAL_5]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_6]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_8]], align 8
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = getelementptr [1024 x i32], ptr %[[VAL_6]], i32 0, i64 0
// CHECK:         store i32 99, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_12]], ptr %[[VAL_13]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPomp_target_enter_exit(%1 : !llvm.ptr<array<1024 x i32>>, %3 : !llvm.ptr<array<512 x i32>>) {
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x i32 {bindc_name = "dvc", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEdvc"} : (i64) -> !llvm.ptr<i32>
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "i", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEi"} : (i64) -> !llvm.ptr<i32>
  %8 = llvm.mlir.constant(5 : i32) : i32
  llvm.store %8, %7 : !llvm.ptr<i32>
  %9 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %9, %5 : !llvm.ptr<i32>
  %10 = llvm.load %7 : !llvm.ptr<i32>
  %11 = llvm.mlir.constant(10 : i32) : i32
  %12 = llvm.icmp "slt" %10, %11 : i32
  %13 = llvm.load %5 : !llvm.ptr<i32>
  %map1 = omp.map_info var_ptr(%1 : !llvm.ptr<array<1024 x i32>>)   map_clauses(to) capture(ByRef) -> !llvm.ptr<array<1024 x i32>> {name = ""}
  %map2 = omp.map_info var_ptr(%3 : !llvm.ptr<array<512 x i32>>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !llvm.ptr<array<512 x i32>> {name = ""}
  omp.target_enter_data   if(%12 : i1) device(%13 : i32) map_entries(%map1, %map2 : !llvm.ptr<array<1024 x i32>>, !llvm.ptr<array<512 x i32>>)
  %14 = llvm.load %7 : !llvm.ptr<i32>
  %15 = llvm.mlir.constant(10 : i32) : i32
  %16 = llvm.icmp "sgt" %14, %15 : i32
  %17 = llvm.load %5 : !llvm.ptr<i32>
  %map3 = omp.map_info var_ptr(%1 : !llvm.ptr<array<1024 x i32>>)   map_clauses(from) capture(ByRef) -> !llvm.ptr<array<1024 x i32>> {name = ""}
  %map4 = omp.map_info var_ptr(%3 : !llvm.ptr<array<512 x i32>>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !llvm.ptr<array<512 x i32>> {name = ""}
  omp.target_exit_data   if(%16 : i1) device(%17 : i32) map_entries(%map3, %map4 : !llvm.ptr<array<1024 x i32>>, !llvm.ptr<array<512 x i32>>)
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
// CHECK:         %[[VAL_15:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16:.*]], ptr %[[VAL_15]], align 8
// CHECK:         %[[VAL_17:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_17]], align 8
// CHECK:         %[[VAL_18:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_5]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_18]], align 8
// CHECK:         %[[VAL_19:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20:.*]], ptr %[[VAL_19]], align 8
// CHECK:         %[[VAL_21:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_21]], align 8
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
// CHECK:       omp_if.then1:                                     ; preds = %[[VAL_25]]
// CHECK:         %[[VAL_31:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_31]], align 8
// CHECK:         %[[VAL_32:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_32]], align 8
// CHECK:         %[[VAL_33:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 0
// CHECK:         store ptr null, ptr %[[VAL_33]], align 8
// CHECK:         %[[VAL_34:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_34]], align 8
// CHECK:         %[[VAL_35:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_35]], align 8
// CHECK:         %[[VAL_36:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_2]], i64 0, i64 1
// CHECK:         store ptr null, ptr %[[VAL_36]], align 8
// CHECK:         %[[VAL_37:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_38:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_37]], ptr %[[VAL_38]], ptr @.offload_sizes.1, ptr @.offload_maptypes.2, ptr @.offload_mapnames.3, ptr null)
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp_if.else5:                                     ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_39]]
// CHECK:       omp_if.end6:                                      ; preds = %[[VAL_30]], %[[VAL_29]]
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_ptr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %map1 = omp.map_info var_ptr(%a : !llvm.ptr<!llvm.ptr<i32>>)   map_clauses(from) capture(ByRef) -> !llvm.ptr<!llvm.ptr<i32>> {name = ""}
  omp.target_data  map_entries(%map1 : !llvm.ptr<!llvm.ptr<i32>>) use_device_ptr(%a : !llvm.ptr<!llvm.ptr<i32>>)  {
  ^bb0(%arg0: !llvm.ptr<!llvm.ptr<i32>>):
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.load %arg0 : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %1, %2 : !llvm.ptr<i32>
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
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_10]], ptr %[[VAL_11]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_12:.*]] = load ptr, ptr %[[VAL_7]], align 8
// CHECK:         store ptr %[[VAL_12]], ptr %[[VAL_3]], align 8
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_3]], align 8
// CHECK:         store i32 10, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_15:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_14]], ptr %[[VAL_15]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %map = omp.map_info var_ptr(%a : !llvm.ptr<!llvm.ptr<i32>>)   map_clauses(from) capture(ByRef) -> !llvm.ptr<!llvm.ptr<i32>> {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr<!llvm.ptr<i32>>) use_device_addr(%a : !llvm.ptr<!llvm.ptr<i32>>)  {
  ^bb0(%arg0: !llvm.ptr<!llvm.ptr<i32>>):
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.load %arg0 : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %1, %2 : !llvm.ptr<i32>
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
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_12:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         store i32 10, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_13]], ptr %[[VAL_14]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr_no_ptr() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr<i32> : (i64) -> !llvm.ptr<i32>
  %map = omp.map_info var_ptr(%a : !llvm.ptr<i32>)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr<i32> {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr<i32>) use_device_addr(%a : !llvm.ptr<i32>)  {
  ^bb0(%arg0: !llvm.ptr<i32>):
    %1 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %1, %arg0 : !llvm.ptr<i32>
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
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_6]], align 8
// CHECK:         store i32 10, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_12]], ptr %[[VAL_13]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_use_dev_addr_nomap() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %a = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %1 = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %map = omp.map_info var_ptr(%b : !llvm.ptr<!llvm.ptr<i32>>)   map_clauses(from) capture(ByRef) -> !llvm.ptr<!llvm.ptr<i32>> {name = ""}
  omp.target_data  map_entries(%map : !llvm.ptr<!llvm.ptr<i32>>) use_device_addr(%a : !llvm.ptr<!llvm.ptr<i32>>)  {
  ^bb0(%arg0: !llvm.ptr<!llvm.ptr<i32>>):
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.load %arg0 : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %2, %3 : !llvm.ptr<i32>
    %4 = llvm.mlir.constant(20 : i32) : i32
    %5 = llvm.load %b : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %4, %5 : !llvm.ptr<i32>
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
  %a = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %1 = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.alloca %0 x !llvm.ptr<!llvm.ptr<i32>> : (i64) -> !llvm.ptr<!llvm.ptr<i32>>
  %map = omp.map_info var_ptr(%a : !llvm.ptr<!llvm.ptr<i32>>)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr<!llvm.ptr<i32>> {name = ""}
  %map1 = omp.map_info var_ptr(%b : !llvm.ptr<!llvm.ptr<i32>>)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr<!llvm.ptr<i32>> {name = ""}
  omp.target_data  map_entries(%map, %map1 : !llvm.ptr<!llvm.ptr<i32>>, !llvm.ptr<!llvm.ptr<i32>>) use_device_ptr(%a : !llvm.ptr<!llvm.ptr<i32>>) use_device_addr(%b : !llvm.ptr<!llvm.ptr<i32>>)  {
  ^bb0(%arg0: !llvm.ptr<!llvm.ptr<i32>>, %arg1: !llvm.ptr<!llvm.ptr<i32>>):
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.load %arg0 : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %2, %3 : !llvm.ptr<i32>
    %4 = llvm.mlir.constant(20 : i32) : i32
    %5 = llvm.load %arg1 : !llvm.ptr<!llvm.ptr<i32>>
    llvm.store %4, %5 : !llvm.ptr<i32>
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
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_14]], ptr %[[VAL_15]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_8]], align 8
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_3]], align 8
// CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_18:.*]] = load ptr, ptr %[[VAL_3]], align 8
// CHECK:         store i32 10, ptr %[[VAL_18]], align 4
// CHECK:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_17]], align 8
// CHECK:         store i32 20, ptr %[[VAL_19]], align 4
// CHECK:         %[[VAL_20:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_21:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_20]], ptr %[[VAL_21]], ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----
