// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @_QPopenmp_target_data() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr<i32>
  omp.target_data   map((tofrom -> %1 : !llvm.ptr<i32>)) {
    %2 = llvm.mlir.constant(99 : i32) : i32
    llvm.store %2, %1 : !llvm.ptr<i32>
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-LABEL: define void @_QPopenmp_target_data() {
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x i64], align 8
// CHECK:         %[[VAL_3:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       [[VAL_4]]:
// CHECK:         %[[VAL_5:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_5]], align 8
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_8]], ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         store i32 99, ptr %[[VAL_3]], align 4
// CHECK:         %[[VAL_11:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_11]], ptr %[[VAL_12]], ptr %[[VAL_13]], ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPopenmp_target_data_region(%1 : !llvm.ptr<array<1024 x i32>>) {
  omp.target_data   map((from -> %1 : !llvm.ptr<array<1024 x i32>>)) {
    %2 = llvm.mlir.constant(99 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.getelementptr %1[0, %5] : (!llvm.ptr<array<1024 x i32>>, i64) -> !llvm.ptr<i32>
    llvm.store %2, %6 : !llvm.ptr<i32>
    omp.terminator
  }
  llvm.return
}

// CHECK:         @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK-LABEL: define void @_QPopenmp_target_data_region
// CHECK:         (ptr %[[ARG_0:.*]]) {
// CHECK:         %[[VAL_0:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [1 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [1 x i64], align 8
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       [[VAL_3]]:
// CHECK:         %[[VAL_4:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_5:.*]], ptr %[[VAL_4]], align 8
// CHECK:         %[[VAL_6:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_5]], ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_8:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_9:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         %[[VAL_10:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_8]], ptr %[[VAL_9]], ptr %[[VAL_10]], ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         %[[VAL_11:.*]] = getelementptr [1024 x i32], ptr %[[VAL_5]], i32 0, i64 0
// CHECK:         store i32 99, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         %[[VAL_14:.*]] = getelementptr inbounds [1 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @2, i64 -1, i32 1, ptr %[[VAL_12]], ptr %[[VAL_13]], ptr %[[VAL_14]], ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         ret void

// -----

llvm.func @_QPomp_target_enter_exit(%1 : !llvm.ptr<array<1024 x i32>>, %3 : !llvm.ptr<array<1024 x i32>>) {
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x i32 {bindc_name = "dvc", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEdvc"} : (i64) -> !llvm.ptr<i32>
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_enter_exitEi"} : (i64) -> !llvm.ptr<i32>
  %8 = llvm.mlir.constant(5 : i32) : i32
  llvm.store %8, %7 : !llvm.ptr<i32>
  %9 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %9, %5 : !llvm.ptr<i32>
  %10 = llvm.load %7 : !llvm.ptr<i32>
  %11 = llvm.mlir.constant(10 : i32) : i32
  %12 = llvm.icmp "slt" %10, %11 : i32
  %13 = llvm.load %5 : !llvm.ptr<i32>
  omp.target_enter_data   if(%12 : i1) device(%13 : i32) map((to -> %1 : !llvm.ptr<array<1024 x i32>>), (alloc -> %3 : !llvm.ptr<array<1024 x i32>>))
  %14 = llvm.load %7 : !llvm.ptr<i32>
  %15 = llvm.mlir.constant(10 : i32) : i32
  %16 = llvm.icmp "sgt" %14, %15 : i32
  %17 = llvm.load %5 : !llvm.ptr<i32>
  omp.target_exit_data   if(%16 : i1) device(%17 : i32) map((from -> %1 : !llvm.ptr<array<1024 x i32>>), (release -> %3 : !llvm.ptr<array<1024 x i32>>))
  llvm.return
}

// CHECK:         @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 1, i64 0]
// CHECK:         @.offload_maptypes.1 = private unnamed_addr constant [2 x i64] [i64 2, i64 0]
// CHECK-LABEL: define void @_QPomp_target_enter_exit
// CHECK:         (ptr %[[ARG_0:.*]], ptr %[[ARG_1:.*]]) {
// CHECK:         %[[VAL_0:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_1:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_2:.*]] = alloca [2 x i64], align 8
// CHECK:         %[[VAL_3:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_4:.*]] = alloca [2 x ptr], align 8
// CHECK:         %[[VAL_5:.*]] = alloca [2 x i64], align 8
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
// CHECK:         %[[VAL_18:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_5]], i32 0, i32 0
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_18]], align 4
// CHECK:         %[[VAL_19:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20:.*]], ptr %[[VAL_19]], align 8
// CHECK:         %[[VAL_21:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_22:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_5]], i32 0, i32 1
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_22]], align 4
// CHECK:         %[[VAL_23:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_3]], i32 0, i32 0
// CHECK:         %[[VAL_24:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_4]], i32 0, i32 0
// CHECK:         %[[VAL_25:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_5]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_begin_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_23]], ptr %[[VAL_24]], ptr %[[VAL_25]], ptr @.offload_maptypes, ptr @.offload_mapnames, ptr null)
// CHECK:         br label %[[VAL_14]]
// CHECK:       omp_if.end:                                       ; preds = %[[VAL_11]], %[[VAL_13]]
// CHECK:         %[[VAL_26:.*]] = load i32, ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_27:.*]] = icmp sgt i32 %[[VAL_26]], 10
// CHECK:         %[[VAL_28:.*]] = load i32, ptr %[[VAL_6]], align 4
// CHECK:         br i1 %[[VAL_27]], label %[[VAL_29:.*]], label %[[VAL_30:.*]]
// CHECK:       omp_if.then4:                                     ; preds = %[[VAL_14]]
// CHECK:         %[[VAL_31:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_31]], align 8
// CHECK:         %[[VAL_32:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_16]], ptr %[[VAL_32]], align 8
// CHECK:         %[[VAL_33:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_33]], align 4
// CHECK:         %[[VAL_34:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_34]], align 8
// CHECK:         %[[VAL_35:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_35]], align 8
// CHECK:         %[[VAL_36:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_2]], i32 0, i32 1
// CHECK:         store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[VAL_36]], align 4
// CHECK:         %[[VAL_37:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         %[[VAL_38:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_1]], i32 0, i32 0
// CHECK:         %[[VAL_39:.*]] = getelementptr inbounds [2 x i64], ptr %[[VAL_2]], i32 0, i32 0
// CHECK:         call void @__tgt_target_data_end_mapper(ptr @3, i64 -1, i32 2, ptr %[[VAL_37]], ptr %[[VAL_38]], ptr %[[VAL_39]], ptr @.offload_maptypes.1, ptr @.offload_mapnames.2, ptr null)
// CHECK:         br label %[[VAL_30]]
// CHECK:       omp_if.end5:                                      ; preds = %[[VAL_14]], %[[VAL_29]]
// CHECK:         ret void

// -----

