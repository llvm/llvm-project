// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression check for a taskloop with private variables but none of the
// private variables go into the context struct.

omp.private {type = private} @_QFtestEi_private_i32 : i32
omp.private {type = private} @_QFtestEt2_private_i32 : i32
omp.private {type = private} @_QFtestEt1_private_i32 : i32
llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(20 : i32) : i32
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "t2"} : (i64) -> !llvm.ptr
  %4 = llvm.alloca %2 x i32 {bindc_name = "t1"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.taskloop private(@_QFtestEt1_private_i32 %4 -> %arg0, @_QFtestEt2_private_i32 %3 -> %arg1, @_QFtestEi_private_i32 %5 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg3) : i32 = (%0) to (%1) inclusive step (%0) {
      llvm.store %arg3, %arg2 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}
// CHECK-LABEL: define void @_QPtest() {
// CHECK:         %[[STRUCTARG:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK:         %[[VAL_0:.*]] = alloca i32, i64 1, align 4
// CHECK:         %[[VAL_1:.*]] = alloca i32, i64 1, align 4
// CHECK:         %[[VAL_2:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_4:.*]]
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_3]]
// CHECK:         %[[VAL_6:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({}, ptr null, i32 1) to i64))
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_5]]
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.taskloop.start:                               ; preds = %[[VAL_7]]
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       codeRepl:                                         ; preds = %[[VAL_8]]
// CHECK:         %[[VAL_10:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CHECK:         store i64 1, ptr %[[VAL_10]], align 4
// CHECK:         %[[VAL_11:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// CHECK:         store i64 20, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 2
// CHECK:         store i64 1, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 3
// CHECK:         store ptr %[[VAL_6]], ptr %[[VAL_13]], align 8
// CHECK:         %[[VAL_14:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_taskgroup(ptr @1, i32 %[[VAL_14]])
// CHECK:         %[[VAL_15:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_14]], i32 1, i64 40, i64 32, ptr @_QPtest..omp_par)
// CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_15]], align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_16]], ptr align 1 %[[STRUCTARG]], i64 32, i1 false)
// CHECK:         %[[VAL_17:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_16]], i32 0, i32 0
// CHECK:         %[[VAL_18:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_16]], i32 0, i32 1
// CHECK:         %[[VAL_19:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_16]], i32 0, i32 2
// CHECK:         %[[VAL_20:.*]] = load i64, ptr %[[VAL_19]], align 4
// CHECK:         call void @__kmpc_taskloop(ptr @1, i32 %[[VAL_14]], ptr %[[VAL_15]], i32 1, ptr %[[VAL_17]], ptr %[[VAL_18]], i64 %[[VAL_20]], i32 1, i32 0, i64 0, ptr @omp_taskloop_dup)
// CHECK:         call void @__kmpc_end_taskgroup(ptr @1, i32 %[[VAL_14]])
// CHECK:         br label %[[VAL_21:.*]]
// CHECK:       taskloop.exit:                                    ; preds = %[[VAL_9]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @_QPtest..omp_par
// CHECK:       taskloop.alloca:
// CHECK:         %[[VAL_22:.*]] = load ptr, ptr %[[VAL_23:.*]], align 8
// CHECK:         %[[VAL_24:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_22]], i32 0, i32 0
// CHECK:         %[[VAL_25:.*]] = load i64, ptr %[[VAL_24]], align 4
// CHECK:         %[[VAL_26:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_22]], i32 0, i32 1
// CHECK:         %[[VAL_27:.*]] = load i64, ptr %[[VAL_26]], align 4
// CHECK:         %[[VAL_28:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_22]], i32 0, i32 2
// CHECK:         %[[VAL_29:.*]] = load i64, ptr %[[VAL_28]], align 4
// CHECK:         %[[VAL_30:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_22]], i32 0, i32 3
// CHECK:         %[[VAL_31:.*]] = load ptr, ptr %[[VAL_30]], align 8, !align !1
// CHECK:         %[[VAL_32:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_33:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_34:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_35:.*]]
// CHECK:       taskloop.body:                                    ; preds = %[[VAL_36:.*]]
// CHECK:         br label %[[VAL_37:.*]]
// CHECK:       omp.taskloop.region:                              ; preds = %[[VAL_35]]
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_37]]
// CHECK:         %[[VAL_39:.*]] = sub i64 %[[VAL_27]], %[[VAL_25]]
// CHECK:         %[[VAL_40:.*]] = sdiv i64 %[[VAL_39]], 1
// CHECK:         %[[VAL_41:.*]] = add i64 %[[VAL_40]], 1
// CHECK:         %[[VAL_42:.*]] = trunc i64 %[[VAL_41]] to i32
// CHECK:         %[[VAL_43:.*]] = trunc i64 %[[VAL_25]] to i32
// CHECK:         br label %[[VAL_44:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_45:.*]], %[[VAL_38]]
// CHECK:         %[[VAL_46:.*]] = phi i32 [ 0, %[[VAL_38]] ], [ %[[VAL_47:.*]], %[[VAL_45]] ]
// CHECK:         br label %[[VAL_48:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_44]]
// CHECK:         %[[VAL_49:.*]] = icmp ult i32 %[[VAL_46]], %[[VAL_42]]
// CHECK:         br i1 %[[VAL_49]], label %[[VAL_50:.*]], label %[[VAL_51:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_48]]
// CHECK:         br label %[[VAL_52:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_51]]
// CHECK:         br label %[[VAL_53:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_52]]
// CHECK:         tail call void @free(ptr %[[VAL_31]])
// CHECK:         br label %[[VAL_54:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_48]]
// CHECK:         %[[VAL_55:.*]] = mul i32 %[[VAL_46]], 1
// CHECK:         %[[VAL_56:.*]] = add i32 %[[VAL_55]], %[[VAL_43]]
// CHECK:         br label %[[VAL_57:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_50]]
// CHECK:         store i32 %[[VAL_56]], ptr %[[VAL_34]], align 4
// CHECK:         br label %[[VAL_58:.*]]
// CHECK:       omp.region.cont3:                                 ; preds = %[[VAL_57]]
// CHECK:         br label %[[VAL_45]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_58]]
// CHECK:         %[[VAL_47]] = add nuw i32 %[[VAL_46]], 1
// CHECK:         br label %[[VAL_44]]
// CHECK:       taskloop.exit.exitStub:                           ; preds = %[[VAL_53]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @omp_taskloop_dup(
// CHECK:       entry:
// CHECK:         %[[VAL_59:.*]] = getelementptr { %[[VAL_60:.*]], { i64, i64, i64, ptr } }, ptr %[[VAL_61:.*]], i32 0, i32 1
// CHECK:         %[[VAL_62:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_59]], i32 0, i32 3
// CHECK:         %[[VAL_63:.*]] = getelementptr { %[[VAL_60]], { i64, i64, i64, ptr } }, ptr %[[VAL_64:.*]], i32 0, i32 1
// CHECK:         %[[VAL_65:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_63]], i32 0, i32 3
// CHECK:         %[[VAL_66:.*]] = load ptr, ptr %[[VAL_65]], align 8
// TODO: don't generate allocation for empty task context struct (for later patch)
// CHECK:         %[[VAL_67:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({}, ptr null, i32 1) to i64))
// CHECK:         store ptr %[[VAL_67]], ptr %[[VAL_62]], align 8
// CHECK:         ret void

