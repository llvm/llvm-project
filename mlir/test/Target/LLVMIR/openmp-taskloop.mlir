// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFtestEi_private_i32 : i32

omp.private {type = firstprivate} @_QFtestEa_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}


llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(5 : i32) : i32
  %9 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%9) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QPtest() {
// CHECK:         %[[STRUCTARG:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK:         %[[VAL_0:.*]] = alloca i32, i64 1, align 4
// CHECK:         %[[VAL_1:.*]] = alloca i32, i64 1, align 4
// CHECK:         store i32 20, ptr %[[VAL_1]], align 4
// CHECK:         br label %[[VAL_2:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_3:.*]]
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_2]]
// CHECK:         %[[VAL_5:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:         %[[VAL_6:.*]] = getelementptr { i32 }, ptr %[[VAL_5]], i32 0, i32 0
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_4]]
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.private.copy1:                                ; preds = %[[VAL_7]]
// CHECK:         %[[VAL_9:.*]] = load i32, ptr %[[VAL_1]], align 4
// CHECK:         store i32 %[[VAL_9]], ptr %[[VAL_6]], align 4
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp.taskloop.start:                               ; preds = %[[VAL_8]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       codeRepl:                                         ; preds = %[[VAL_10]]
// CHECK:         %[[VAL_12:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CHECK:         store i64 1, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// CHECK:         store i64 5, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_14:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 2
// CHECK:         store i64 1, ptr %[[VAL_14]], align 4
// CHECK:         %[[VAL_15:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 3
// CHECK:         store ptr %[[VAL_5]], ptr %[[VAL_15]], align 8
// CHECK:         %[[VAL_16:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_taskgroup(ptr @1, i32 %[[VAL_16]])
// CHECK:         %[[VAL_17:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_16]], i32 1, i64 40, i64 32, ptr @_QPtest..omp_par)
// CHECK:         %[[VAL_18:.*]] = load ptr, ptr %[[VAL_17]], align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_18]], ptr align 1 %[[STRUCTARG]], i64 32, i1 false)
// CHECK:         %[[VAL_19:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_18]], i32 0, i32 0
// CHECK:         %[[VAL_20:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_18]], i32 0, i32 1
// CHECK:         %[[VAL_21:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_18]], i32 0, i32 2
// CHECK:         %[[VAL_22:.*]] = load i64, ptr %[[VAL_21]], align 4
// CHECK:         call void @__kmpc_taskloop(ptr @1, i32 %[[VAL_16]], ptr %[[VAL_17]], i32 1, ptr %[[VAL_19]], ptr %[[VAL_20]], i64 %[[VAL_22]], i32 1, i32 0, i64 0, ptr @omp_taskloop_dup)
// CHECK:         call void @__kmpc_end_taskgroup(ptr @1, i32 %[[VAL_16]])
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       taskloop.exit:                                    ; preds = %[[VAL_11]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @_QPtest..omp_par(
// CHECK:       taskloop.alloca:
// CHECK:         %[[VAL_24:.*]] = load ptr, ptr %[[VAL_25:.*]], align 8
// CHECK:         %[[VAL_26:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_24]], i32 0, i32 0
// CHECK:         %[[VAL_27:.*]] = load i64, ptr %[[VAL_26]], align 4
// CHECK:         %[[VAL_28:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_24]], i32 0, i32 1
// CHECK:         %[[VAL_29:.*]] = load i64, ptr %[[VAL_28]], align 4
// CHECK:         %[[VAL_30:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_24]], i32 0, i32 2
// CHECK:         %[[VAL_31:.*]] = load i64, ptr %[[VAL_30]], align 4
// CHECK:         %[[VAL_32:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_24]], i32 0, i32 3
// CHECK:         %[[VAL_33:.*]] = load ptr, ptr %[[VAL_32]], align 8, !align !1
// CHECK:         %[[VAL_34:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_35:.*]]
// CHECK:       taskloop.body:                                    ; preds = %[[VAL_36:.*]]
// CHECK:         %[[VAL_37:.*]] = getelementptr { i32 }, ptr %[[VAL_33]], i32 0, i32 0
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp.taskloop.region:                              ; preds = %[[VAL_35]]
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_38]]
// CHECK:         %[[VAL_40:.*]] = sub i64 %[[VAL_29]], %[[VAL_27]]
// CHECK:         %[[VAL_41:.*]] = sdiv i64 %[[VAL_40]], 1
// CHECK:         %[[VAL_42:.*]] = add i64 %[[VAL_41]], 1
// CHECK:         %[[VAL_43:.*]] = trunc i64 %[[VAL_42]] to i32
// CHECK:         %[[VAL_44:.*]] = trunc i64 %[[VAL_27]] to i32
// CHECK:         br label %[[VAL_45:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_46:.*]], %[[VAL_39]]
// CHECK:         %[[VAL_47:.*]] = phi i32 [ 0, %[[VAL_39]] ], [ %[[VAL_48:.*]], %[[VAL_46]] ]
// CHECK:         br label %[[VAL_49:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_45]]
// CHECK:         %[[VAL_50:.*]] = icmp ult i32 %[[VAL_47]], %[[VAL_43]]
// CHECK:         br i1 %[[VAL_50]], label %[[VAL_51:.*]], label %[[VAL_52:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_49]]
// CHECK:         br label %[[VAL_53:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_52]]
// CHECK:         br label %[[VAL_54:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_53]]
// CHECK:         tail call void @free(ptr %[[VAL_33]])
// CHECK:         br label %[[VAL_55:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_49]]
// CHECK:         %[[VAL_56:.*]] = mul i32 %[[VAL_47]], 1
// CHECK:         %[[VAL_57:.*]] = add i32 %[[VAL_56]], %[[VAL_44]]
// CHECK:         br label %[[VAL_58:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_51]]
// CHECK:         store i32 %[[VAL_57]], ptr %[[VAL_34]], align 4
// CHECK:         %[[VAL_59:.*]] = load i32, ptr %[[VAL_37]], align 4
// CHECK:         %[[VAL_60:.*]] = add i32 %[[VAL_59]], 1
// CHECK:         store i32 %[[VAL_60]], ptr %[[VAL_37]], align 4
// CHECK:         br label %[[VAL_61:.*]]
// CHECK:       omp.region.cont2:                                 ; preds = %[[VAL_58]]
// CHECK:         br label %[[VAL_46]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_61]]
// CHECK:         %[[VAL_48]] = add nuw i32 %[[VAL_47]], 1
// CHECK:         br label %[[VAL_45]]
// CHECK:       taskloop.exit.exitStub:                           ; preds = %[[VAL_54]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @omp_taskloop_dup(
// CHECK:       entry:
// CHECK:         %[[VAL_62:.*]] = getelementptr { %[[VAL_63:.*]], { i64, i64, i64, ptr } }, ptr %[[VAL_64:.*]], i32 0, i32 1
// CHECK:         %[[VAL_65:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_62]], i32 0, i32 3
// CHECK:         %[[VAL_66:.*]] = getelementptr { %[[VAL_63]], { i64, i64, i64, ptr } }, ptr %[[VAL_67:.*]], i32 0, i32 1
// CHECK:         %[[VAL_68:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_66]], i32 0, i32 3
// CHECK:         %[[VAL_69:.*]] = load ptr, ptr %[[VAL_68]], align 8
// CHECK:         %[[VAL_70:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:         store ptr %[[VAL_70]], ptr %[[VAL_65]], align 8
// CHECK:         %[[VAL_71:.*]] = getelementptr { i32 }, ptr %[[VAL_69]], i32 0, i32 0
// CHECK:         %[[VAL_72:.*]] = getelementptr { i32 }, ptr %[[VAL_70]], i32 0, i32 0
// CHECK:         br label %[[VAL_73:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_74:.*]]
// CHECK:         %[[VAL_75:.*]] = load i32, ptr %[[VAL_71]], align 4
// CHECK:         store i32 %[[VAL_75]], ptr %[[VAL_72]], align 4
// CHECK:         ret void

