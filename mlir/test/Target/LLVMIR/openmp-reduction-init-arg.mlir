// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that the block argument to the initialization region of
// omp.declare_reduction gets mapped properly when translating to LLVMIR.

module {
  omp.declare_reduction @add_reduction_byref_box_Uxf64 : !llvm.ptr init {
  ^bb0(%arg0: !llvm.ptr):
// test usage of %arg0:
    %11 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    omp.yield(%arg0 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg0 : !llvm.ptr)
  }

  llvm.func internal @_QFPreduce(%arg0: !llvm.ptr {fir.bindc_name = "r"}, %arg1: !llvm.ptr {fir.bindc_name = "r2"}) attributes {sym_visibility = "private"} {
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.mlir.constant(10 : i32) : i32
  %10 = llvm.mlir.constant(0 : i32) : i32
  %83 = llvm.mlir.constant(1 : i64) : i64
  %84 = llvm.alloca %83 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
  %86 = llvm.mlir.constant(1 : i64) : i64
  %87 = llvm.alloca %86 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
// test multiple reduction variables to ensure they don't intefere with each other
// when inlining the reduction init region multiple times
    omp.parallel reduction(byref @add_reduction_byref_box_Uxf64 %84 -> %arg3, byref @add_reduction_byref_box_Uxf64 %87 -> %arg4 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @_QFPreduce
// CHECK:         %[[VAL_0:.*]] = alloca { ptr, ptr }, align 8
// CHECK:         %[[VAL_1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         %[[VAL_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_4:.*]]
// CHECK:         %[[VAL_5:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       omp_parallel:                                     ; preds = %[[VAL_3]]
// CHECK:         %[[VAL_7:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_1]], ptr %[[VAL_7]], align 8
// CHECK:         %[[VAL_8:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_2]], ptr %[[VAL_8]], align 8
// CHECK:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @_QFPreduce..omp_par, ptr %[[VAL_0]])
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.par.outlined.exit:                            ; preds = %[[VAL_6]]
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp.par.exit.split:                               ; preds = %[[VAL_9]]
// CHECK:         ret void
// CHECK:       [[PAR_ENTRY:omp.par.entry]]:
// CHECK:         %[[VAL_11:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_12:.*]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_14:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_12]], i32 0, i32 1
// CHECK:         %[[VAL_15:.*]] = load ptr, ptr %[[VAL_14]], align 8
// CHECK:         %[[VAL_16:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_17:.*]] = load i32, ptr %[[VAL_18:.*]], align 4
// CHECK:         store i32 %[[VAL_17]], ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_19:.*]] = load i32, ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_21:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_23:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_24:.*]] = alloca [2 x ptr], align 8
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[PAR_ENTRY]]
// CHECK:         br label %[[INIT_LABEL:.*]]
// CHECK: [[INIT_LABEL]]:
// CHECK:         %[[VAL_20:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[VAL_13]], align 8
// CHECK:         store ptr %[[VAL_13]], ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_22:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[VAL_15]], align 8
// CHECK:         store ptr %[[VAL_15]], ptr %[[VAL_23]], align 8
// CHECK:         br label %[[VAL_27:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[INIT_LABEL]]
// CHECK:         br label %[[VAL_28:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_27]]
// CHECK:         %[[VAL_29:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_24]], i64 0, i64 0
// CHECK:         store ptr %[[VAL_21]], ptr %[[VAL_29]], align 8
// CHECK:         %[[VAL_30:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_24]], i64 0, i64 1
// CHECK:         store ptr %[[VAL_23]], ptr %[[VAL_30]], align 8
// CHECK:         %[[VAL_31:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_32:.*]] = call i32 @__kmpc_reduce(ptr @1, i32 %[[VAL_31]], i32 2, i64 16, ptr %[[VAL_24]], ptr @.omp.reduction.func, ptr @.gomp_critical_user_.reduction.var)
// CHECK:         switch i32 %[[VAL_32]], label %[[VAL_33:.*]] [
// CHECK:           i32 1, label %[[VAL_34:.*]]
// CHECK:           i32 2, label %[[VAL_35:.*]]
// CHECK:         ]
// CHECK:       reduce.switch.atomic:                             ; preds = %[[VAL_28]]
// CHECK:         unreachable
// CHECK:       reduce.switch.nonatomic:                          ; preds = %[[VAL_28]]
// CHECK:         %[[VAL_36:.*]] = load ptr, ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_37:.*]] = load ptr, ptr %[[VAL_23]], align 8
// CHECK:         call void @__kmpc_end_reduce(ptr @1, i32 %[[VAL_31]], ptr @.gomp_critical_user_.reduction.var)
// CHECK:         br label %[[VAL_33]]
// CHECK:       reduce.finalize:                                  ; preds = %[[VAL_34]], %[[VAL_28]]
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_33]]
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp.par.outlined.exit.exitStub:                   ; preds = %[[VAL_38]]
// CHECK:         ret void
// CHECK:         %[[VAL_40:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_41:.*]], i64 0, i64 0
// CHECK:         %[[VAL_42:.*]] = load ptr, ptr %[[VAL_40]], align 8
// CHECK:         %[[VAL_43:.*]] = load ptr, ptr %[[VAL_42]], align 8
// CHECK:         %[[VAL_44:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_45:.*]], i64 0, i64 0
// CHECK:         %[[VAL_46:.*]] = load ptr, ptr %[[VAL_44]], align 8
// CHECK:         %[[VAL_47:.*]] = load ptr, ptr %[[VAL_46]], align 8
// CHECK:         %[[VAL_48:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_41]], i64 0, i64 1
// CHECK:         %[[VAL_49:.*]] = load ptr, ptr %[[VAL_48]], align 8
// CHECK:         %[[VAL_50:.*]] = load ptr, ptr %[[VAL_49]], align 8
// CHECK:         %[[VAL_51:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_45]], i64 0, i64 1
// CHECK:         %[[VAL_52:.*]] = load ptr, ptr %[[VAL_51]], align 8
// CHECK:         %[[VAL_53:.*]] = load ptr, ptr %[[VAL_52]], align 8
// CHECK:         ret void

