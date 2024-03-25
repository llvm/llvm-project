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
    omp.parallel {
      %83 = llvm.mlir.constant(1 : i64) : i64
      %84 = llvm.alloca %83 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
      %86 = llvm.mlir.constant(1 : i64) : i64
      %87 = llvm.alloca %86 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
// test multiple reduction variables to ensure they don't intefere with eachother
// when inlining the reduction init region multiple times
      omp.wsloop byref reduction(@add_reduction_byref_box_Uxf64 %84 -> %arg3 : !llvm.ptr, @add_reduction_byref_box_Uxf64 %87 -> %arg4 : !llvm.ptr)  for  (%arg2) : i32 = (%10) to (%9) inclusive step (%8) {
        omp.yield
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @_QFPreduce(ptr %{{.*}}, ptr %{{.*}})
// CHECK:         br label %entry
// CHECK:       entry:                                            ; preds = %[[VAL_1:.*]]
// CHECK:         %[[VAL_2:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       omp_parallel:                                     ; preds = %entry
// CHECK:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @_QFPreduce..omp_par)
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.par.outlined.exit:                            ; preds = %[[VAL_3]]
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       omp.par.exit.split:                               ; preds = %[[VAL_4]]
// CHECK:         ret void
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_6:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_7:.*]] = load i32, ptr %[[VAL_8:.*]], align 4
// CHECK:         store i32 %[[VAL_7]], ptr %[[VAL_6]], align 4
// CHECK:         %[[VAL_9:.*]] = load i32, ptr %[[VAL_6]], align 4
// CHECK:         %[[VAL_10:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_11:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_12:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_13:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_14:.*]] = alloca [2 x ptr], align 8
// CHECK:         br label %[[VAL_15:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_16:.*]]
// CHECK:         br label %[[VAL_17:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_15]]
// CHECK:         %[[VAL_18:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         %[[VAL_19:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         %[[VAL_20:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[VAL_18]], align 8
// CHECK:         %[[VAL_21:.*]] = alloca ptr, align 8
// CHECK:         store ptr %[[VAL_18]], ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_22:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[VAL_19]], align 8
// CHECK:         %[[VAL_23:.*]] = alloca ptr, align 8
// CHECK:         store ptr %[[VAL_19]], ptr %[[VAL_23]], align 8
// CHECK:         br label %[[VAL_24:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_17]]
// CHECK:         store i32 0, ptr %[[VAL_11]], align 4
// CHECK:         store i32 10, ptr %[[VAL_12]], align 4
// CHECK:         store i32 1, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_25:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_25]], i32 34, ptr %[[VAL_10]], ptr %[[VAL_11]], ptr %[[VAL_12]], ptr %[[VAL_13]], i32 1, i32 0)
// CHECK:         %[[VAL_26:.*]] = load i32, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_27:.*]] = load i32, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_28:.*]] = sub i32 %[[VAL_27]], %[[VAL_26]]
// CHECK:         %[[VAL_29:.*]] = add i32 %[[VAL_28]], 1
// CHECK:         br label %[[VAL_30:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_31:.*]], %[[VAL_24]]
// CHECK:         %[[VAL_32:.*]] = phi i32 [ 0, %[[VAL_24]] ], [ %[[VAL_33:.*]], %[[VAL_31]] ]
// CHECK:         br label %[[VAL_34:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_30]]
// CHECK:         %[[VAL_35:.*]] = icmp ult i32 %[[VAL_32]], %[[VAL_29]]
// CHECK:         br i1 %[[VAL_35]], label %[[VAL_36:.*]], label %[[VAL_37:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_34]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_25]])
// CHECK:         %[[VAL_38:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_38]])
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_37]]
// CHECK:         %[[VAL_40:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_14]], i64 0, i64 0
// CHECK:         store ptr %[[VAL_21]], ptr %[[VAL_40]], align 8
// CHECK:         %[[VAL_41:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_14]], i64 0, i64 1
// CHECK:         store ptr %[[VAL_23]], ptr %[[VAL_41]], align 8
// CHECK:         %[[VAL_42:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_43:.*]] = call i32 @__kmpc_reduce(ptr @1, i32 %[[VAL_42]], i32 2, i64 16, ptr %[[VAL_14]], ptr @.omp.reduction.func, ptr @.gomp_critical_user_.reduction.var)
// CHECK:         switch i32 %[[VAL_43]], label %[[VAL_44:.*]] [
// CHECK:           i32 1, label %[[VAL_45:.*]]
// CHECK:           i32 2, label %[[VAL_46:.*]]
// CHECK:         ]
// CHECK:       reduce.switch.atomic:                             ; preds = %[[VAL_39]]
// CHECK:         unreachable
// CHECK:       reduce.switch.nonatomic:                          ; preds = %[[VAL_39]]
// CHECK:         %[[VAL_47:.*]] = load ptr, ptr %[[VAL_21]], align 8
// CHECK:         %[[VAL_48:.*]] = load ptr, ptr %[[VAL_23]], align 8
// CHECK:         call void @__kmpc_end_reduce(ptr @1, i32 %[[VAL_42]], ptr @.gomp_critical_user_.reduction.var)
// CHECK:         br label %[[VAL_44]]
// CHECK:       reduce.finalize:                                  ; preds = %[[VAL_45]], %[[VAL_39]]
// CHECK:         %[[VAL_49:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_49]])
// CHECK:         br label %[[VAL_50:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_44]]
// CHECK:         br label %[[VAL_51:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_50]]
// CHECK:         br label %[[VAL_52:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_34]]
// CHECK:         %[[VAL_53:.*]] = add i32 %[[VAL_32]], %[[VAL_26]]
// CHECK:         %[[VAL_54:.*]] = mul i32 %[[VAL_53]], 1
// CHECK:         %[[VAL_55:.*]] = add i32 %[[VAL_54]], 0
// CHECK:         br label %[[VAL_56:.*]]
// CHECK:       omp.wsloop.region:                                ; preds = %[[VAL_36]]
// CHECK:         br label %[[VAL_57:.*]]
// CHECK:       omp.region.cont2:                                 ; preds = %[[VAL_56]]
// CHECK:         br label %[[VAL_31]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_57]]
// CHECK:         %[[VAL_33]] = add nuw i32 %[[VAL_32]], 1
// CHECK:         br label %[[VAL_30]]
// CHECK:       omp.par.outlined.exit.exitStub:                   ; preds = %[[VAL_51]]
// CHECK:         ret void
// CHECK:         %[[VAL_58:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_59:.*]], i64 0, i64 0
// CHECK:         %[[VAL_60:.*]] = load ptr, ptr %[[VAL_58]], align 8
// CHECK:         %[[VAL_61:.*]] = load ptr, ptr %[[VAL_60]], align 8
// CHECK:         %[[VAL_62:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_63:.*]], i64 0, i64 0
// CHECK:         %[[VAL_64:.*]] = load ptr, ptr %[[VAL_62]], align 8
// CHECK:         %[[VAL_65:.*]] = load ptr, ptr %[[VAL_64]], align 8
// CHECK:         %[[VAL_66:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_59]], i64 0, i64 1
// CHECK:         %[[VAL_67:.*]] = load ptr, ptr %[[VAL_66]], align 8
// CHECK:         %[[VAL_68:.*]] = load ptr, ptr %[[VAL_67]], align 8
// CHECK:         %[[VAL_69:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_63]], i64 0, i64 1
// CHECK:         %[[VAL_70:.*]] = load ptr, ptr %[[VAL_69]], align 8
// CHECK:         %[[VAL_71:.*]] = load ptr, ptr %[[VAL_70]], align 8
// CHECK:         ret void

