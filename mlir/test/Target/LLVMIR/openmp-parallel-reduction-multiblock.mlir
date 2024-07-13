// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This is basically a test that we don't crash while translating this IR

omp.declare_reduction @add_reduction_byref_box_heap_i32 : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr):
  %7 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  %17 = llvm.icmp "eq" %16, %7 : i64
  llvm.cond_br %17, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  omp.yield(%arg0 : !llvm.ptr)
} combiner {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}  cleanup {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}
llvm.func @missordered_blocks_(%arg0: !llvm.ptr {fir.bindc_name = "x"}, %arg1: !llvm.ptr {fir.bindc_name = "y"}) attributes {fir.internal_name = "_QPmissordered_blocks", frame_pointer = #llvm.framePointerKind<"non-leaf">, target_cpu = "generic", target_features = #llvm.target_features<["+outline-atomics", "+v8a", "+fp-armv8", "+neon"]>} {
  omp.parallel reduction(byref @add_reduction_byref_box_heap_i32 %arg0 -> %arg2 : !llvm.ptr, byref @add_reduction_byref_box_heap_i32 %arg1 -> %arg3 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK:         %[[VAL_0:.*]] = alloca { ptr, ptr }, align 8
// CHECK:         br label %[[VAL_1:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_2:.*]]
// CHECK:         %[[VAL_3:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp_parallel:                                     ; preds = %[[VAL_1]]
// CHECK:         %[[VAL_5:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_0]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_6:.*]], ptr %[[VAL_5]], align 8
// CHECK:         %[[VAL_7:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_0]], i32 0, i32 1
// CHECK:         store ptr %[[VAL_8:.*]], ptr %[[VAL_7]], align 8
// CHECK:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @missordered_blocks_..omp_par, ptr %[[VAL_0]])
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.par.outlined.exit:                            ; preds = %[[VAL_4]]
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp.par.exit.split:                               ; preds = %[[VAL_9]]
// CHECK:         ret void
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_11:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_12:.*]], i32 0, i32 0
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_14:.*]] = getelementptr { ptr, ptr }, ptr %[[VAL_12]], i32 0, i32 1
// CHECK:         %[[VAL_15:.*]] = load ptr, ptr %[[VAL_14]], align 8
// CHECK:         %[[VAL_16:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_17:.*]] = load i32, ptr %[[VAL_18:.*]], align 4
// CHECK:         store i32 %[[VAL_17]], ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_19:.*]] = load i32, ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_20:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_21:.*]] = alloca ptr, align 8
// CHECK:         %[[VAL_22:.*]] = alloca [2 x ptr], align 8
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       omp.reduction.init:                               ; preds = %[[VAL_24:.*]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.reduction.neutral:                            ; preds = %[[VAL_23]]
// CHECK:         %[[VAL_26:.*]] = ptrtoint ptr %[[VAL_13]] to i64
// CHECK:         %[[VAL_27:.*]] = icmp eq i64 %[[VAL_26]], 0
// CHECK:         br i1 %[[VAL_27]], label %[[VAL_28:.*]], label %[[VAL_29:.*]]
// CHECK:       omp.reduction.neutral2:                           ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_30:.*]]
// CHECK:       omp.reduction.neutral3:                           ; preds = %[[VAL_28]], %[[VAL_29]]
// CHECK:         br label %[[VAL_31:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_30]]
// CHECK:         %[[VAL_32:.*]] = phi ptr [ %[[VAL_13]], %[[VAL_30]] ]
// CHECK:         store ptr %[[VAL_32]], ptr %[[VAL_20]], align 8
// CHECK:         br label %[[VAL_33:.*]]
// CHECK:       omp.reduction.neutral5:                           ; preds = %[[VAL_31]]
// CHECK:         %[[VAL_34:.*]] = ptrtoint ptr %[[VAL_15]] to i64
// CHECK:         %[[VAL_35:.*]] = icmp eq i64 %[[VAL_34]], 0
// CHECK:         br i1 %[[VAL_35]], label %[[VAL_36:.*]], label %[[VAL_37:.*]]
// CHECK:       omp.reduction.neutral7:                           ; preds = %[[VAL_33]]
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp.reduction.neutral8:                           ; preds = %[[VAL_36]], %[[VAL_37]]
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp.region.cont4:                                 ; preds = %[[VAL_38]]
// CHECK:         %[[VAL_40:.*]] = phi ptr [ %[[VAL_15]], %[[VAL_38]] ]
// CHECK:         store ptr %[[VAL_40]], ptr %[[VAL_21]], align 8
// CHECK:         br label %[[VAL_41:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_39]]
// CHECK:         br label %[[VAL_42:.*]]
// CHECK:       omp.par.region10:                                 ; preds = %[[VAL_41]]
// CHECK:         br label %[[VAL_43:.*]]
// CHECK:       omp.region.cont9:                                 ; preds = %[[VAL_42]]
// CHECK:         %[[VAL_44:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_22]], i64 0, i64 0
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_44]], align 8
// CHECK:         %[[VAL_45:.*]] = getelementptr inbounds [2 x ptr], ptr %[[VAL_22]], i64 0, i64 1
// CHECK:         store ptr %[[VAL_21]], ptr %[[VAL_45]], align 8
// CHECK:         %[[VAL_46:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_47:.*]] = call i32 @__kmpc_reduce(ptr @1, i32 %[[VAL_46]], i32 2, i64 16, ptr %[[VAL_22]], ptr @.omp.reduction.func, ptr @.gomp_critical_user_.reduction.var)
// CHECK:         switch i32 %[[VAL_47]], label %[[VAL_48:.*]] [
// CHECK:           i32 1, label %[[VAL_49:.*]]
// CHECK:           i32 2, label %[[VAL_50:.*]]
// CHECK:         ]
// CHECK:       reduce.switch.atomic:                             ; preds = %[[VAL_43]]
// CHECK:         unreachable
// CHECK:       reduce.switch.nonatomic:                          ; preds = %[[VAL_43]]
// CHECK:         %[[VAL_51:.*]] = load ptr, ptr %[[VAL_20]], align 8
// CHECK:         %[[VAL_52:.*]] = load ptr, ptr %[[VAL_21]], align 8
// CHECK:         call void @__kmpc_end_reduce(ptr @1, i32 %[[VAL_46]], ptr @.gomp_critical_user_.reduction.var)
// CHECK:         br label %[[VAL_48]]
// CHECK:       reduce.finalize:                                  ; preds = %[[VAL_49]], %[[VAL_43]]
// CHECK:         br label %[[VAL_53:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_48]]
// CHECK:         %[[VAL_54:.*]] = load ptr, ptr %[[VAL_20]], align 8
// CHECK:         %[[VAL_55:.*]] = load ptr, ptr %[[VAL_21]], align 8
// CHECK:         br label %[[VAL_56:.*]]
// CHECK:       omp.reduction.neutral6:                           ; preds = %[[VAL_33]]
// CHECK:         br label %[[VAL_38]]
// CHECK:       omp.reduction.neutral1:                           ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_30]]
// CHECK:       omp.par.outlined.exit.exitStub:                   ; preds = %[[VAL_53]]
// CHECK:         ret void
