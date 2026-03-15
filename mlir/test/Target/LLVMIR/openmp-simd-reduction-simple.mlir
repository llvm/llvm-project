// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFsimd_reductionEi_private_i32 : i32
omp.declare_reduction @add_reduction_f32 : f32 init {
^bb0(%arg0: f32):
  %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  omp.yield(%0 : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %0 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<contract>} : f32
  omp.yield(%0 : f32)
}
llvm.func @_QPsimd_reduction(%arg0: !llvm.ptr {fir.bindc_name = "a", llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "sum", llvm.nocapture}) {
  %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(1024 : i32) : i32
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  llvm.store %0, %arg1 : f32, !llvm.ptr
  omp.simd private(@_QFsimd_reductionEi_private_i32 %4 -> %arg2 : !llvm.ptr) reduction(@add_reduction_f32 %arg1 -> %arg3 : !llvm.ptr) {
    omp.loop_nest (%arg4) : i32 = (%1) to (%2) inclusive step (%1) {
      llvm.store %arg4, %arg2 : i32, !llvm.ptr
      %5 = llvm.load %arg3 : !llvm.ptr -> f32
      %6 = llvm.load %arg2 : !llvm.ptr -> i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.sub %7, %3 overflow<nsw> : i64
      %9 = llvm.getelementptr %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.fadd %5, %10 {fastmathFlags = #llvm.fastmath<contract>} : f32
      llvm.store %11, %arg3 : f32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QPsimd_reduction(
// CHECK:         %[[ORIG_I:.*]] = alloca i32, i64 1, align 4
// CHECK:         store float 0.000000e+00, ptr %[[ORIG_SUM:.*]], align 4
// CHECK:         %[[PRIV_I:.*]] = alloca i32, align 4
// CHECK:         %[[RED_VAR:.*]] = alloca float, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_5:.*]]
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_4]]
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_6]]
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.reduction.init:                               ; preds = %[[VAL_7]]
// CHECK:         store float 0.000000e+00, ptr %[[RED_VAR]], align 4
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.simd.region:                                  ; preds = %[[VAL_8]]
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_9]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_12:.*]], %[[VAL_10]]
// CHECK:         %[[VAL_13:.*]] = phi i32 [ 0, %[[VAL_10]] ], [ %[[VAL_14:.*]], %[[VAL_12]] ]
// CHECK:         br label %[[VAL_15:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_11]]
// CHECK:         %[[VAL_16:.*]] = icmp ult i32 %[[VAL_13]], 1024
// CHECK:         br i1 %[[VAL_16]], label %[[VAL_17:.*]], label %[[VAL_18:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_15]]
// CHECK:         %[[VAL_19:.*]] = mul i32 %[[VAL_13]], 1
// CHECK:         %[[VAL_20:.*]] = add i32 %[[VAL_19]], 1
// CHECK:         br label %[[VAL_21:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_17]]
// CHECK:         store i32 %[[VAL_20]], ptr %[[PRIV_I]], align 4, !llvm.access.group ![[ACCESS_GROUP:.*]]
// CHECK:         %[[RED_VAL:.*]] = load float, ptr %[[RED_VAR]], align 4, !llvm.access.group ![[ACCESS_GROUP]]
// CHECK:         %[[VAL_23:.*]] = load i32, ptr %[[PRIV_I]], align 4, !llvm.access.group ![[ACCESS_GROUP]]
// CHECK:         %[[VAL_24:.*]] = sext i32 %[[VAL_23]] to i64
// CHECK:         %[[VAL_25:.*]] = sub nsw i64 %[[VAL_24]], 1
// CHECK:         %[[VAL_26:.*]] = getelementptr float, ptr %[[VAL_27:.*]], i64 %[[VAL_25]]
// CHECK:         %[[VAL_28:.*]] = load float, ptr %[[VAL_26]], align 4, !llvm.access.group ![[ACCESS_GROUP]]
// CHECK:         %[[VAL_29:.*]] = fadd contract float %[[RED_VAL]], %[[VAL_28]]
// CHECK:         store float %[[VAL_29]], ptr %[[RED_VAR]], align 4, !llvm.access.group ![[ACCESS_GROUP]]
// CHECK:         br label %[[VAL_30:.*]]
// CHECK:       omp.region.cont1:                                 ; preds = %[[VAL_21]]
// CHECK:         br label %[[VAL_12]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_30]]
// CHECK:         %[[VAL_14]] = add nuw i32 %[[VAL_13]], 1
// CHECK:         br label %[[VAL_11]], !llvm.loop ![[LOOP:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_15]]
// CHECK:         br label %[[VAL_31:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_18]]
// CHECK:         br label %[[VAL_32:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_31]]
// CHECK:         %[[SUM_VAL:.*]] = load float, ptr %[[ORIG_SUM]], align 4
// CHECK:         %[[RED_VAL:.*]] = load float, ptr %[[RED_VAR]], align 4
// CHECK:         %[[COMBINED_VAL:.*]] = fadd contract float %[[SUM_VAL]], %[[RED_VAL]]
// CHECK:         store float %[[COMBINED_VAL]], ptr %[[ORIG_SUM]], align 4
// CHECK:         ret void

// CHECK: ![[ACCESS_GROUP]] = distinct !{}
// CHECK: ![[LOOP]] = distinct !{![[LOOP]], ![[PARALLEL_ACCESS:.*]], ![[VECTORIZE:.*]]}
// CHECK: ![[PARALLEL_ACCESS]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP]]}
// CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
