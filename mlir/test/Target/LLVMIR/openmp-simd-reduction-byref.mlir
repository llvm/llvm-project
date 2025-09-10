// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

llvm.func @init(%arg0: !llvm.ptr {llvm.nocapture}, %arg1: !llvm.ptr {llvm.nocapture}) {
  llvm.return
}
llvm.func @combine(%arg0: !llvm.ptr {llvm.nocapture}, %arg1: !llvm.ptr {llvm.nocapture}) {
  llvm.return
}
llvm.func @cleanup(%arg0: !llvm.ptr {llvm.nocapture}) {
  llvm.return
}
omp.private {type = private} @_QFsimd_reductionEi_private_i32 : i32
omp.declare_reduction @add_reduction_byref_box_2xf32 : !llvm.ptr alloc {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
} init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @init(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  omp.yield(%arg1 : !llvm.ptr)
} combiner {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @combine(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  omp.yield(%arg0 : !llvm.ptr)
} cleanup {
^bb0(%arg0: !llvm.ptr):
  llvm.call @cleanup(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}
llvm.func @_QPsimd_reduction(%arg0: !llvm.ptr {fir.bindc_name = "a", llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "sum", llvm.nocapture}) {
  %0 = llvm.mlir.constant(1024 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
  %4 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.simd private(@_QFsimd_reductionEi_private_i32 %4 -> %arg2 : !llvm.ptr) reduction(byref @add_reduction_byref_box_2xf32 %3 -> %arg3 : !llvm.ptr) {
    omp.loop_nest (%arg4) : i32 = (%1) to (%0) inclusive step (%1) {
      llvm.store %arg4, %arg2 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QPsimd_reduction
// CHECK:         %[[MOLD:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         %[[ORIG_I:.*]] = alloca i32, i64 1, align 4
// CHECK:         %[[PRIV_I:.*]] = alloca i32, align 4
// CHECK:         %[[RED_VAR:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
// CHECK:         %[[PTR_RED_VAR:.*]] = alloca ptr, align 8
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_6:.*]]
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_5]]
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_7]]
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.reduction.init:                               ; preds = %[[VAL_8]]
// CHECK:         store ptr %[[RED_VAR]], ptr %[[PTR_RED_VAR]], align 8
// CHECK:         call void @init(ptr %[[MOLD]], ptr %[[RED_VAR]])
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp.simd.region:                                  ; preds = %[[VAL_9]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_10]]
// CHECK:         br label %[[VAL_12:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_13:.*]], %[[VAL_11]]
// CHECK:         %[[VAL_14:.*]] = phi i32 [ 0, %[[VAL_11]] ], [ %[[VAL_15:.*]], %[[VAL_13]] ]
// CHECK:         br label %[[VAL_16:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_12]]
// CHECK:         %[[VAL_17:.*]] = icmp ult i32 %[[VAL_14]], 1024
// CHECK:         br i1 %[[VAL_17]], label %[[VAL_18:.*]], label %[[VAL_19:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_16]]
// CHECK:         %[[VAL_20:.*]] = mul i32 %[[VAL_14]], 1
// CHECK:         %[[VAL_21:.*]] = add i32 %[[VAL_20]], 1
// CHECK:         br label %[[VAL_22:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_18]]
// CHECK:         store i32 %[[VAL_21]], ptr %[[PRIV_I]], align 4, !llvm.access.group ![[ACCESS_GROUP:.*]]
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       omp.region.cont1:                                 ; preds = %[[VAL_22]]
// CHECK:         br label %[[VAL_13]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_23]]
// CHECK:         %[[VAL_15]] = add nuw i32 %[[VAL_14]], 1
// CHECK:         br label %[[VAL_12]], !llvm.loop ![[LOOP:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_16]]
// CHECK:         br label %[[VAL_24:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_19]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_24]]
// CHECK:         %[[RED_VAR2:.*]] = load ptr, ptr %[[PTR_RED_VAR]], align 8
// CHECK:         call void @combine(ptr %[[MOLD]], ptr %[[RED_VAR2]])
// CHECK:         %[[RED_VAR3:.*]] = load ptr, ptr %[[PTR_RED_VAR]], align 8
// CHECK:         call void @cleanup(ptr %[[RED_VAR3]])
// CHECK:         ret void

// CHECK: ![[ACCESS_GROUP]] = distinct !{}
// CHECK: ![[LOOP]] = distinct !{![[LOOP]], ![[PARALLEL_ACCESS:.*]], ![[VECTORIZE:.*]]}
// CHECK: ![[PARALLEL_ACCESS]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP]]}
// CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
