// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @_QPfoo(%arg0: !llvm.ptr {fir.bindc_name = "array", llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.nocapture}) {
  %0 = llvm.mlir.constant(0 : i64) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(10 : i64) : i64
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
  %5 = llvm.load %arg1 : !llvm.ptr -> i32
  %6 = llvm.icmp "ne" %5, %0 : i32
  %7 = llvm.trunc %2 : i64 to i32
  omp.wsloop {
    omp.simd if(%6) {
      omp.loop_nest (%arg2) : i32 = (%1) to (%7) inclusive step (%1) {
        llvm.store %arg2, %4 : i32, !llvm.ptr
        %8 = llvm.load %4 : !llvm.ptr -> i32
        %9 = llvm.sext %8 : i32 to i64
        %10 = llvm.getelementptr %arg0[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %8, %10 : i32, !llvm.ptr
        omp.yield
      }
    } {omp.composite}
  } {omp.composite}
  llvm.return
}

// CHECK-LABEL: @_QPfoo
// ...
// CHECK:       omp_loop.preheader:                               ; preds =
// CHECK:         store i32 0, ptr %[[LB_ADDR:.*]], align 4
// CHECK:         store i32 9, ptr %[[UB_ADDR:.*]], align 4
// CHECK:         store i32 1, ptr %[[STEP_ADDR:.*]], align 4
// CHECK:         %[[VAL_15:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_15]], i32 34, ptr %{{.*}}, ptr %[[LB_ADDR]], ptr %[[UB_ADDR]], ptr %[[STEP_ADDR]], i32 1, i32 0)
// CHECK:         %[[LB:.*]] = load i32, ptr %[[LB_ADDR]], align 4
// CHECK:         %[[UB:.*]] = load i32, ptr %[[UB_ADDR]], align 4
// CHECK:         %[[VAL_18:.*]] = sub i32 %[[UB]], %[[LB]]
// CHECK:         %[[COUNT:.*]] = add i32 %[[VAL_18]], 1
// CHECK:         br label %[[OMP_LOOP_HEADER:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[OMP_LOOP_INC:.*]], %[[OMP_LOOP_PREHEADER:.*]]
// CHECK:         %[[IV:.*]] = phi i32 [ 0, %[[OMP_LOOP_PREHEADER]] ], [ %[[NEW_IV:.*]], %[[OMP_LOOP_INC]] ]
// CHECK:         br label %[[OMP_LOOP_COND:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[OMP_LOOP_HEADER]]
// CHECK:         %[[VAL_25:.*]] = icmp ult i32 %[[IV]], %[[COUNT]]
// CHECK:         br i1 %[[VAL_25]], label %[[OMP_LOOP_BODY:.*]], label %[[OMP_LOOP_EXIT:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[OMP_LOOP_COND]]
// CHECK:         %[[VAL_28:.*]] = add i32 %[[IV]], %[[LB]]
// This is the IF clause:
// CHECK:         br i1 %{{.*}}, label %[[SIMD_IF_THEN:.*]], label %[[SIMD_IF_ELSE:.*]]

// CHECK:       simd.if.then:                                     ; preds = %[[OMP_LOOP_BODY]]
// CHECK:         %[[VAL_29:.*]] = mul i32 %[[VAL_28]], 1
// CHECK:         %[[VAL_30:.*]] = add i32 %[[VAL_29]], 1
// CHECK:         br label %[[VAL_33:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[SIMD_IF_THEN]]
// This version contains !llvm.access.group metadata for SIMD
// CHECK:         store i32 %[[VAL_30]], ptr %{{.*}}, align 4, !llvm.access.group !1
// CHECK:         %[[VAL_34:.*]] = load i32, ptr %{{.*}}, align 4, !llvm.access.group !1
// CHECK:         %[[VAL_35:.*]] = sext i32 %[[VAL_34]] to i64
// CHECK:         %[[VAL_36:.*]] = getelementptr i32, ptr %[[VAL_37:.*]], i64 %[[VAL_35]]
// CHECK:         store i32 %[[VAL_34]], ptr %[[VAL_36]], align 4, !llvm.access.group !1
// CHECK:         br label %[[OMP_REGION_CONT3:.*]]
// CHECK:       omp.region.cont3:                                 ; preds = %[[VAL_33]]
// CHECK:         br label %[[SIMD_PRE_LATCH:.*]]

// CHECK:       simd.pre_latch:                                   ; preds = %[[OMP_REGION_CONT3]], %[[OMP_REGION_CONT35:.*]]
// CHECK:         br label %[[OMP_LOOP_INC]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[SIMD_PRE_LATCH]]
// CHECK:         %[[NEW_IV]] = add nuw i32 %[[IV]], 1
// CHECK:         br label %[[OMP_LOOP_HEADER]], !llvm.loop !2

// CHECK:       simd.if.else:                                     ; preds = %[[OMP_LOOP_BODY]]
// CHECK:         br label %[[SIMD_IF_ELSE2:.*]]
// CHECK:       simd.if.else5:
// CHECK:         %[[MUL:.*]] = mul i32 %[[VAL_28]], 1
// CHECK:         %[[ADD:.*]] = add i32 %[[MUL]], 1
// CHECK:         br label %[[LOOP_NEST_REGION:.*]]
// CHECK:       omp.loop_nest.region6:                            ; preds = %[[SIMD_IF_ELSE2]]
// No llvm.access.group metadata for else clause
// CHECK:         store i32 %[[ADD]], ptr %{{.*}}, align 4
// CHECK:         %[[VAL_42:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK:         %[[VAL_43:.*]] = sext i32 %[[VAL_42]] to i64
// CHECK:         %[[VAL_44:.*]] = getelementptr i32, ptr %[[VAL_37]], i64 %[[VAL_43]]
// CHECK:         store i32 %[[VAL_42]], ptr %[[VAL_44]], align 4
// CHECK:         br label %[[OMP_REGION_CONT35]]
// CHECK:       omp.region.cont37:                                ; preds = %[[LOOP_NEST_REGION]]
// CHECK:         br label %[[SIMD_PRE_LATCH]]

// CHECK:       omp_loop.exit:                                    ; preds = %[[OMP_LOOP_COND]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_15]])
// CHECK:         %[[VAL_45:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_45]])

// CHECK: !1 = distinct !{}
// CHECK: !2 = distinct !{!2, !3}
// CHECK: !3 = !{!"llvm.loop.parallel_accesses", !1}
// CHECK-NOT: llvm.loop.vectorize
