// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s
// XFAIL: *

llvm.func @cancel_parallel() {
  omp.parallel {
    omp.cancel cancellation_construct_type(parallel)
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_parallel..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_5:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_6:.*]] = load i32, ptr %[[VAL_7:.*]], align 4
// CHECK:         store i32 %[[VAL_6]], ptr %[[VAL_5]], align 4
// CHECK:         %[[VAL_8:.*]] = load i32, ptr %[[VAL_5]], align 4
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_10:.*]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_9]]
// CHECK:         br label %[[VAL_12:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_11]]
// CHECK:         %[[VAL_13:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_14:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_13]], i32 1)
// CHECK:         %[[VAL_15:.*]] = icmp eq i32 %[[VAL_14]], 0
// CHECK:         br i1 %[[VAL_15]], label %[[VAL_16:.*]], label %[[VAL_17:.*]]
// CHECK:       omp.par.region1.cncl:                             ; preds = %[[VAL_12]]
// CHECK:         %[[VAL_18:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_19:.*]] = call i32 @__kmpc_cancel_barrier(ptr @2, i32 %[[VAL_18]])
// CHECK:         br label %[[VAL_20:.*]]
// CHECK:       omp.par.region1.split:                            ; preds = %[[VAL_12]]
// CHECK:         br label %[[VAL_21:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_16]]
// CHECK:         br label %[[VAL_22:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_21]]
// CHECK:         br label %[[VAL_20]]
// CHECK:       omp.par.exit.exitStub:                            ; preds = %[[VAL_22]], %[[VAL_17]]
// CHECK:         ret void

llvm.func @cancel_parallel_if(%arg0 : i1) {
  omp.parallel {
    omp.cancel cancellation_construct_type(parallel) if(%arg0)
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_parallel_if..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_9:.*]] = getelementptr { ptr }, ptr %[[VAL_10:.*]], i32 0, i32 0
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_12:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_13:.*]] = load i32, ptr %[[VAL_14:.*]], align 4
// CHECK:         store i32 %[[VAL_13]], ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_15:.*]] = load i32, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_16:.*]] = load i1, ptr %[[VAL_11]], align 1
// CHECK:         br label %[[VAL_17:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_18:.*]]
// CHECK:         br label %[[VAL_19:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_17]]
// CHECK:         br label %[[VAL_20:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_19]]
// CHECK:         br i1 %[[VAL_16]], label %[[VAL_21:.*]], label %[[VAL_22:.*]]
// CHECK:       3:                                                ; preds = %[[VAL_20]]
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       4:                                                ; preds = %[[VAL_22]], %[[VAL_24:.*]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_23]]
// CHECK:         br label %[[VAL_26:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_27:.*]]
// CHECK:       5:                                                ; preds = %[[VAL_20]]
// CHECK:         %[[VAL_28:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_29:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_28]], i32 1)
// CHECK:         %[[VAL_30:.*]] = icmp eq i32 %[[VAL_29]], 0
// CHECK:         br i1 %[[VAL_30]], label %[[VAL_24]], label %[[VAL_31:.*]]
// CHECK:       .cncl:                                            ; preds = %[[VAL_21]]
// CHECK:         %[[VAL_32:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_33:.*]] = call i32 @__kmpc_cancel_barrier(ptr @2, i32 %[[VAL_32]])
// CHECK:         br label %[[VAL_27]]
// CHECK:       .split:                                           ; preds = %[[VAL_21]]
// CHECK:         br label %[[VAL_23]]
// CHECK:       omp.par.exit.exitStub:                            ; preds = %[[VAL_31]], %[[VAL_26]]
// CHECK:         ret void

llvm.func @cancel_sections_if(%cond : i1) {
  omp.sections {
    omp.section {
      omp.cancel cancellation_construct_type(sections) if(%cond)
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define void @cancel_sections_if
// CHECK:         %[[VAL_0:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_1:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_2:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_3:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_5:.*]]
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       omp_section_loop.preheader:                       ; preds = %[[VAL_4]]
// CHECK:         store i32 0, ptr %[[VAL_1]], align 4
// CHECK:         store i32 0, ptr %[[VAL_2]], align 4
// CHECK:         store i32 1, ptr %[[VAL_3]], align 4
// CHECK:         %[[VAL_7:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_7]], i32 34, ptr %[[VAL_0]], ptr %[[VAL_1]], ptr %[[VAL_2]], ptr %[[VAL_3]], i32 1, i32 0)
// CHECK:         %[[VAL_8:.*]] = load i32, ptr %[[VAL_1]], align 4
// CHECK:         %[[VAL_9:.*]] = load i32, ptr %[[VAL_2]], align 4
// CHECK:         %[[VAL_10:.*]] = sub i32 %[[VAL_9]], %[[VAL_8]]
// CHECK:         %[[VAL_11:.*]] = add i32 %[[VAL_10]], 1
// CHECK:         br label %[[VAL_12:.*]]
// CHECK:       omp_section_loop.header:                          ; preds = %[[VAL_13:.*]], %[[VAL_6]]
// CHECK:         %[[VAL_14:.*]] = phi i32 [ 0, %[[VAL_6]] ], [ %[[VAL_15:.*]], %[[VAL_13]] ]
// CHECK:         br label %[[VAL_16:.*]]
// CHECK:       omp_section_loop.cond:                            ; preds = %[[VAL_12]]
// CHECK:         %[[VAL_17:.*]] = icmp ult i32 %[[VAL_14]], %[[VAL_11]]
// CHECK:         br i1 %[[VAL_17]], label %[[VAL_18:.*]], label %[[VAL_19:.*]]
// CHECK:       omp_section_loop.body:                            ; preds = %[[VAL_16]]
// CHECK:         %[[VAL_20:.*]] = add i32 %[[VAL_14]], %[[VAL_8]]
// CHECK:         %[[VAL_21:.*]] = mul i32 %[[VAL_20]], 1
// CHECK:         %[[VAL_22:.*]] = add i32 %[[VAL_21]], 0
// CHECK:         switch i32 %[[VAL_22]], label %[[VAL_23:.*]] [
// CHECK:           i32 0, label %[[VAL_24:.*]]
// CHECK:         ]
// CHECK:       omp_section_loop.body.case:                       ; preds = %[[VAL_18]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.section.region:                               ; preds = %[[VAL_24]]
// CHECK:         br i1 %[[VAL_26:.*]], label %[[VAL_27:.*]], label %[[VAL_28:.*]]
// CHECK:       8:                                                ; preds = %[[VAL_25]]
// CHECK:         %[[VAL_29:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_30:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_29]], i32 3)
// CHECK:         %[[VAL_31:.*]] = icmp eq i32 %[[VAL_30]], 0
// CHECK:         br i1 %[[VAL_31]], label %[[VAL_32:.*]], label %[[VAL_33:.*]]
// CHECK:       .split:                                           ; preds = %[[VAL_27]]
// CHECK:         br label %[[VAL_34:.*]]
// CHECK:       11:                                               ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_34]]
// CHECK:       12:                                               ; preds = %[[VAL_28]], %[[VAL_32]]
// CHECK:         br label %[[VAL_35:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_34]]
// CHECK:         br label %[[VAL_23]]
// CHECK:       omp_section_loop.body.sections.after:             ; preds = %[[VAL_35]], %[[VAL_18]]
// CHECK:         br label %[[VAL_13]]
// CHECK:       omp_section_loop.inc:                             ; preds = %[[VAL_23]]
// CHECK:         %[[VAL_15]] = add nuw i32 %[[VAL_14]], 1
// CHECK:         br label %[[VAL_12]]
// CHECK:       omp_section_loop.exit:                            ; preds = %[[VAL_33]], %[[VAL_16]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_7]])
// CHECK:         %[[VAL_36:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_36]])
// CHECK:         br label %[[VAL_37:.*]]
// CHECK:       omp_section_loop.after:                           ; preds = %[[VAL_19]]
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp_section_loop.aftersections.fini:              ; preds = %[[VAL_37]]
// CHECK:         ret void
// CHECK:       .cncl:                                            ; preds = %[[VAL_27]]
// CHECK:         br label %[[VAL_19]]

llvm.func @cancel_wsloop_if(%lb : i32, %ub : i32, %step : i32, %cond : i1) {
  omp.wsloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.cancel cancellation_construct_type(loop) if(%cond)
      omp.yield
    }
  }
  llvm.return
}
// CHECK-LABEL: define void @cancel_wsloop_if
// CHECK:         %[[VAL_0:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_1:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_2:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_3:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_5:.*]]
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_4]]
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       omp.wsloop.region:                                ; preds = %[[VAL_6]]
// CHECK:         %[[VAL_8:.*]] = icmp slt i32 %[[VAL_9:.*]], 0
// CHECK:         %[[VAL_10:.*]] = sub i32 0, %[[VAL_9]]
// CHECK:         %[[VAL_11:.*]] = select i1 %[[VAL_8]], i32 %[[VAL_10]], i32 %[[VAL_9]]
// CHECK:         %[[VAL_12:.*]] = select i1 %[[VAL_8]], i32 %[[VAL_13:.*]], i32 %[[VAL_14:.*]]
// CHECK:         %[[VAL_15:.*]] = select i1 %[[VAL_8]], i32 %[[VAL_14]], i32 %[[VAL_13]]
// CHECK:         %[[VAL_16:.*]] = sub nsw i32 %[[VAL_15]], %[[VAL_12]]
// CHECK:         %[[VAL_17:.*]] = icmp sle i32 %[[VAL_15]], %[[VAL_12]]
// CHECK:         %[[VAL_18:.*]] = sub i32 %[[VAL_16]], 1
// CHECK:         %[[VAL_19:.*]] = udiv i32 %[[VAL_18]], %[[VAL_11]]
// CHECK:         %[[VAL_20:.*]] = add i32 %[[VAL_19]], 1
// CHECK:         %[[VAL_21:.*]] = icmp ule i32 %[[VAL_16]], %[[VAL_11]]
// CHECK:         %[[VAL_22:.*]] = select i1 %[[VAL_21]], i32 1, i32 %[[VAL_20]]
// CHECK:         %[[VAL_23:.*]] = select i1 %[[VAL_17]], i32 0, i32 %[[VAL_22]]
// CHECK:         br label %[[VAL_24:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_7]]
// CHECK:         store i32 0, ptr %[[VAL_1]], align 4
// CHECK:         %[[VAL_25:.*]] = sub i32 %[[VAL_23]], 1
// CHECK:         store i32 %[[VAL_25]], ptr %[[VAL_2]], align 4
// CHECK:         store i32 1, ptr %[[VAL_3]], align 4
// CHECK:         %[[VAL_26:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_26]], i32 34, ptr %[[VAL_0]], ptr %[[VAL_1]], ptr %[[VAL_2]], ptr %[[VAL_3]], i32 1, i32 0)
// CHECK:         %[[VAL_27:.*]] = load i32, ptr %[[VAL_1]], align 4
// CHECK:         %[[VAL_28:.*]] = load i32, ptr %[[VAL_2]], align 4
// CHECK:         %[[VAL_29:.*]] = sub i32 %[[VAL_28]], %[[VAL_27]]
// CHECK:         %[[VAL_30:.*]] = add i32 %[[VAL_29]], 1
// CHECK:         br label %[[VAL_31:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_32:.*]], %[[VAL_24]]
// CHECK:         %[[VAL_33:.*]] = phi i32 [ 0, %[[VAL_24]] ], [ %[[VAL_34:.*]], %[[VAL_32]] ]
// CHECK:         br label %[[VAL_35:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_31]]
// CHECK:         %[[VAL_36:.*]] = icmp ult i32 %[[VAL_33]], %[[VAL_30]]
// CHECK:         br i1 %[[VAL_36]], label %[[VAL_37:.*]], label %[[VAL_38:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_35]]
// CHECK:         %[[VAL_39:.*]] = add i32 %[[VAL_33]], %[[VAL_27]]
// CHECK:         %[[VAL_40:.*]] = mul i32 %[[VAL_39]], %[[VAL_9]]
// CHECK:         %[[VAL_41:.*]] = add i32 %[[VAL_40]], %[[VAL_14]]
// CHECK:         br label %[[VAL_42:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_37]]
// CHECK:         br i1 %[[VAL_43:.*]], label %[[VAL_44:.*]], label %[[VAL_45:.*]]
// CHECK:       25:                                               ; preds = %[[VAL_42]]
// CHECK:         %[[VAL_46:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_47:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_46]], i32 2)
// CHECK:         %[[VAL_48:.*]] = icmp eq i32 %[[VAL_47]], 0
// CHECK:         br i1 %[[VAL_48]], label %[[VAL_49:.*]], label %[[VAL_50:.*]]
// CHECK:       .split:                                           ; preds = %[[VAL_44]]
// CHECK:         br label %[[VAL_51:.*]]
// CHECK:       28:                                               ; preds = %[[VAL_42]]
// CHECK:         br label %[[VAL_51]]
// CHECK:       29:                                               ; preds = %[[VAL_45]], %[[VAL_49]]
// CHECK:         br label %[[VAL_52:.*]]
// CHECK:       omp.region.cont1:                                 ; preds = %[[VAL_51]]
// CHECK:         br label %[[VAL_32]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_52]]
// CHECK:         %[[VAL_34]] = add nuw i32 %[[VAL_33]], 1
// CHECK:         br label %[[VAL_31]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_50]], %[[VAL_35]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_26]])
// CHECK:         %[[VAL_53:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_53]])
// CHECK:         br label %[[VAL_54:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_38]]
// CHECK:         br label %[[VAL_55:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_54]]
// CHECK:         ret void
// CHECK:       .cncl:                                            ; preds = %[[VAL_44]]
// CHECK:         br label %[[VAL_38]]

omp.private {type = firstprivate} @i32_priv : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @do_something(!llvm.ptr)

llvm.func @cancel_taskgroup(%arg0: !llvm.ptr) {
  omp.taskgroup {
// Using firstprivate clause so we have some end of task cleanup to branch to
// after the cancellation.
    omp.task private(@i32_priv %arg0 -> %arg1 : !llvm.ptr) {
      omp.cancel cancellation_construct_type(taskgroup)
      llvm.call @do_something(%arg1) : (!llvm.ptr) -> ()
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_taskgroup..omp_par(
// CHECK:       task.alloca:
// CHECK:         %[[VAL_21:.*]] = load ptr, ptr %[[VAL_22:.*]], align 8
// CHECK:         %[[VAL_23:.*]] = getelementptr { ptr }, ptr %[[VAL_21]], i32 0, i32 0
// CHECK:         %[[VAL_24:.*]] = load ptr, ptr %[[VAL_23]], align 8, !align !1
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       task.body:                                        ; preds = %[[VAL_26:.*]]
// CHECK:         %[[VAL_27:.*]] = getelementptr { i32 }, ptr %[[VAL_24]], i32 0, i32 0
// CHECK:         br label %[[VAL_28:.*]]
// CHECK:       omp.task.region:                                  ; preds = %[[VAL_25]]
// CHECK:         %[[VAL_29:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_30:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_29]], i32 4)
// CHECK:         %[[VAL_31:.*]] = icmp eq i32 %[[VAL_30]], 0
// CHECK:         br i1 %[[VAL_31]], label %omp.task.region.split, label %omp.task.region.cncl
// CHECK:       omp.task.region.cncl:
// CHECK:         br label %omp.region.cont2
// CHECK:       omp.region.cont2:
// Both cancellation and normal paths reach the end-of-task cleanup:
// CHECK:         tail call void @free(ptr %[[VAL_24]])
// CHECK:         br label %task.exit.exitStub
// CHECK:       omp.task.region.split:
// CHECK:         call void @do_something(ptr %[[VAL_27]])
// CHECK:         br label %omp.region.cont2
// CHECK:       task.exit.exitStub:
// CHECK:         ret void
