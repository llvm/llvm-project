// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

llvm.func @cancellation_point_parallel() {
  omp.parallel {
    omp.cancellation_point cancellation_construct_type(parallel)
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancellation_point_parallel..omp_par
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
// CHECK:         %[[VAL_14:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_13]], i32 1)
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

llvm.func @cancellation_point_sections() {
  omp.sections {
    omp.section {
      omp.cancellation_point cancellation_construct_type(sections)
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define void @cancellation_point_sections
// CHECK:         %[[VAL_23:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_24:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_25:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_26:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_27:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_28:.*]]
// CHECK:         br label %[[VAL_29:.*]]
// CHECK:       omp_section_loop.preheader:                       ; preds = %[[VAL_27]]
// CHECK:         store i32 0, ptr %[[VAL_24]], align 4
// CHECK:         store i32 0, ptr %[[VAL_25]], align 4
// CHECK:         store i32 1, ptr %[[VAL_26]], align 4
// CHECK:         %[[VAL_30:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_30]], i32 34, ptr %[[VAL_23]], ptr %[[VAL_24]], ptr %[[VAL_25]], ptr %[[VAL_26]], i32 1, i32 0)
// CHECK:         %[[VAL_31:.*]] = load i32, ptr %[[VAL_24]], align 4
// CHECK:         %[[VAL_32:.*]] = load i32, ptr %[[VAL_25]], align 4
// CHECK:         %[[VAL_33:.*]] = sub i32 %[[VAL_32]], %[[VAL_31]]
// CHECK:         %[[VAL_34:.*]] = add i32 %[[VAL_33]], 1
// CHECK:         br label %[[VAL_35:.*]]
// CHECK:       omp_section_loop.header:                          ; preds = %[[VAL_36:.*]], %[[VAL_29]]
// CHECK:         %[[VAL_37:.*]] = phi i32 [ 0, %[[VAL_29]] ], [ %[[VAL_38:.*]], %[[VAL_36]] ]
// CHECK:         br label %[[VAL_39:.*]]
// CHECK:       omp_section_loop.cond:                            ; preds = %[[VAL_35]]
// CHECK:         %[[VAL_40:.*]] = icmp ult i32 %[[VAL_37]], %[[VAL_34]]
// CHECK:         br i1 %[[VAL_40]], label %[[VAL_41:.*]], label %[[VAL_42:.*]]
// CHECK:       omp_section_loop.body:                            ; preds = %[[VAL_39]]
// CHECK:         %[[VAL_43:.*]] = add i32 %[[VAL_37]], %[[VAL_31]]
// CHECK:         %[[VAL_44:.*]] = mul i32 %[[VAL_43]], 1
// CHECK:         %[[VAL_45:.*]] = add i32 %[[VAL_44]], 0
// CHECK:         switch i32 %[[VAL_45]], label %[[VAL_46:.*]] [
// CHECK:           i32 0, label %[[VAL_47:.*]]
// CHECK:         ]
// CHECK:       omp_section_loop.body.case:                       ; preds = %[[VAL_41]]
// CHECK:         br label %[[VAL_48:.*]]
// CHECK:       omp.section.region:                               ; preds = %[[VAL_47]]
// CHECK:         %[[VAL_49:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_50:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_49]], i32 3)
// CHECK:         %[[VAL_51:.*]] = icmp eq i32 %[[VAL_50]], 0
// CHECK:         br i1 %[[VAL_51]], label %[[VAL_52:.*]], label %[[VAL_53:.*]]
// CHECK:       omp.section.region.split:                         ; preds = %[[VAL_48]]
// CHECK:         br label %[[VAL_54:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_52]]
// CHECK:         br label %[[VAL_46]]
// CHECK:       omp_section_loop.body.sections.after:             ; preds = %[[VAL_54]], %[[VAL_41]]
// CHECK:         br label %[[VAL_36]]
// CHECK:       omp_section_loop.inc:                             ; preds = %[[VAL_46]]
// CHECK:         %[[VAL_38]] = add nuw i32 %[[VAL_37]], 1
// CHECK:         br label %[[VAL_35]]
// CHECK:       omp_section_loop.exit:                            ; preds = %[[VAL_53]], %[[VAL_39]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_30]])
// CHECK:         %[[VAL_55:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_55]])
// CHECK:         br label %[[VAL_56:.*]]
// CHECK:       omp_section_loop.after:                           ; preds = %[[VAL_42]]
// CHECK:         br label %[[VAL_57:.*]]
// CHECK:       omp_section_loop.aftersections.fini:              ; preds = %[[VAL_56]]
// CHECK:         ret void
// CHECK:       omp.section.region.cncl:                          ; preds = %[[VAL_48]]
// CHECK:         br label %[[VAL_42]]

llvm.func @cancellation_point_wsloop(%lb : i32, %ub : i32, %step : i32) {
  omp.wsloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.cancellation_point cancellation_construct_type(loop)
      omp.yield
    }
  }
  llvm.return
}
// CHECK-LABEL: define void @cancellation_point_wsloop
// CHECK:         %[[VAL_58:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_59:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_60:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_61:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_62:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_63:.*]]
// CHECK:         br label %[[VAL_64:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_62]]
// CHECK:         br label %[[VAL_65:.*]]
// CHECK:       omp.wsloop.region:                                ; preds = %[[VAL_64]]
// CHECK:         %[[VAL_66:.*]] = icmp slt i32 %[[VAL_67:.*]], 0
// CHECK:         %[[VAL_68:.*]] = sub i32 0, %[[VAL_67]]
// CHECK:         %[[VAL_69:.*]] = select i1 %[[VAL_66]], i32 %[[VAL_68]], i32 %[[VAL_67]]
// CHECK:         %[[VAL_70:.*]] = select i1 %[[VAL_66]], i32 %[[VAL_71:.*]], i32 %[[VAL_72:.*]]
// CHECK:         %[[VAL_73:.*]] = select i1 %[[VAL_66]], i32 %[[VAL_72]], i32 %[[VAL_71]]
// CHECK:         %[[VAL_74:.*]] = sub nsw i32 %[[VAL_73]], %[[VAL_70]]
// CHECK:         %[[VAL_75:.*]] = icmp sle i32 %[[VAL_73]], %[[VAL_70]]
// CHECK:         %[[VAL_76:.*]] = sub i32 %[[VAL_74]], 1
// CHECK:         %[[VAL_77:.*]] = udiv i32 %[[VAL_76]], %[[VAL_69]]
// CHECK:         %[[VAL_78:.*]] = add i32 %[[VAL_77]], 1
// CHECK:         %[[VAL_79:.*]] = icmp ule i32 %[[VAL_74]], %[[VAL_69]]
// CHECK:         %[[VAL_80:.*]] = select i1 %[[VAL_79]], i32 1, i32 %[[VAL_78]]
// CHECK:         %[[VAL_81:.*]] = select i1 %[[VAL_75]], i32 0, i32 %[[VAL_80]]
// CHECK:         br label %[[VAL_82:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_65]]
// CHECK:         store i32 0, ptr %[[VAL_59]], align 4
// CHECK:         %[[VAL_83:.*]] = sub i32 %[[VAL_81]], 1
// CHECK:         store i32 %[[VAL_83]], ptr %[[VAL_60]], align 4
// CHECK:         store i32 1, ptr %[[VAL_61]], align 4
// CHECK:         %[[VAL_84:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_84]], i32 34, ptr %[[VAL_58]], ptr %[[VAL_59]], ptr %[[VAL_60]], ptr %[[VAL_61]], i32 1, i32 0)
// CHECK:         %[[VAL_85:.*]] = load i32, ptr %[[VAL_59]], align 4
// CHECK:         %[[VAL_86:.*]] = load i32, ptr %[[VAL_60]], align 4
// CHECK:         %[[VAL_87:.*]] = sub i32 %[[VAL_86]], %[[VAL_85]]
// CHECK:         %[[VAL_88:.*]] = add i32 %[[VAL_87]], 1
// CHECK:         br label %[[VAL_89:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_90:.*]], %[[VAL_82]]
// CHECK:         %[[VAL_91:.*]] = phi i32 [ 0, %[[VAL_82]] ], [ %[[VAL_92:.*]], %[[VAL_90]] ]
// CHECK:         br label %[[VAL_93:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_89]]
// CHECK:         %[[VAL_94:.*]] = icmp ult i32 %[[VAL_91]], %[[VAL_88]]
// CHECK:         br i1 %[[VAL_94]], label %[[VAL_95:.*]], label %[[VAL_96:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_93]]
// CHECK:         %[[VAL_97:.*]] = add i32 %[[VAL_91]], %[[VAL_85]]
// CHECK:         %[[VAL_98:.*]] = mul i32 %[[VAL_97]], %[[VAL_67]]
// CHECK:         %[[VAL_99:.*]] = add i32 %[[VAL_98]], %[[VAL_72]]
// CHECK:         br label %[[VAL_100:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_95]]
// CHECK:         %[[VAL_101:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_102:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_101]], i32 2)
// CHECK:         %[[VAL_103:.*]] = icmp eq i32 %[[VAL_102]], 0
// CHECK:         br i1 %[[VAL_103]], label %[[VAL_104:.*]], label %[[VAL_105:.*]]
// CHECK:       omp.loop_nest.region.split:                       ; preds = %[[VAL_100]]
// CHECK:         br label %[[VAL_106:.*]]
// CHECK:       omp.region.cont1:                                 ; preds = %[[VAL_104]]
// CHECK:         br label %[[VAL_90]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_106]]
// CHECK:         %[[VAL_92]] = add nuw i32 %[[VAL_91]], 1
// CHECK:         br label %[[VAL_89]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_105]], %[[VAL_93]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_84]])
// CHECK:         %[[VAL_107:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_107]])
// CHECK:         br label %[[VAL_108:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_96]]
// CHECK:         br label %[[VAL_109:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_108]]
// CHECK:         ret void
// CHECK:       omp.loop_nest.region.cncl:                        ; preds = %[[VAL_100]]
// CHECK:         br label %[[VAL_96]]


llvm.func @cancellation_point_taskgroup() {
  omp.taskgroup {
    omp.task {
      omp.cancellation_point cancellation_construct_type(taskgroup)
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancellation_point_taskgroup..omp_par(
// CHECK:       task.alloca:
// CHECK:         br label %[[VAL_50:.*]]
// CHECK:       task.body:                                        ; preds = %[[VAL_51:.*]]
// CHECK:         br label %[[VAL_52:.*]]
// CHECK:       omp.task.region:                                  ; preds = %[[VAL_50]]
// CHECK:         %[[VAL_53:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_54:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_53]], i32 4)
// CHECK:         %[[VAL_55:.*]] = icmp eq i32 %[[VAL_54]], 0
// CHECK:         br i1 %[[VAL_55]], label %omp.task.region.split, label %omp.task.region.cncl
// CHECK:       omp.task.region.cncl:
// CHECK:         br label %omp.region.cont1
// CHECK:       omp.region.cont1:
// CHECK:         br label %task.exit.exitStub
// CHECK:       omp.task.region.split:
// CHECK:         br label %omp.region.cont1
// CHECK:       task.exit.exitStub:
// CHECK:         ret void
