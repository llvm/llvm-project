// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFtestEi_private_i32 : i32
omp.private {type = firstprivate} @_QFtestEarg_firstprivate_i32 : i32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @_init(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  omp.yield(%arg1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @_copy(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @_dealloc(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

// Test cancelling implicit taskgroup created by taskloop
llvm.func @_QPtest(%arg0: !llvm.ptr {fir.bindc_name = "arg", llvm.noalias, llvm.nocapture}) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(100 : i32) : i32
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.taskloop private(@_QFtestEarg_firstprivate_i32 %arg0 -> %arg1, @_QFtestEi_private_i32 %3 -> %arg2 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg3) : i32 = (%0) to (%1) inclusive step (%0) {
      llvm.store %arg3, %arg2 : i32, !llvm.ptr
      llvm.call @_QPbefore(%arg1) : (!llvm.ptr) -> ()
      omp.cancellation_point cancellation_construct_type(taskgroup)
      llvm.call @_QPafter(%arg1) : (!llvm.ptr) -> ()
      omp.yield
    }
  }
  llvm.return
}
// CHECK-LABEL: define void @_QPtest(
// CHECK:         %[[STRUCTARG:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK:         %[[VAL_0:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_1:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_2:.*]]
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_1]]
// CHECK:         %[[VAL_4:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:         %[[VAL_5:.*]] = getelementptr { i32 }, ptr %[[VAL_4]], i32 0, i32 0
// CHECK:         call void @_init(ptr %[[VAL_6:.*]], ptr %[[VAL_5]])
// CHECK:         br label %[[VAL_7:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_3]]
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.private.copy1:                                ; preds = %[[VAL_7]]
// CHECK:         call void @_copy(ptr %[[VAL_6]], ptr %[[VAL_5]])
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.taskloop.start:                               ; preds = %[[VAL_8]]
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       codeRepl:                                         ; preds = %[[VAL_9]]
// CHECK:         %[[VAL_11:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CHECK:         store i64 1, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_12:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// CHECK:         store i64 100, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_13:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 2
// CHECK:         store i64 1, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_14:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCTARG]], i32 0, i32 3
// CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_14]], align 8
// CHECK:         %[[VAL_15:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_taskgroup(ptr @1, i32 %[[VAL_15]])
// CHECK:         %[[VAL_16:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_15]], i32 1, i64 40, i64 32, ptr @_QPtest..omp_par)
// CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_16]], align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_17]], ptr align 1 %[[STRUCTARG]], i64 32, i1 false)
// CHECK:         %[[VAL_18:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_17]], i32 0, i32 0
// CHECK:         %[[VAL_19:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_17]], i32 0, i32 1
// CHECK:         %[[VAL_20:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_17]], i32 0, i32 2
// CHECK:         %[[VAL_21:.*]] = load i64, ptr %[[VAL_20]], align 4
// CHECK:         call void @__kmpc_taskloop(ptr @1, i32 %[[VAL_15]], ptr %[[VAL_16]], i32 1, ptr %[[VAL_18]], ptr %[[VAL_19]], i64 %[[VAL_21]], i32 1, i32 0, i64 0, ptr @omp_taskloop_dup)
// CHECK:         call void @__kmpc_end_taskgroup(ptr @1, i32 %[[VAL_15]])
// CHECK:         br label %[[VAL_22:.*]]
// CHECK:       taskloop.exit:                                    ; preds = %[[VAL_10]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @_QPtest..omp_par
// CHECK:       taskloop.alloca:
// CHECK:         %[[VAL_23:.*]] = load ptr, ptr %[[VAL_24:.*]], align 8
// CHECK:         %[[VAL_25:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_23]], i32 0, i32 0
// CHECK:         %[[VAL_26:.*]] = load i64, ptr %[[VAL_25]], align 4
// CHECK:         %[[VAL_27:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_23]], i32 0, i32 1
// CHECK:         %[[VAL_28:.*]] = load i64, ptr %[[VAL_27]], align 4
// CHECK:         %[[VAL_29:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_23]], i32 0, i32 2
// CHECK:         %[[VAL_30:.*]] = load i64, ptr %[[VAL_29]], align 4
// CHECK:         %[[VAL_31:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_23]], i32 0, i32 3
// CHECK:         %[[VAL_32:.*]] = load ptr, ptr %[[VAL_31]], align 8, !align !1
// CHECK:         %[[VAL_33:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_34:.*]]
// CHECK:       taskloop.body:                                    ; preds = %[[VAL_35:.*]]
// CHECK:         %[[VAL_36:.*]] = getelementptr { i32 }, ptr %[[VAL_32]], i32 0, i32 0
// CHECK:         br label %[[VAL_37:.*]]
// CHECK:       omp.taskloop.region:                              ; preds = %[[VAL_34]]
// CHECK:         br label %[[VAL_38:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_37]]
// CHECK:         %[[VAL_39:.*]] = sub i64 %[[VAL_28]], %[[VAL_26]]
// CHECK:         %[[VAL_40:.*]] = sdiv i64 %[[VAL_39]], 1
// CHECK:         %[[VAL_41:.*]] = add i64 %[[VAL_40]], 1
// CHECK:         %[[VAL_42:.*]] = trunc i64 %[[VAL_41]] to i32
// CHECK:         %[[VAL_43:.*]] = trunc i64 %[[VAL_26]] to i32
// CHECK:         br label %[[VAL_44:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_45:.*]], %[[VAL_38]]
// CHECK:         %[[VAL_46:.*]] = phi i32 [ 0, %[[VAL_38]] ], [ %[[VAL_47:.*]], %[[VAL_45]] ]
// CHECK:         br label %[[VAL_48:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_44]]
// CHECK:         %[[VAL_49:.*]] = icmp ult i32 %[[VAL_46]], %[[VAL_42]]
// CHECK:         br i1 %[[VAL_49]], label %[[VAL_50:omp_loop.body]], label %[[VAL_51:omp_loop.exit]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_48]]
// CHECK:         br label %[[OMP_LOOP_AFTER:omp_loop.after]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_51]]
// CHECK:         br label %[[CONT:omp.region.cont]]
// CHECK:       omp.region.cont:                                  ; preds = %[[FINI:.fini]], %[[OMP_LOOP_AFTER]]
// CHECK:         call void @_dealloc(ptr %[[VAL_36]])
// CHECK:         tail call void @free(ptr %[[VAL_32]])
// CHECK:         br label %[[VAL_55:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_48]]
// CHECK:         %[[VAL_56:.*]] = mul i32 %[[VAL_46]], 1
// CHECK:         %[[VAL_57:.*]] = add i32 %[[VAL_56]], %[[VAL_43]]
// CHECK:         br label %[[LOOP_REGION:omp.loop_nest.region]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_50]]
// CHECK:         store i32 %[[VAL_57]], ptr %[[VAL_33]], align 4
// CHECK:         call void @_QPbefore(ptr %[[VAL_36]])
// CHECK:         %[[VAL_59:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_60:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_59]], i32 4)
// CHECK:         %[[VAL_61:.*]] = icmp eq i32 %[[VAL_60]], 0
// CHECK:         br i1 %[[VAL_61]], label %[[VAL_62:omp.loop_nest.region.split]], label %[[CNCL:omp.loop_nest.region.cncl]]
// CHECK:       omp.loop_nest.region.cncl:                        ; preds = %[[LOOP_REGION]]
// CHECK:         br label %[[FINI]]
// CHECK:       .fini:                                            ; preds = %[[CNCL]]
// CHECK:         br label %[[CONT]]
// CHECK:       omp.loop_nest.region.split:                       ; preds = %[[LOOP_REGION]]
// CHECK:         call void @_QPafter(ptr %[[VAL_36]])
// CHECK:         br label %[[VAL_64:.*]]
// CHECK:       omp.region.cont2:                                 ; preds = %[[VAL_62]]
// CHECK:         br label %[[VAL_45]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_64]]
// CHECK:         %[[VAL_47]] = add nuw i32 %[[VAL_46]], 1
// CHECK:         br label %[[VAL_44]]
// CHECK:       taskloop.exit.exitStub:                           ; preds = %[[CONT]]
// CHECK:         ret void

// Tesk cancelling explicit taskgroup enclosing taskloop
llvm.func @_QPtest2(%arg0: !llvm.ptr {fir.bindc_name = "arg", llvm.noalias, llvm.nocapture}) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(100 : i32) : i32
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.taskgroup {
    omp.taskloop nogroup private(@_QFtestEarg_firstprivate_i32 %arg0 -> %arg1, @_QFtestEi_private_i32 %3 -> %arg2 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%arg3) : i32 = (%0) to (%1) inclusive step (%0) {
        llvm.store %arg3, %arg2 : i32, !llvm.ptr
        llvm.call @_QPbefore(%arg1) : (!llvm.ptr) -> ()
        omp.cancellation_point cancellation_construct_type(taskgroup)
        llvm.call @_QPafter(%arg1) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define void @_QPtest2(
// CHECK:         %[[VAL_65:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK:         %[[VAL_66:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_67:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_68:.*]]
// CHECK:         %[[VAL_69:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_taskgroup(ptr @1, i32 %[[VAL_69]])
// CHECK:         br label %[[VAL_70:.*]]
// CHECK:       omp.taskgroup.region:                             ; preds = %[[VAL_67]]
// CHECK:         br label %[[VAL_71:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_70]]
// CHECK:         %[[VAL_72:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:         %[[VAL_73:.*]] = getelementptr { i32 }, ptr %[[VAL_72]], i32 0, i32 0
// CHECK:         call void @_init(ptr %[[VAL_74:.*]], ptr %[[VAL_73]])
// CHECK:         br label %[[VAL_75:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_71]]
// CHECK:         br label %[[VAL_76:.*]]
// CHECK:       omp.private.copy1:                                ; preds = %[[VAL_75]]
// CHECK:         call void @_copy(ptr %[[VAL_74]], ptr %[[VAL_73]])
// CHECK:         br label %[[VAL_77:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_78:.*]]
// CHECK:         br label %[[VAL_79:.*]]
// CHECK:       taskgroup.exit:                                   ; preds = %[[VAL_80:.*]]
// CHECK:         call void @__kmpc_end_taskgroup(ptr @1, i32 %[[VAL_69]])
// CHECK:         ret void
// CHECK:       omp.taskloop.start:                               ; preds = %[[VAL_76]]
// CHECK:         br label %[[VAL_81:.*]]
// CHECK:       codeRepl:                                         ; preds = %[[VAL_77]]
// CHECK:         %[[VAL_82:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_65]], i32 0, i32 0
// CHECK:         store i64 1, ptr %[[VAL_82]], align 4
// CHECK:         %[[VAL_83:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_65]], i32 0, i32 1
// CHECK:         store i64 100, ptr %[[VAL_83]], align 4
// CHECK:         %[[VAL_84:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_65]], i32 0, i32 2
// CHECK:         store i64 1, ptr %[[VAL_84]], align 4
// CHECK:         %[[VAL_85:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_65]], i32 0, i32 3
// CHECK:         store ptr %[[VAL_72]], ptr %[[VAL_85]], align 8
// CHECK:         %[[VAL_86:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_87:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_86]], i32 1, i64 40, i64 32, ptr @_QPtest2..omp_par)
// CHECK:         %[[VAL_88:.*]] = load ptr, ptr %[[VAL_87]], align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_88]], ptr align 1 %[[VAL_65]], i64 32, i1 false)
// CHECK:         %[[VAL_89:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_88]], i32 0, i32 0
// CHECK:         %[[VAL_90:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_88]], i32 0, i32 1
// CHECK:         %[[VAL_91:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_88]], i32 0, i32 2
// CHECK:         %[[VAL_92:.*]] = load i64, ptr %[[VAL_91]], align 4
// CHECK:         call void @__kmpc_taskloop(ptr @1, i32 %[[VAL_86]], ptr %[[VAL_87]], i32 1, ptr %[[VAL_89]], ptr %[[VAL_90]], i64 %[[VAL_92]], i32 1, i32 0, i64 0, ptr @omp_taskloop_dup.1)
// CHECK:         br label %[[VAL_78]]
// CHECK:       taskloop.exit:                                    ; preds = %[[VAL_81]]
// CHECK:         br label %[[VAL_80]]

// CHECK-LABEL: define internal void @_QPtest2..omp_par(
// CHECK:       taskloop.alloca:
// CHECK:         %[[VAL_93:.*]] = load ptr, ptr %[[VAL_94:.*]], align 8
// CHECK:         %[[VAL_95:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_93]], i32 0, i32 0
// CHECK:         %[[VAL_96:.*]] = load i64, ptr %[[VAL_95]], align 4
// CHECK:         %[[VAL_97:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_93]], i32 0, i32 1
// CHECK:         %[[VAL_98:.*]] = load i64, ptr %[[VAL_97]], align 4
// CHECK:         %[[VAL_99:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_93]], i32 0, i32 2
// CHECK:         %[[VAL_100:.*]] = load i64, ptr %[[VAL_99]], align 4
// CHECK:         %[[VAL_101:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_93]], i32 0, i32 3
// CHECK:         %[[VAL_102:.*]] = load ptr, ptr %[[VAL_101]], align 8, !align !1
// CHECK:         %[[VAL_103:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_104:.*]]
// CHECK:       taskloop.body:                                    ; preds = %[[VAL_105:.*]]
// CHECK:         %[[VAL_106:.*]] = getelementptr { i32 }, ptr %[[VAL_102]], i32 0, i32 0
// CHECK:         br label %[[VAL_107:.*]]
// CHECK:       omp.taskloop.region:                              ; preds = %[[VAL_104]]
// CHECK:         br label %[[VAL_108:.*]]
// CHECK:       omp_loop.preheader:                               ; preds = %[[VAL_107]]
// CHECK:         %[[VAL_109:.*]] = sub i64 %[[VAL_98]], %[[VAL_96]]
// CHECK:         %[[VAL_110:.*]] = sdiv i64 %[[VAL_109]], 1
// CHECK:         %[[VAL_111:.*]] = add i64 %[[VAL_110]], 1
// CHECK:         %[[VAL_112:.*]] = trunc i64 %[[VAL_111]] to i32
// CHECK:         %[[VAL_113:.*]] = trunc i64 %[[VAL_96]] to i32
// CHECK:         br label %[[VAL_114:.*]]
// CHECK:       omp_loop.header:                                  ; preds = %[[VAL_115:.*]], %[[VAL_108]]
// CHECK:         %[[VAL_116:.*]] = phi i32 [ 0, %[[VAL_108]] ], [ %[[VAL_117:.*]], %[[VAL_115]] ]
// CHECK:         br label %[[VAL_118:.*]]
// CHECK:       omp_loop.cond:                                    ; preds = %[[VAL_114]]
// CHECK:         %[[VAL_119:.*]] = icmp ult i32 %[[VAL_116]], %[[VAL_112]]
// CHECK:         br i1 %[[VAL_119]], label %[[VAL_120:.*]], label %[[VAL_121:.*]]
// CHECK:       omp_loop.exit:                                    ; preds = %[[VAL_118]]
// CHECK:         br label %[[VAL_122:.*]]
// CHECK:       omp_loop.after:                                   ; preds = %[[VAL_121]]
// CHECK:         br label %[[VAL_123:omp.region.cont2]]
// CHECK:       omp.region.cont2:                                 ; preds = %[[VAL_124:.fini]], %[[VAL_122]]
// CHECK:         call void @_dealloc(ptr %[[VAL_106]])
// CHECK:         tail call void @free(ptr %[[VAL_102]])
// CHECK:         br label %[[VAL_125:.*]]
// CHECK:       omp_loop.body:                                    ; preds = %[[VAL_118]]
// CHECK:         %[[VAL_126:.*]] = mul i32 %[[VAL_116]], 1
// CHECK:         %[[VAL_127:.*]] = add i32 %[[VAL_126]], %[[VAL_113]]
// CHECK:         br label %[[VAL_128:.*]]
// CHECK:       omp.loop_nest.region:                             ; preds = %[[VAL_120]]
// CHECK:         store i32 %[[VAL_127]], ptr %[[VAL_103]], align 4
// CHECK:         call void @_QPbefore(ptr %[[VAL_106]])
// CHECK:         %[[VAL_129:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_130:.*]] = call i32 @__kmpc_cancellationpoint(ptr @1, i32 %[[VAL_129]], i32 4)
// CHECK:         %[[VAL_131:.*]] = icmp eq i32 %[[VAL_130]], 0
// CHECK:         br i1 %[[VAL_131]], label %[[VAL_132:omp.loop_nest.region.split]], label %[[VAL_133:omp.loop_nest.region.cncl]]
// CHECK:       omp.loop_nest.region.cncl:                        ; preds = %[[VAL_128]]
// CHECK:         br label %[[VAL_124]]
// CHECK:       .fini:                                            ; preds = %[[VAL_133]]
// CHECK:         br label %[[VAL_123]]
// CHECK:       omp.loop_nest.region.split:                       ; preds = %[[VAL_128]]
// CHECK:         call void @_QPafter(ptr %[[VAL_106]])
// CHECK:         br label %[[VAL_134:.*]]
// CHECK:       omp.region.cont3:                                 ; preds = %[[VAL_132]]
// CHECK:         br label %[[VAL_115]]
// CHECK:       omp_loop.inc:                                     ; preds = %[[VAL_134]]
// CHECK:         %[[VAL_117]] = add nuw i32 %[[VAL_116]], 1
// CHECK:         br label %[[VAL_114]]
// CHECK:       taskloop.exit.exitStub:                           ; preds = %[[VAL_123]]
// CHECK:         ret void

llvm.func @_QPbefore(!llvm.ptr) attributes {sym_visibility = "private"}
llvm.func @_QPafter(!llvm.ptr) attributes {sym_visibility = "private"}
llvm.func @_init(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
llvm.func @_copy(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
llvm.func @_dealloc(!llvm.ptr) attributes {sym_visibility = "private"}
