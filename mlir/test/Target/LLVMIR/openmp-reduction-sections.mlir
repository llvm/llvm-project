// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.declare_reduction @add_reduction_f32 : f32 init {
^bb0(%arg0: f32):
  %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  omp.yield(%0 : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %0 = llvm.fadd %arg0, %arg1  {fastmathFlags = #llvm.fastmath<contract>} : f32
  omp.yield(%0 : f32)
}  
llvm.func @sections_(%arg0: !llvm.ptr {fir.bindc_name = "x"}) attributes {fir.internal_name = "_QPsections"} {
  %0 = llvm.mlir.constant(2.000000e+00 : f32) : f32
  %1 = llvm.mlir.constant(1.000000e+00 : f32) : f32
  omp.parallel {
    omp.sections reduction(@add_reduction_f32 %arg0 -> %arg1 : !llvm.ptr) {
      omp.section {
      ^bb0(%arg2: !llvm.ptr):
        %2 = llvm.load %arg2 : !llvm.ptr -> f32
        %3 = llvm.fadd %2, %1  {fastmathFlags = #llvm.fastmath<contract>} : f32
        llvm.store %3, %arg2 : f32, !llvm.ptr
        omp.terminator
      }
      omp.section {
      ^bb0(%arg2: !llvm.ptr):
        %2 = llvm.load %arg2 : !llvm.ptr -> f32
        %3 = llvm.fadd %2, %0  {fastmathFlags = #llvm.fastmath<contract>} : f32
        llvm.store %3, %arg2 : f32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @sections_..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_9:.*]] = getelementptr { ptr }, ptr %[[VAL_10:.*]], i32 0, i32 0
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_12:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_13:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_14:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_15:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_16:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_17:.*]] = load i32, ptr %[[VAL_18:.*]], align 4
// CHECK:         store i32 %[[VAL_17]], ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_19:.*]] = load i32, ptr %[[VAL_16]], align 4
// CHECK:         %[[VAL_20:.*]] = alloca float, align 4
// CHECK:         %[[VAL_21:.*]] = alloca [1 x ptr], align 8
// CHECK:         br label %[[VAL_22:.*]]
// CHECK:       omp.reduction.init:                               ; preds = %[[VAL_23:.*]]
// CHECK:         br label %[[VAL_24:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_22]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_24]]
// CHECK:         store float 0.000000e+00, ptr %[[VAL_20]], align 4
// CHECK:         br label %[[VAL_26:.*]]
// CHECK:       omp_section_loop.preheader:                       ; preds = %[[VAL_25]]
// CHECK:         store i32 0, ptr %[[VAL_13]], align 4
// CHECK:         store i32 1, ptr %[[VAL_14]], align 4
// CHECK:         store i32 1, ptr %[[VAL_15]], align 4
// CHECK:         %[[VAL_27:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @1, i32 %[[VAL_27]], i32 34, ptr %[[VAL_12]], ptr %[[VAL_13]], ptr %[[VAL_14]], ptr %[[VAL_15]], i32 1, i32 0)
// CHECK:         %[[VAL_28:.*]] = load i32, ptr %[[VAL_13]], align 4
// CHECK:         %[[VAL_29:.*]] = load i32, ptr %[[VAL_14]], align 4
// CHECK:         %[[VAL_30:.*]] = sub i32 %[[VAL_29]], %[[VAL_28]]
// CHECK:         %[[VAL_31:.*]] = add i32 %[[VAL_30]], 1
// CHECK:         br label %[[VAL_32:.*]]
// CHECK:       omp_section_loop.header:                          ; preds = %[[VAL_33:.*]], %[[VAL_26]]
// CHECK:         %[[VAL_34:.*]] = phi i32 [ 0, %[[VAL_26]] ], [ %[[VAL_35:.*]], %[[VAL_33]] ]
// CHECK:         br label %[[VAL_36:.*]]
// CHECK:       omp_section_loop.cond:                            ; preds = %[[VAL_32]]
// CHECK:         %[[VAL_37:.*]] = icmp ult i32 %[[VAL_34]], %[[VAL_31]]
// CHECK:         br i1 %[[VAL_37]], label %[[VAL_38:.*]], label %[[VAL_39:.*]]
// CHECK:       omp_section_loop.exit:                            ; preds = %[[VAL_36]]
// CHECK:         call void @__kmpc_for_static_fini(ptr @1, i32 %[[VAL_27]])
// CHECK:         %[[VAL_40:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_40]])
// CHECK:         br label %[[VAL_41:.*]]
// CHECK:       omp_section_loop.after:                           ; preds = %[[VAL_39]]
// CHECK:         br label %[[VAL_42:.*]]
// CHECK:       omp_section_loop.aftersections.fini:              ; preds = %[[VAL_41]]
// CHECK:         %[[VAL_43:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_21]], i64 0, i64 0
// CHECK:         store ptr %[[VAL_20]], ptr %[[VAL_43]], align 8
// CHECK:         %[[VAL_44:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_45:.*]] = call i32 @__kmpc_reduce(ptr @1, i32 %[[VAL_44]], i32 1, i64 8, ptr %[[VAL_21]], ptr @.omp.reduction.func, ptr @.gomp_critical_user_.reduction.var)
// CHECK:         switch i32 %[[VAL_45]], label %[[VAL_46:.*]] [
// CHECK:           i32 1, label %[[VAL_47:.*]]
// CHECK:           i32 2, label %[[VAL_48:.*]]
// CHECK:         ]
// CHECK:       reduce.switch.atomic:                             ; preds = %[[VAL_42]]
// CHECK:         unreachable
// CHECK:       reduce.switch.nonatomic:                          ; preds = %[[VAL_42]]
// CHECK:         %[[VAL_49:.*]] = load float, ptr %[[VAL_11]], align 4
// CHECK:         %[[VAL_50:.*]] = load float, ptr %[[VAL_20]], align 4
// CHECK:         %[[VAL_51:.*]] = fadd contract float %[[VAL_49]], %[[VAL_50]]
// CHECK:         store float %[[VAL_51]], ptr %[[VAL_11]], align 4
// CHECK:         call void @__kmpc_end_reduce(ptr @1, i32 %[[VAL_44]], ptr @.gomp_critical_user_.reduction.var)
// CHECK:         br label %[[VAL_46]]
// CHECK:       reduce.finalize:                                  ; preds = %[[VAL_47]], %[[VAL_42]]
// CHECK:         %[[VAL_52:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_52]])
// CHECK:         br label %[[VAL_53:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_46]]
// CHECK:         br label %[[VAL_54:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_53]]
// CHECK:         br label %[[VAL_55:.*]]
// CHECK:       omp_section_loop.body:                            ; preds = %[[VAL_36]]
// CHECK:         %[[VAL_56:.*]] = add i32 %[[VAL_34]], %[[VAL_28]]
// CHECK:         %[[VAL_57:.*]] = mul i32 %[[VAL_56]], 1
// CHECK:         %[[VAL_58:.*]] = add i32 %[[VAL_57]], 0
// CHECK:         switch i32 %[[VAL_58]], label %[[VAL_59:.*]] [
// CHECK:           i32 0, label %[[VAL_60:.*]]
// CHECK:           i32 1, label %[[VAL_61:.*]]
// CHECK:         ]
// CHECK:       omp_section_loop.body.case3:                      ; preds = %[[VAL_38]]
// CHECK:         br label %[[VAL_62:.*]]
// CHECK:       omp.section.region5:                              ; preds = %[[VAL_61]]
// CHECK:         %[[VAL_63:.*]] = load float, ptr %[[VAL_20]], align 4
// CHECK:         %[[VAL_64:.*]] = fadd contract float %[[VAL_63]], 2.000000e+00
// CHECK:         store float %[[VAL_64]], ptr %[[VAL_20]], align 4
// CHECK:         br label %[[VAL_65:.*]]
// CHECK:       omp.region.cont4:                                 ; preds = %[[VAL_62]]
// CHECK:         br label %[[VAL_59]]
// CHECK:       omp_section_loop.body.case:                       ; preds = %[[VAL_38]]
// CHECK:         br label %[[VAL_66:.*]]
// CHECK:       omp.section.region:                               ; preds = %[[VAL_60]]
// CHECK:         %[[VAL_67:.*]] = load float, ptr %[[VAL_20]], align 4
// CHECK:         %[[VAL_68:.*]] = fadd contract float %[[VAL_67]], 1.000000e+00
// CHECK:         store float %[[VAL_68]], ptr %[[VAL_20]], align 4
// CHECK:         br label %[[VAL_69:.*]]
// CHECK:       omp.region.cont2:                                 ; preds = %[[VAL_66]]
// CHECK:         br label %[[VAL_59]]
// CHECK:       omp_section_loop.body.sections.after:             ; preds = %[[VAL_65]], %[[VAL_69]], %[[VAL_38]]
// CHECK:         br label %[[VAL_33]]
// CHECK:       omp_section_loop.inc:                             ; preds = %[[VAL_59]]
// CHECK:         %[[VAL_35]] = add nuw i32 %[[VAL_34]], 1
// CHECK:         br label %[[VAL_32]]
// CHECK:       omp.par.outlined.exit.exitStub:                   ; preds = %[[VAL_54]]
// CHECK:         ret void
// CHECK:         %[[VAL_70:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_71:.*]], i64 0, i64 0
// CHECK:         %[[VAL_72:.*]] = load ptr, ptr %[[VAL_70]], align 8
// CHECK:         %[[VAL_73:.*]] = load float, ptr %[[VAL_72]], align 4
// CHECK:         %[[VAL_74:.*]] = getelementptr inbounds [1 x ptr], ptr %[[VAL_75:.*]], i64 0, i64 0
// CHECK:         %[[VAL_76:.*]] = load ptr, ptr %[[VAL_74]], align 8
// CHECK:         %[[VAL_77:.*]] = load float, ptr %[[VAL_76]], align 4
// CHECK:         %[[VAL_78:.*]] = fadd contract float %[[VAL_73]], %[[VAL_77]]
// CHECK:         store float %[[VAL_78]], ptr %[[VAL_72]], align 4
// CHECK:         ret void
