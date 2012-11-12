; RUN: llc < %s -march=x86 -mattr=+sse2 | grep andpd | not grep esp

declare double @llvm.sqrt.f64(double) nounwind readnone 

declare fastcc void @ApplyGivens(double**, double, double, i32, i32, i32, i32) nounwind 

declare double @fabs(double)

define void @main_bb114_2E_outer_2E_i_bb3_2E_i27(double** %tmp12.sub.i.i, [51 x double*]* %tmp12.i.i.i, i32 %i.0.reg2mem.0.ph.i, i32 %tmp11688.i, i32 %tmp19.i, i32 %tmp24.i, [51 x double*]* %tmp12.i.i) {
newFuncRoot:
	br label %bb3.i27

bb111.i77.bb121.i_crit_edge.exitStub:		; preds = %bb111.i77
	ret void

bb3.i27:		; preds = %bb111.i77.bb3.i27_crit_edge, %newFuncRoot
	%indvar94.i = phi i32 [ 0, %newFuncRoot ], [ %tmp113.i76, %bb111.i77.bb3.i27_crit_edge ]		; <i32> [#uses=6]
	%tmp6.i20 = getelementptr [51 x double*]* %tmp12.i.i, i32 0, i32 %indvar94.i		; <double**> [#uses=1]
	%tmp7.i21 = load double** %tmp6.i20, align 4		; <double*> [#uses=2]
	%tmp10.i = add i32 %indvar94.i, %i.0.reg2mem.0.ph.i		; <i32> [#uses=5]
	%tmp11.i22 = getelementptr double* %tmp7.i21, i32 %tmp10.i		; <double*> [#uses=1]
	%tmp12.i23 = load double* %tmp11.i22, align 8		; <double> [#uses=4]
	%tmp20.i24 = add i32 %tmp19.i, %indvar94.i		; <i32> [#uses=3]
	%tmp21.i = getelementptr double* %tmp7.i21, i32 %tmp20.i24		; <double*> [#uses=1]
	%tmp22.i25 = load double* %tmp21.i, align 8		; <double> [#uses=3]
	%tmp1.i.i26 = fcmp oeq double %tmp12.i23, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp1.i.i26, label %bb3.i27.Givens.exit.i49_crit_edge, label %bb5.i.i31

bb5.i.i31:		; preds = %bb3.i27
	%tmp7.i.i28 = call double @fabs( double %tmp12.i23 ) nounwind 		; <double> [#uses=1]
	%tmp9.i.i29 = call double @fabs( double %tmp22.i25 ) nounwind 		; <double> [#uses=1]
	%tmp10.i.i30 = fcmp ogt double %tmp7.i.i28, %tmp9.i.i29		; <i1> [#uses=1]
	br i1 %tmp10.i.i30, label %bb13.i.i37, label %bb30.i.i43

bb13.i.i37:		; preds = %bb5.i.i31
	%tmp15.i.i32 = fsub double -0.000000e+00, %tmp22.i25		; <double> [#uses=1]
	%tmp17.i.i33 = fdiv double %tmp15.i.i32, %tmp12.i23		; <double> [#uses=3]
	%tmp20.i4.i = fmul double %tmp17.i.i33, %tmp17.i.i33		; <double> [#uses=1]
	%tmp21.i.i34 = fadd double %tmp20.i4.i, 1.000000e+00		; <double> [#uses=1]
	%tmp22.i.i35 = call double @llvm.sqrt.f64( double %tmp21.i.i34 ) nounwind 		; <double> [#uses=1]
	%tmp23.i5.i = fdiv double 1.000000e+00, %tmp22.i.i35		; <double> [#uses=2]
	%tmp28.i.i36 = fmul double %tmp23.i5.i, %tmp17.i.i33		; <double> [#uses=1]
	br label %Givens.exit.i49

bb30.i.i43:		; preds = %bb5.i.i31
	%tmp32.i.i38 = fsub double -0.000000e+00, %tmp12.i23		; <double> [#uses=1]
	%tmp34.i.i39 = fdiv double %tmp32.i.i38, %tmp22.i25		; <double> [#uses=3]
	%tmp37.i6.i = fmul double %tmp34.i.i39, %tmp34.i.i39		; <double> [#uses=1]
	%tmp38.i.i40 = fadd double %tmp37.i6.i, 1.000000e+00		; <double> [#uses=1]
	%tmp39.i7.i = call double @llvm.sqrt.f64( double %tmp38.i.i40 ) nounwind 		; <double> [#uses=1]
	%tmp40.i.i41 = fdiv double 1.000000e+00, %tmp39.i7.i		; <double> [#uses=2]
	%tmp45.i.i42 = fmul double %tmp40.i.i41, %tmp34.i.i39		; <double> [#uses=1]
	br label %Givens.exit.i49

Givens.exit.i49:		; preds = %bb3.i27.Givens.exit.i49_crit_edge, %bb30.i.i43, %bb13.i.i37
	%s.0.i44 = phi double [ %tmp45.i.i42, %bb30.i.i43 ], [ %tmp23.i5.i, %bb13.i.i37 ], [ 0.000000e+00, %bb3.i27.Givens.exit.i49_crit_edge ]		; <double> [#uses=2]
	%c.0.i45 = phi double [ %tmp40.i.i41, %bb30.i.i43 ], [ %tmp28.i.i36, %bb13.i.i37 ], [ 1.000000e+00, %bb3.i27.Givens.exit.i49_crit_edge ]		; <double> [#uses=2]
	%tmp26.i46 = add i32 %tmp24.i, %indvar94.i		; <i32> [#uses=2]
	%tmp27.i47 = icmp slt i32 %tmp26.i46, 51		; <i1> [#uses=1]
	%min.i48 = select i1 %tmp27.i47, i32 %tmp26.i46, i32 50		; <i32> [#uses=1]
	call fastcc void @ApplyGivens( double** %tmp12.sub.i.i, double %s.0.i44, double %c.0.i45, i32 %tmp20.i24, i32 %tmp10.i, i32 %indvar94.i, i32 %min.i48 ) nounwind 
	br label %codeRepl

codeRepl:		; preds = %Givens.exit.i49
	call void @main_bb114_2E_outer_2E_i_bb3_2E_i27_bb_2E_i48_2E_i( i32 %tmp10.i, i32 %tmp20.i24, double %s.0.i44, double %c.0.i45, [51 x double*]* %tmp12.i.i.i )
	br label %ApplyRGivens.exit49.i

ApplyRGivens.exit49.i:		; preds = %codeRepl
	%tmp10986.i = icmp sgt i32 %tmp11688.i, %tmp10.i		; <i1> [#uses=1]
	br i1 %tmp10986.i, label %ApplyRGivens.exit49.i.bb52.i57_crit_edge, label %ApplyRGivens.exit49.i.bb111.i77_crit_edge

codeRepl1:		; preds = %ApplyRGivens.exit49.i.bb52.i57_crit_edge
	call void @main_bb114_2E_outer_2E_i_bb3_2E_i27_bb52_2E_i57( i32 %tmp10.i, double** %tmp12.sub.i.i, [51 x double*]* %tmp12.i.i.i, i32 %i.0.reg2mem.0.ph.i, i32 %tmp11688.i, i32 %tmp19.i, i32 %tmp24.i, [51 x double*]* %tmp12.i.i )
	br label %bb105.i.bb111.i77_crit_edge

bb111.i77:		; preds = %bb105.i.bb111.i77_crit_edge, %ApplyRGivens.exit49.i.bb111.i77_crit_edge
	%tmp113.i76 = add i32 %indvar94.i, 1		; <i32> [#uses=2]
	%tmp118.i = icmp sgt i32 %tmp11688.i, %tmp113.i76		; <i1> [#uses=1]
	br i1 %tmp118.i, label %bb111.i77.bb3.i27_crit_edge, label %bb111.i77.bb121.i_crit_edge.exitStub

bb3.i27.Givens.exit.i49_crit_edge:		; preds = %bb3.i27
	br label %Givens.exit.i49

ApplyRGivens.exit49.i.bb52.i57_crit_edge:		; preds = %ApplyRGivens.exit49.i
	br label %codeRepl1

ApplyRGivens.exit49.i.bb111.i77_crit_edge:		; preds = %ApplyRGivens.exit49.i
	br label %bb111.i77

bb105.i.bb111.i77_crit_edge:		; preds = %codeRepl1
	br label %bb111.i77

bb111.i77.bb3.i27_crit_edge:		; preds = %bb111.i77
	br label %bb3.i27
}

declare void @main_bb114_2E_outer_2E_i_bb3_2E_i27_bb_2E_i48_2E_i(i32, i32, double, double, [51 x double*]*)

declare void @main_bb114_2E_outer_2E_i_bb3_2E_i27_bb52_2E_i57(i32, double**, [51 x double*]*, i32, i32, i32, i32, [51 x double*]*)
