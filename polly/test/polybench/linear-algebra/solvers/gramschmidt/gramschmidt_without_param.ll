; RUN: opt %loadPolly %defaultOpts -polly-prepare -polly-cloog -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@nrm = common global double 0.000000e+00, align 8
@A = common global [512 x [512 x double]] zeroinitializer, align 16
@R = common global [512 x [512 x double]] zeroinitializer, align 16
@Q = common global [512 x [512 x double]] zeroinitializer, align 16

define void @scop_func() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb51, %bb
  %tmp2 = phi i64 [ 0, %bb ], [ %tmp52, %bb51 ]
  %tmp49 = mul i64 %tmp2, 513
  %tmp50 = add i64 %tmp49, 1
  %tmp53 = add i64 %tmp2, 1
  %tmp57 = mul i64 %tmp2, -1
  %tmp28 = add i64 %tmp57, 511
  %scevgep39 = getelementptr [512 x [512 x double]]* @R, i64 0, i64 0, i64 %tmp49
  %exitcond48 = icmp ne i64 %tmp2, 512
  br i1 %exitcond48, label %bb3, label %bb53

bb3:                                              ; preds = %bb1
  store double 0.000000e+00, double* @nrm, align 8
  br label %bb4

bb4:                                              ; preds = %bb11, %bb3
  %i.0 = phi i64 [ 0, %bb3 ], [ %tmp12, %bb11 ]
  %scevgep = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.0, i64 %tmp2
  %exitcond2 = icmp ne i64 %i.0, 512
  br i1 %exitcond2, label %bb5, label %bb13

bb5:                                              ; preds = %bb4
  %tmp6 = load double* %scevgep
  %tmp7 = load double* %scevgep
  %tmp8 = fmul double %tmp6, %tmp7
  %tmp9 = load double* @nrm, align 8
  %tmp10 = fadd double %tmp9, %tmp8
  store double %tmp10, double* @nrm, align 8
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nsw i64 %i.0, 1
  br label %bb4

bb13:                                             ; preds = %bb4
  %tmp15 = load double* @nrm, align 8
  %tmp16 = call double @sqrt(double %tmp15)
  store double %tmp16, double* %scevgep39
  br label %bb17

bb17:                                             ; preds = %bb22, %bb13
  %i.1 = phi i64 [ 0, %bb13 ], [ %tmp23, %bb22 ]
  %scevgep3 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.1, i64 %tmp2
  %scevgep2 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.1, i64 %tmp2
  %exitcond = icmp ne i64 %i.1, 512
  br i1 %exitcond, label %bb18, label %bb24

bb18:                                             ; preds = %bb17
  %tmp19 = load double* %scevgep3
  %tmp20 = load double* %scevgep39
  %tmp21 = fdiv double %tmp19, %tmp20
  store double %tmp21, double* %scevgep2
  br label %bb22

bb22:                                             ; preds = %bb18
  %tmp23 = add nsw i64 %i.1, 1
  br label %bb17

bb24:                                             ; preds = %bb17
  br label %bb25

bb25:                                             ; preds = %bb49, %bb24
  %indvar = phi i64 [ %indvar.next, %bb49 ], [ 0, %bb24 ]
  %tmp51 = add i64 %tmp50, %indvar
  %scevgep23 = getelementptr [512 x [512 x double]]* @R, i64 0, i64 0, i64 %tmp51
  %tmp54 = add i64 %tmp53, %indvar
  %exitcond29 = icmp ne i64 %indvar, %tmp28
  br i1 %exitcond29, label %bb26, label %bb50

bb26:                                             ; preds = %bb25
  store double 0.000000e+00, double* %scevgep23
  br label %bb27

bb27:                                             ; preds = %bb36, %bb26
  %i.2 = phi i64 [ 0, %bb26 ], [ %tmp37, %bb36 ]
  %scevgep6 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.2, i64 %tmp54
  %scevgep7 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.2, i64 %tmp2
  %exitcond9 = icmp ne i64 %i.2, 512
  br i1 %exitcond9, label %bb28, label %bb38

bb28:                                             ; preds = %bb27
  %tmp30 = load double* %scevgep7
  %tmp31 = load double* %scevgep6
  %tmp33 = fmul double %tmp30, %tmp31
  %tmp34 = load double* %scevgep23
  %tmp35 = fadd double %tmp34, %tmp33
  store double %tmp35, double* %scevgep23
  br label %bb36

bb36:                                             ; preds = %bb28
  %tmp37 = add nsw i64 %i.2, 1
  br label %bb27

bb38:                                             ; preds = %bb27
  br label %bb39

bb39:                                             ; preds = %bb46, %bb38
  %i.3 = phi i64 [ 0, %bb38 ], [ %tmp47, %bb46 ]
  %scevgep11 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.3, i64 %tmp54
  %scevgep12 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.3, i64 %tmp2
  %exitcond16 = icmp ne i64 %i.3, 512
  br i1 %exitcond16, label %bb40, label %bb48

bb40:                                             ; preds = %bb39
  %tmp41 = load double* %scevgep11
  %tmp42 = load double* %scevgep12
  %tmp43 = load double* %scevgep23
  %tmp44 = fmul double %tmp42, %tmp43
  %tmp45 = fsub double %tmp41, %tmp44
  store double %tmp45, double* %scevgep11
  br label %bb46

bb46:                                             ; preds = %bb40
  %tmp47 = add nsw i64 %i.3, 1
  br label %bb39

bb48:                                             ; preds = %bb39
  br label %bb49

bb49:                                             ; preds = %bb48
  %indvar.next = add i64 %indvar, 1
  br label %bb25

bb50:                                             ; preds = %bb25
  br label %bb51

bb51:                                             ; preds = %bb50
  %tmp52 = add nsw i64 %tmp2, 1
  br label %bb1

bb53:                                             ; preds = %bb1
  ret void
}

declare double @sqrt(double) nounwind readnone

define i32 @main(i32 %argc, i8** %argv) nounwind {
bb:
  call void (...)* @init_array()
  call void @scop_func()
  call void @print_array(i32 %argc, i8** %argv)
  ret i32 0
}

declare void @init_array(...)

declare void @print_array(i32, i8**)
; CHECK: for region: 'bb1 => bb53' in function 'scop_func':
