; RUN: opt %loadPolly %defaultOpts -polly-prepare -polly-cloog -analyze  %s | FileCheck %s
; XFAIL: *
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@nrm = common global double 0.000000e+00, align 8
@A = common global [512 x [512 x double]] zeroinitializer, align 16
@R = common global [512 x [512 x double]] zeroinitializer, align 16
@Q = common global [512 x [512 x double]] zeroinitializer, align 16

define void @scop_func(i64 %m, i64 %n) nounwind {
bb:
  %tmp3 = icmp sgt i64 %m, 0
  %smax4 = select i1 %tmp3, i64 %m, i64 0
  %tmp = icmp sgt i64 %m, 0
  %smax = select i1 %tmp, i64 %m, i64 0
  %tmp12 = icmp sgt i64 %m, 0
  %smax13 = select i1 %tmp12, i64 %m, i64 0
  %tmp25 = icmp sgt i64 %m, 0
  %smax26 = select i1 %tmp25, i64 %m, i64 0
  %tmp40 = add i64 %n, -1
  %tmp60 = icmp sgt i64 %n, 0
  %smax61 = select i1 %tmp60, i64 %n, i64 0
  br label %bb1

bb1:                                              ; preds = %bb58, %bb
  %tmp2 = phi i64 [ 0, %bb ], [ %tmp59, %bb58 ]
  %tmp63 = mul i64 %tmp2, 513
  %tmp64 = add i64 %tmp63, 1
  %tmp67 = add i64 %tmp2, 1
  %tmp71 = mul i64 %tmp2, -1
  %tmp45 = add i64 %tmp40, %tmp71
  %scevgep50 = getelementptr [512 x [512 x double]]* @R, i64 0, i64 0, i64 %tmp63
  %exitcond62 = icmp ne i64 %tmp2, %smax61
  br i1 %exitcond62, label %bb3, label %bb60

bb3:                                              ; preds = %bb1
  store double 0.000000e+00, double* @nrm, align 8
  br label %bb4

bb4:                                              ; preds = %bb12, %bb3
  %i.0 = phi i64 [ 0, %bb3 ], [ %tmp14, %bb12 ]
  %scevgep = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.0, i64 %tmp2
  %exitcond5 = icmp ne i64 %i.0, %smax4
  br i1 %exitcond5, label %bb5, label %bb15

bb5:                                              ; preds = %bb4
  %tmp7 = load double* %scevgep
  %tmp8 = load double* %scevgep
  %tmp9 = fmul double %tmp7, %tmp8
  %tmp10 = load double* @nrm, align 8
  %tmp11 = fadd double %tmp10, %tmp9
  store double %tmp11, double* @nrm, align 8
  br label %bb12

bb12:                                             ; preds = %bb5
  %tmp14 = add nsw i64 %i.0, 1
  br label %bb4

bb15:                                             ; preds = %bb4
  %tmp16 = load double* @nrm, align 8
  %tmp17 = call double @sqrt(double %tmp16)
  store double %tmp17, double* %scevgep50
  br label %bb18

bb18:                                             ; preds = %bb25, %bb15
  %i.1 = phi i64 [ 0, %bb15 ], [ %tmp26, %bb25 ]
  %scevgep5 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.1, i64 %tmp2
  %scevgep4 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.1, i64 %tmp2
  %exitcond = icmp ne i64 %i.1, %smax
  br i1 %exitcond, label %bb19, label %bb27

bb19:                                             ; preds = %bb18
  %tmp21 = load double* %scevgep5
  %tmp23 = load double* %scevgep50
  %tmp24 = fdiv double %tmp21, %tmp23
  store double %tmp24, double* %scevgep4
  br label %bb25

bb25:                                             ; preds = %bb19
  %tmp26 = add nsw i64 %i.1, 1
  br label %bb18

bb27:                                             ; preds = %bb18
  br label %bb28

bb28:                                             ; preds = %bb56, %bb27
  %indvar = phi i64 [ %indvar.next, %bb56 ], [ 0, %bb27 ]
  %tmp65 = add i64 %tmp64, %indvar
  %scevgep31 = getelementptr [512 x [512 x double]]* @R, i64 0, i64 0, i64 %tmp65
  %tmp68 = add i64 %tmp67, %indvar
  %exitcond46 = icmp ne i64 %indvar, %tmp45
  br i1 %exitcond46, label %bb29, label %bb57

bb29:                                             ; preds = %bb28
  store double 0.000000e+00, double* %scevgep31
  br label %bb30

bb30:                                             ; preds = %bb43, %bb29
  %i.2 = phi i64 [ 0, %bb29 ], [ %tmp44, %bb43 ]
  %scevgep11 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.2, i64 %tmp68
  %scevgep12 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.2, i64 %tmp2
  %exitcond14 = icmp ne i64 %i.2, %smax13
  br i1 %exitcond14, label %bb31, label %bb45

bb31:                                             ; preds = %bb30
  %tmp33 = load double* %scevgep12
  %tmp34 = load double* %scevgep11
  %tmp38 = fmul double %tmp33, %tmp34
  %tmp41 = load double* %scevgep31
  %tmp42 = fadd double %tmp41, %tmp38
  store double %tmp42, double* %scevgep31
  br label %bb43

bb43:                                             ; preds = %bb31
  %tmp44 = add nsw i64 %i.2, 1
  br label %bb30

bb45:                                             ; preds = %bb30
  br label %bb46

bb46:                                             ; preds = %bb53, %bb45
  %i.3 = phi i64 [ 0, %bb45 ], [ %tmp54, %bb53 ]
  %scevgep18 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %i.3, i64 %tmp68
  %scevgep19 = getelementptr [512 x [512 x double]]* @Q, i64 0, i64 %i.3, i64 %tmp2
  %exitcond27 = icmp ne i64 %i.3, %smax26
  br i1 %exitcond27, label %bb47, label %bb55

bb47:                                             ; preds = %bb46
  %tmp48 = load double* %scevgep18
  %tmp49 = load double* %scevgep19
  %tmp50 = load double* %scevgep31
  %tmp51 = fmul double %tmp49, %tmp50
  %tmp52 = fsub double %tmp48, %tmp51
  store double %tmp52, double* %scevgep18
  br label %bb53

bb53:                                             ; preds = %bb47
  %tmp54 = add nsw i64 %i.3, 1
  br label %bb46

bb55:                                             ; preds = %bb46
  br label %bb56

bb56:                                             ; preds = %bb55
  %indvar.next = add i64 %indvar, 1
  br label %bb28

bb57:                                             ; preds = %bb28
  br label %bb58

bb58:                                             ; preds = %bb57
  %tmp59 = add nsw i64 %tmp2, 1
  br label %bb1

bb60:                                             ; preds = %bb1
  ret void
}

declare double @sqrt(double) nounwind readnone

define i32 @main(i32 %argc, i8** %argv) nounwind {
bb:
  call void (...)* @init_array()
  %tmp = sext i32 512 to i64
  %tmp1 = sext i32 512 to i64
  call void @scop_func(i64 %tmp, i64 %tmp1)
  call void @print_array(i32 %argc, i8** %argv)
  ret i32 0
}

declare void @init_array(...)

declare void @print_array(i32, i8**)
