; RUN: opt < %s -instcombine -S | FileCheck %s
; <rdar://problem/10889741>

define void @func(double %r, double %g, double %b, double* %outH, double* %outS, double* %outL) nounwind uwtable ssp {
bb:
  %tmp = alloca double, align 8
  %tmp1 = alloca double, align 8
  %tmp2 = alloca double, align 8
  store double %r, double* %tmp, align 8
  store double %g, double* %tmp1, align 8
  store double %b, double* %tmp2, align 8
  %tmp3 = fcmp ogt double %r, %g
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb
  %tmp5 = fcmp ogt double %r, %b
  br i1 %tmp5, label %bb6, label %bb7

bb6:                                              ; preds = %bb4
  br label %bb12

bb7:                                              ; preds = %bb4
  br label %bb12

bb8:                                              ; preds = %bb
  %tmp9 = fcmp ogt double %g, %b
  br i1 %tmp9, label %bb10, label %bb11

bb10:                                             ; preds = %bb8
  br label %bb12

bb11:                                             ; preds = %bb8
  br label %bb12

bb12:                                             ; preds = %bb11, %bb10, %bb7, %bb6
  %max.0 = phi double* [ %tmp, %bb6 ], [ %tmp2, %bb7 ], [ %tmp1, %bb10 ], [ %tmp2, %bb11 ]
; CHECK: %tmp13 = load double* %tmp, align 8
; CHECK: %tmp14 = load double* %tmp1, align 8
; CHECK: %tmp15 = fcmp olt double %tmp13, %tmp14
  %tmp13 = load double* %tmp, align 8
  %tmp14 = load double* %tmp1, align 8
  %tmp15 = fcmp olt double %tmp13, %tmp14
  br i1 %tmp15, label %bb16, label %bb21

bb16:                                             ; preds = %bb12
  %tmp17 = load double* %tmp2, align 8
  %tmp18 = fcmp olt double %tmp13, %tmp17
  br i1 %tmp18, label %bb19, label %bb20

bb19:                                             ; preds = %bb16
  br label %bb26

bb20:                                             ; preds = %bb16
  br label %bb26

bb21:                                             ; preds = %bb12
  %tmp22 = load double* %tmp2, align 8
  %tmp23 = fcmp olt double %tmp14, %tmp22
  br i1 %tmp23, label %bb24, label %bb25

bb24:                                             ; preds = %bb21
  br label %bb26

bb25:                                             ; preds = %bb21
  br label %bb26

bb26:                                             ; preds = %bb25, %bb24, %bb20, %bb19
  %min.0 = phi double* [ %tmp, %bb19 ], [ %tmp2, %bb20 ], [ %tmp1, %bb24 ], [ %tmp2, %bb25 ]
; CHECK: %tmp27 = load double* %min.0, align 8
; CHECK: %tmp28 = load double* %max.0
; CHECK: %tmp29 = fadd double %tmp27, %tmp28
  %tmp27 = load double* %min.0, align 8
  %tmp28 = load double* %max.0
  %tmp29 = fadd double %tmp27, %tmp28
  %tmp30 = fdiv double %tmp29, 2.000000e+00
  store double %tmp30, double* %outL
  %tmp31 = load double* %min.0
  %tmp32 = load double* %max.0
  %tmp33 = fcmp oeq double %tmp31, %tmp32
  br i1 %tmp33, label %bb34, label %bb35

bb34:                                             ; preds = %bb26
  store double 0.000000e+00, double* %outS
  store double 0.000000e+00, double* %outH
  br label %bb81

bb35:                                             ; preds = %bb26
  %tmp36 = fcmp olt double %tmp30, 5.000000e-01
  %tmp37 = fsub double %tmp32, %tmp31
  br i1 %tmp36, label %bb38, label %bb41

bb38:                                             ; preds = %bb35
  %tmp39 = fadd double %tmp32, %tmp31
  %tmp40 = fdiv double %tmp37, %tmp39
  store double %tmp40, double* %outS
  br label %bb45

bb41:                                             ; preds = %bb35
  %tmp42 = fsub double 2.000000e+00, %tmp32
  %tmp43 = fsub double %tmp42, %tmp31
  %tmp44 = fdiv double %tmp37, %tmp43
  store double %tmp44, double* %outS
  br label %bb45

bb45:                                             ; preds = %bb41, %bb38
  %tmp46 = icmp eq double* %max.0, %tmp
  br i1 %tmp46, label %bb47, label %bb55

bb47:                                             ; preds = %bb45
  %tmp48 = load double* %tmp1, align 8
  %tmp49 = load double* %tmp2, align 8
  %tmp50 = fsub double %tmp48, %tmp49
  %tmp51 = load double* %max.0
  %tmp52 = load double* %min.0
  %tmp53 = fsub double %tmp51, %tmp52
  %tmp54 = fdiv double %tmp50, %tmp53
  store double %tmp54, double* %outH
  br label %bb75

bb55:                                             ; preds = %bb45
  %tmp56 = icmp eq double* %max.0, %tmp1
  br i1 %tmp56, label %bb57, label %bb66

bb57:                                             ; preds = %bb55
  %tmp58 = load double* %tmp2, align 8
  %tmp59 = load double* %tmp, align 8
  %tmp60 = fsub double %tmp58, %tmp59
  %tmp61 = load double* %max.0
  %tmp62 = load double* %min.0
  %tmp63 = fsub double %tmp61, %tmp62
  %tmp64 = fdiv double %tmp60, %tmp63
  %tmp65 = fadd double 2.000000e+00, %tmp64
  store double %tmp65, double* %outH
  br label %bb75

bb66:                                             ; preds = %bb55
  %tmp67 = load double* %tmp, align 8
  %tmp68 = load double* %tmp1, align 8
  %tmp69 = fsub double %tmp67, %tmp68
  %tmp70 = load double* %max.0
  %tmp71 = load double* %min.0
  %tmp72 = fsub double %tmp70, %tmp71
  %tmp73 = fdiv double %tmp69, %tmp72
  %tmp74 = fadd double 4.000000e+00, %tmp73
  store double %tmp74, double* %outH
  br label %bb75

bb75:                                             ; preds = %bb66, %bb57, %bb47
  %tmp76 = load double* %outH
  %tmp77 = fdiv double %tmp76, 6.000000e+00
  store double %tmp77, double* %outH
  %tmp78 = fcmp olt double %tmp77, 0.000000e+00
  br i1 %tmp78, label %bb79, label %bb81

bb79:                                             ; preds = %bb75
  %tmp80 = fadd double %tmp77, 1.000000e+00
  store double %tmp80, double* %outH
  br label %bb81

bb81:                                             ; preds = %bb79, %bb75, %bb34
  ret void
}
