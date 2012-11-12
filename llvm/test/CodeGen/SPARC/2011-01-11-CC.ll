; RUN: llc -march=sparc <%s | FileCheck %s -check-prefix=V8
; RUN: llc -march=sparc -mattr=v9 <%s | FileCheck %s -check-prefix=V9


define i32 @test_addx(i64 %a, i64 %b, i64 %c) nounwind readnone noinline {
entry:
; V8: addcc
; V8-NOT: subcc
; V8: addx
; V9: addcc
; V9-NOT: subcc
; V9: addx
; V9: mov{{e|ne}} %icc
  %0 = add i64 %a, %b
  %1 = icmp ugt i64 %0, %c
  %2 = zext i1 %1 to i32
  ret i32 %2
}


define i32 @test_select_int_icc(i32 %a, i32 %b, i32 %c) nounwind readnone noinline {
entry:
; V8: test_select_int_icc
; V8: subcc
; V8: {{be|bne}}
; V9: test_select_int_icc
; V9: subcc
; V9-NOT: {{be|bne}}
; V9: mov{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, i32 %b, i32 %c
  ret i32 %1
}


define float @test_select_fp_icc(i32 %a, float %f1, float %f2) nounwind readnone noinline {
entry:
; V8: test_select_fp_icc
; V8: subcc
; V8: {{be|bne}}
; V9: test_select_fp_icc
; V9: subcc
; V9-NOT: {{be|bne}}
; V9: fmovs{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_icc(i32 %a, double %f1, double %f2) nounwind readnone noinline {
entry:
; V8: test_select_dfp_icc
; V8: subcc
; V8: {{be|bne}}
; V9: test_select_dfp_icc
; V9: subcc
; V9-NOT: {{be|bne}}
; V9: fmovd{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}

define i32 @test_select_int_fcc(float %f, i32 %a, i32 %b) nounwind readnone noinline {
entry:
;V8: test_select_int_fcc
;V8: fcmps
;V8: {{fbe|fbne}}
;V9: test_select_int_fcc
;V9: fcmps
;V9-NOT: {{fbe|fbne}}
;V9: mov{{e|ne}} %fcc0
  %0 = fcmp une float %f, 0.000000e+00
  %a.b = select i1 %0, i32 %a, i32 %b
  ret i32 %a.b
}


define float @test_select_fp_fcc(float %f, float %f1, float %f2) nounwind readnone noinline {
entry:
;V8: test_select_fp_fcc
;V8: fcmps
;V8: {{fbe|fbne}}
;V9: test_select_fp_fcc
;V9: fcmps
;V9-NOT: {{fbe|fbne}}
;V9: fmovs{{e|ne}} %fcc0
  %0 = fcmp une float %f, 0.000000e+00
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_fcc(double %f, double %f1, double %f2) nounwind readnone noinline {
entry:
;V8: test_select_dfp_fcc
;V8: fcmpd
;V8: {{fbne|fbe}}
;V9: test_select_dfp_fcc
;V9: fcmpd
;V9-NOT: {{fbne|fbe}}
;V9: fmovd{{e|ne}} %fcc0
  %0 = fcmp une double %f, 0.000000e+00
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}
