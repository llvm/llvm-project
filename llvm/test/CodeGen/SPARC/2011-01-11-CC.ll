; RUN: llc -march=sparc <%s | FileCheck %s -check-prefix=V8
; RUN: llc -march=sparc -mattr=v9 <%s | FileCheck %s -check-prefix=V9
; RUN: llc -mtriple=sparc64-unknown-linux <%s | FileCheck %s -check-prefix=SPARC64


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
; V8: cmp
; V8: {{be|bne}}
; V9: test_select_int_icc
; V9: cmp
; V9-NOT: {{be|bne}}
; V9: mov{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, i32 %b, i32 %c
  ret i32 %1
}


define float @test_select_fp_icc(i32 %a, float %f1, float %f2) nounwind readnone noinline {
entry:
; V8: test_select_fp_icc
; V8: cmp
; V8: {{be|bne}}
; V9: test_select_fp_icc
; V9: cmp
; V9-NOT: {{be|bne}}
; V9: fmovs{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_icc(i32 %a, double %f1, double %f2) nounwind readnone noinline {
entry:
; V8: test_select_dfp_icc
; V8: cmp
; V8: {{be|bne}}
; V9: test_select_dfp_icc
; V9: cmp
; V9-NOT: {{be|bne}}
; V9: fmovd{{e|ne}} %icc
  %0 = icmp eq i32 %a, 0
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}

define i32 @test_select_int_fcc(float %f, i32 %a, i32 %b) nounwind readnone noinline {
entry:
;V8-LABEL: test_select_int_fcc:
;V8: fcmps
;V8-NEXT: nop
;V8: {{fbe|fbne}}
;V9-LABEL: test_select_int_fcc:
;V9: fcmps
;V9-NOT: nop
;V9-NOT: {{fbe|fbne}}
;V9: mov{{e|ne}} %fcc0
  %0 = fcmp une float %f, 0.000000e+00
  %a.b = select i1 %0, i32 %a, i32 %b
  ret i32 %a.b
}


define float @test_select_fp_fcc(float %f, float %f1, float %f2) nounwind readnone noinline {
entry:
;V8-LABEL: test_select_fp_fcc:
;V8: fcmps
;V8: {{fbe|fbne}}
;V9-LABEL: test_select_fp_fcc:
;V9: fcmps
;V9-NOT: {{fbe|fbne}}
;V9: fmovs{{e|ne}} %fcc0
  %0 = fcmp une float %f, 0.000000e+00
  %1 = select i1 %0, float %f1, float %f2
  ret float %1
}

define double @test_select_dfp_fcc(double %f, double %f1, double %f2) nounwind readnone noinline {
entry:
;V8-LABEL: test_select_dfp_fcc:
;V8: fcmpd
;V8-NEXT: nop
;V8: {{fbne|fbe}}
;V9-LABEL: test_select_dfp_fcc:
;V9: fcmpd
;V9-NOT: nop
;V9-NOT: {{fbne|fbe}}
;V9: fmovd{{e|ne}} %fcc0
  %0 = fcmp une double %f, 0.000000e+00
  %1 = select i1 %0, double %f1, double %f2
  ret double %1
}

define i32 @test_float_cc(double %a, double %b, i32 %c, i32 %d) {
entry:
; V8-LABEL: test_float_cc
; V8:       fcmpd
; V8:       {{fbl|fbuge}} .LBB
; V8:       fcmpd
; V8:       {{fbule|fbg}} .LBB

; V9-LABEL: test_float_cc
; V9:       fcmpd %fcc0
; V9:       {{fbl|fbuge}} %fcc0, .LBB
; V9:       fcmpd %fcc0
; V9:       {{fbule|fbg}} %fcc0, .LBB

   %0 = fcmp uge double %a, 0.000000e+00
   br i1 %0, label %loop, label %loop.2

loop:
   %1 = icmp eq i32 %c, 10
   br i1 %1, label %loop, label %exit.0

loop.2:
   %2 = fcmp ogt double %b, 0.000000e+00
   br i1 %2, label %exit.1, label %loop

exit.0:
   ret i32 0

exit.1:
   ret i32 1
}

; V8-LABEL: test_adde_sube
; V8:       addcc
; V8:       addxcc
; V8:       addxcc
; V8:       addxcc
; V8:       subcc
; V8:       subxcc
; V8:       subxcc
; V8:       subxcc


; V9-LABEL: test_adde_sube
; V9:       addcc
; V9:       addxcc
; V9:       addxcc
; V9:       addxcc
; V9:       subcc
; V9:       subxcc
; V9:       subxcc
; V9:       subxcc

; SPARC64-LABEL: test_adde_sube
; SPARC64:       addcc
; SPARC64:       addxcc
; SPARC64:       addxcc
; SPARC64:       addxcc
; SPARC64:       subcc
; SPARC64:       subxcc
; SPARC64:       subxcc
; SPARC64:       subxcc


define void @test_adde_sube(ptr %a, ptr %b, ptr %sum, ptr %diff) {
entry:
   %0 = bitcast ptr %a to ptr
   %1 = bitcast ptr %b to ptr
   %2 = load i128, ptr %0
   %3 = load i128, ptr %1
   %4 = add i128 %2, %3
   %5 = bitcast ptr %sum to ptr
   store i128 %4, ptr %5
   tail call void asm sideeffect "", "=*m,*m"(ptr elementtype(i128) %0, ptr elementtype(i128) %5) nounwind
   %6 = load i128, ptr %0
   %7 = sub i128 %2, %6
   %8 = bitcast ptr %diff to ptr
   store i128 %7, ptr %8
   ret void
}
