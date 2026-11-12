; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z10 -mattr=soft-float -O0 < %s | FileCheck %s

; Arithmetic functions

define float @test_addsf3(float %a, float %b) {
  ; CHECK-LABEL:  test_addsf3:
  ; CHECK:        brasl %r14, __addsf3
  %add = fadd float %a, %b
  ret float %add
}

define double @test_adddf3(double %a, double %b) {
  ; CHECK-LABEL:  test_adddf3:
  ; CHECK:        brasl %r14, __adddf3
  %add = fadd double %a, %b
  ret double %add
}

define fp128 @test_addtf3(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_addtf3:
  ; CHECK:        brasl %r14, __addtf3
  %add = fadd fp128 %a, %b
  ret fp128 %add
}

define float @test_mulsf3(float %a, float %b) {
  ; CHECK-LABEL:  test_mulsf3:
  ; CHECK:        brasl %r14, __mulsf3
  %mul = fmul float %a, %b
  ret float %mul
}

define double @test_muldf3(double %a, double %b) {
  ; CHECK-LABEL:  test_muldf3:
  ; CHECK:        brasl %r14, __muldf3
  %mul = fmul double %a, %b
  ret double %mul
}

define fp128 @test_multf3(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_multf3:
  ; CHECK:        brasl %r14, __multf3
  %mul = fmul fp128 %a, %b
  ret fp128 %mul
}

define float @test_subsf3(float %a, float %b) {
  ; CHECK-LABEL:  test_subsf3:
  ; CHECK:        brasl %r14, __subsf3
  %sub = fsub float %a, %b
  ret float %sub
}

define double @test_subdf3(double %a, double %b) {
  ; CHECK-LABEL:  test_subdf3:
  ; CHECK:        brasl %r14, __subdf3
  %sub = fsub double %a, %b
  ret double %sub
}

define fp128 @test_subtf3(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_subtf3:
  ; CHECK:        brasl %r14, __subtf3
  %sub = fsub fp128 %a, %b
  ret fp128 %sub
}

define float @test_divsf3(float %a, float %b) {
  ; CHECK-LABEL:  test_divsf3:
  ; CHECK:        brasl %r14, __divsf3
  %div = fdiv float %a, %b
  ret float %div
}

define double @test_divdf3(double %a, double %b) {
  ; CHECK-LABEL:  test_divdf3:
  ; CHECK:        brasl %r14, __divdf3
  %div = fdiv double %a, %b
  ret double %div
}

define fp128 @test_divtf3(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_divtf3:
  ; CHECK:        brasl %r14, __divtf3
  %div = fdiv fp128 %a, %b
  ret fp128 %div
}

; Comparison functions
define i1 @test_unordsf2(float %a, float %b) {
  ; CHECK-LABEL:  test_unordsf2:
  ; CHECK:        brasl %r14, __unordsf2
  %cmp = fcmp uno float %a, %b
  ret i1 %cmp
}

define i1 @test_unorddf2(double %a, double %b) {
  ; CHECK-LABEL:  test_unorddf2:
  ; CHECK:        brasl %r14, __unorddf2
  %cmp = fcmp uno double %a, %b
  ret i1 %cmp
}

define i1 @test_unordtf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_unordtf2:
  ; CHECK:        brasl %r14, __unordtf2
  %cmp = fcmp uno fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_eqsf2(float %a, float %b) {
  ; CHECK-LABEL:  test_eqsf2:
  ; CHECK:        brasl %r14, __eqsf2
  %cmp = fcmp oeq float %a, %b
  ret i1 %cmp
}

define i1 @test_eqdf2(double %a, double %b) {
  ; CHECK-LABEL:  test_eqdf2:
  ; CHECK:        brasl %r14, __eqdf2
  %cmp = fcmp oeq double %a, %b
  ret i1 %cmp
}

define i1 @test_eqtf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_eqtf2:
  ; CHECK:        brasl %r14, __eqtf2
  %cmp = fcmp oeq fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_nesf2(float %a, float %b) {
  ; CHECK-LABEL:  test_nesf2:
  ; CHECK:        brasl %r14, __nesf2
  %cmp = fcmp une float %a, %b
  ret i1 %cmp
}

define i1 @test_nedf2(double %a, double %b) {
  ; CHECK-LABEL:  test_nedf2:
  ; CHECK:        brasl %r14, __nedf2
  %cmp = fcmp une double %a, %b
  ret i1 %cmp
}

define i1 @test_netf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_netf2:
  ; CHECK:        brasl %r14, __netf2
  %cmp = fcmp une fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_gesf2(float %a, float %b) {
  ; CHECK-LABEL:  test_gesf2:
  ; CHECK:        brasl %r14, __gesf2
  %cmp = fcmp oge float %a, %b
  ret i1 %cmp
}

define i1 @test_gedf2(double %a, double %b) {
  ; CHECK-LABEL:  test_gedf2:
  ; CHECK:        brasl %r14, __gedf2
  %cmp = fcmp oge double %a, %b
  ret i1 %cmp
}

define i1 @test_getf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_getf2:
  ; CHECK:        brasl %r14, __getf2
  %cmp = fcmp oge fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_ltsf2(float %a, float %b) {
  ; CHECK-LABEL:  test_ltsf2:
  ; CHECK:        brasl %r14, __ltsf2
  %cmp = fcmp olt float %a, %b
  ret i1 %cmp
}

define i1 @test_ltdf2(double %a, double %b) {
  ; CHECK-LABEL:  test_ltdf2:
  ; CHECK:        brasl %r14, __ltdf2
  %cmp = fcmp olt double %a, %b
  ret i1 %cmp
}

define i1 @test_lttf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_lttf2:
  ; CHECK:        brasl %r14, __lttf2
  %cmp = fcmp olt fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_lesf2(float %a, float %b) {
  ; CHECK-LABEL:  test_lesf2:
  ; CHECK:        brasl %r14, __lesf2
  %cmp = fcmp ole float %a, %b
  ret i1 %cmp
}

define i1 @test_ledf2(double %a, double %b) {
  ; CHECK-LABEL:  test_ledf2:
  ; CHECK:        brasl %r14, __ledf2
  %cmp = fcmp ole double %a, %b
  ret i1 %cmp
}

define i1 @test_letf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_letf2:
  ; CHECK:        brasl %r14, __letf2
  %cmp = fcmp ole fp128 %a, %b
  ret i1 %cmp
}

define i1 @test_gtsf2(float %a, float %b) {
  ; CHECK-LABEL:  test_gtsf2:
  ; CHECK:        brasl %r14, __gtsf2
  %cmp = fcmp ogt float %a, %b
  ret i1 %cmp
}

define i1 @test_gtdf2(double %a, double %b) {
  ; CHECK-LABEL:  test_gtdf2:
  ; CHECK:        brasl %r14, __gtdf2
  %cmp = fcmp ogt double %a, %b
  ret i1 %cmp
}

define i1 @test_gttf2(fp128 %a, fp128 %b) {
  ; CHECK-LABEL:  test_gttf2:
  ; CHECK:        brasl %r14, __gttf2
  %cmp = fcmp ogt fp128 %a, %b
  ret i1 %cmp
}
