; RUN: opt -codegenprepare -S < %s | FileCheck %s

target triple = "x86_64-unknown-unknown"

; Nothing to sink and convert here.

define i32 @no_sink(double %a, double* %b, i32 %x, i32 %y)  {
entry:
  %load = load double, double* %b, align 8
  %cmp = fcmp olt double %load, %a
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel

; CHECK-LABEL: @no_sink(
; CHECK:    %load = load double, double* %b, align 8
; CHECK:    %cmp = fcmp olt double %load, %a
; CHECK:    %sel = select i1 %cmp, i32 %x, i32 %y
; CHECK:    ret i32 %sel
}


; An 'fdiv' is expensive, so sink it rather than speculatively execute it.

define float @fdiv_true_sink(float %a, float %b) {
entry:
  %div = fdiv float %a, %b
  %cmp = fcmp ogt float %a, 1.0
  %sel = select i1 %cmp, float %div, float 2.0
  ret float %sel

; CHECK-LABEL: @fdiv_true_sink(
; CHECK:    %cmp = fcmp ogt float %a, 1.0
; CHECK:    br i1 %cmp, label %select.true.sink, label %select.end
; CHECK:  select.true.sink:
; CHECK:    %div = fdiv float %a, %b
; CHECK:    br label %select.end
; CHECK:  select.end:
; CHECK:    %sel = phi float [ %div, %select.true.sink ], [ 2.000000e+00, %entry ]
; CHECK:    ret float %sel
}

define float @fdiv_false_sink(float %a, float %b) {
entry:
  %div = fdiv float %a, %b
  %cmp = fcmp ogt float %a, 3.0
  %sel = select i1 %cmp, float 4.0, float %div
  ret float %sel

; CHECK-LABEL: @fdiv_false_sink(
; CHECK:    %cmp = fcmp ogt float %a, 3.0
; CHECK:    br i1 %cmp, label %select.end, label %select.false.sink
; CHECK:  select.false.sink:
; CHECK:    %div = fdiv float %a, %b
; CHECK:    br label %select.end
; CHECK:  select.end:
; CHECK:    %sel = phi float [ 4.000000e+00, %entry ], [ %div, %select.false.sink ]
; CHECK:    ret float %sel
}

define float @fdiv_both_sink(float %a, float %b) {
entry:
  %div1 = fdiv float %a, %b
  %div2 = fdiv float %b, %a
  %cmp = fcmp ogt float %a, 5.0
  %sel = select i1 %cmp, float %div1, float %div2
  ret float %sel

; CHECK-LABEL: @fdiv_both_sink(
; CHECK:    %cmp = fcmp ogt float %a, 5.0
; CHECK:    br i1 %cmp, label %select.true.sink, label %select.false.sink
; CHECK:  select.true.sink:
; CHECK:    %div1 = fdiv float %a, %b
; CHECK:    br label %select.end
; CHECK:  select.false.sink:
; CHECK:    %div2 = fdiv float %b, %a
; CHECK:    br label %select.end
; CHECK:  select.end:
; CHECK:    %sel = phi float [ %div1, %select.true.sink ], [ %div2, %select.false.sink ]
; CHECK:    ret float %sel
}

; But if the select is marked unpredictable, then don't turn it into a branch.

define float @unpredictable_select(float %a, float %b) {
; CHECK-LABEL: @unpredictable_select(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DIV:%.*]] = fdiv float %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float %a, 1.000000e+00
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[CMP]], float [[DIV]], float 2.000000e+00, !unpredictable !0
; CHECK-NEXT:    ret float [[SEL]]
;
entry:
  %div = fdiv float %a, %b
  %cmp = fcmp ogt float %a, 1.0
  %sel = select i1 %cmp, float %div, float 2.0, !unpredictable !0
  ret float %sel
}

!0 = !{}

; An 'fadd' is not too expensive, so it's ok to speculate.

define float @fadd_no_sink(float %a, float %b) {
  %add = fadd float %a, %b
  %cmp = fcmp ogt float 6.0, %a
  %sel = select i1 %cmp, float %add, float 7.0
  ret float %sel

; CHECK-LABEL: @fadd_no_sink(
; CHECK:  %sel = select i1 %cmp, float %add, float 7.0
}

; Possible enhancement: sinkability is only calculated with the direct
; operand of the select, so we don't try to sink this. The fdiv cost is not
; taken into account.

define float @fdiv_no_sink(float %a, float %b) {
entry:
  %div = fdiv float %a, %b
  %add = fadd float %div, %b
  %cmp = fcmp ogt float %a, 1.0
  %sel = select i1 %cmp, float %add, float 8.0
  ret float %sel

; CHECK-LABEL: @fdiv_no_sink(
; CHECK:  %sel = select i1 %cmp, float %add, float 8.0
}

; Do not transform the CFG if the select operands may have side effects.

declare i64* @bar(i32, i32, i32)
declare i64* @baz(i32, i32, i32)

define i64* @calls_no_sink(i32 %in) {
  %call1 = call i64* @bar(i32 1, i32 2, i32 3)
  %call2 = call i64* @baz(i32 1, i32 2, i32 3)
  %tobool = icmp ne i32 %in, 0
  %sel = select i1 %tobool, i64* %call1, i64* %call2
  ret i64* %sel

; CHECK-LABEL: @calls_no_sink(
; CHECK:  %sel = select i1 %tobool, i64* %call1, i64* %call2
}

define i32 @sdiv_no_sink(i32 %a, i32 %b) {
  %div1 = sdiv i32 %a, %b
  %div2 = sdiv i32 %b, %a
  %cmp = icmp sgt i32 %a, 5
  %sel = select i1 %cmp, i32 %div1, i32 %div2
  ret i32 %sel

; CHECK-LABEL: @sdiv_no_sink(
; CHECK:  %sel = select i1 %cmp, i32 %div1, i32 %div2
}

