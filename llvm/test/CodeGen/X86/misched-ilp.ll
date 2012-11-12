; RUN: llc < %s -mtriple=x86_64-apple-macosx -mcpu=core2 -enable-misched -misched=ilpmax | FileCheck -check-prefix=MAX %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx -mcpu=core2 -enable-misched -misched=ilpmin | FileCheck -check-prefix=MIN %s
;
; Basic verification of the ScheduleDAGILP metric.
;
; MAX: addss
; MAX: addss
; MAX: addss
; MAX: subss
; MAX: addss
;
; MIN: addss
; MIN: addss
; MIN: subss
; MIN: addss
; MIN: addss
define float @ilpsched(float %a, float %b, float %c, float %d, float %e, float %f) nounwind uwtable readnone ssp {
entry:
  %add = fadd float %a, %b
  %add1 = fadd float %c, %d
  %add2 = fadd float %e, %f
  %add3 = fsub float %add1, %add2
  %add4 = fadd float %add, %add3
  ret float %add4
}
