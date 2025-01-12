; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: m2and_rr
define i1 @m2and_rr(i1 %a, i1 %b) {
; CHECK: and.pred %p{{[0-9]+}}, %p{{[0-9]+}}, %p{{[0-9]+}}
; CHECK-NOT: mul
  %r = mul i1 %a, %b
  ret i1 %r
}

; CHECK-LABEL: m2and_ri
define i1 @m2and_ri(i1 %a) {
; CHECK-NOT: mul
  %r = mul i1 %a, 1
  ret i1 %r
}

; CHECK-LABEL: select2or
define i1 @select2or(i1 %a, i1 %b) {
; CHECK: or.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK-NOT: selp
  %r = select i1 %a, i1 1, i1 %b
  ret i1 %r
}

; CHECK-LABEL: select2and
define i1 @select2and(i1 %a, i1 %b) {
; CHECK: and.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK-NOT: selp
  %r = select i1 %a, i1 %b, i1 0
  ret i1 %r
}
