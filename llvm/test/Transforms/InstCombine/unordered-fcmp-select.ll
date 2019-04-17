; RUN: opt -S -instcombine < %s | FileCheck %s

; CHECK-LABEL: @select_max_ugt(
; CHECK: %cmp.inv = fcmp ole float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_max_ugt(float %a, float %b) {
  %cmp = fcmp ugt float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_max_uge(
; CHECK: %cmp.inv = fcmp olt float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_max_uge(float %a, float %b) {
  %cmp = fcmp uge float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_min_ugt(
; CHECK: %cmp.inv = fcmp ole float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %a, float %b
; CHECK-NEXT: ret float %sel
define float @select_min_ugt(float %a, float %b) {
  %cmp = fcmp ugt float %a, %b
  %sel = select i1 %cmp, float %b, float %a
  ret float %sel
}

; CHECK-LABEL: @select_min_uge(
; CHECK: %cmp.inv = fcmp olt float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %a, float %b
; CHECK-NEXT: ret float %sel
define float @select_min_uge(float %a, float %b) {
  %cmp = fcmp uge float %a, %b
  %sel = select i1 %cmp, float %b, float %a
  ret float %sel
}

; CHECK-LABEL: @select_max_ult(
; CHECK: %cmp.inv = fcmp oge float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %a, float %b
; CHECK-NEXT: ret float %sel
define float @select_max_ult(float %a, float %b) {
  %cmp = fcmp ult float %a, %b
  %sel = select i1 %cmp, float %b, float %a
  ret float %sel
}

; CHECK-LABEL: @select_max_ule(
; CHECK: %cmp.inv = fcmp ogt float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %a, float %b
; CHECK: ret float %sel
define float @select_max_ule(float %a, float %b) {
  %cmp = fcmp ule float %a, %b
  %sel = select i1 %cmp, float %b, float %a
  ret float %sel
}

; CHECK-LABEL: @select_min_ult(
; CHECK: %cmp.inv = fcmp oge float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_min_ult(float %a, float %b) {
  %cmp = fcmp ult float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_min_ule(
; CHECK: %cmp.inv = fcmp ogt float %a, %b
; CHECK-NEXT: %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_min_ule(float %a, float %b) {
  %cmp = fcmp ule float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_fcmp_une(
; CHECK:  %cmp.inv = fcmp oeq float %a, %b
; CHECK-NEXT:  %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_fcmp_une(float %a, float %b) {
  %cmp = fcmp une float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_fcmp_ueq
; CHECK:  %cmp.inv = fcmp one float %a, %b
; CHECK-NEXT:  %sel = select i1 %cmp.inv, float %b, float %a
; CHECK-NEXT: ret float %sel
define float @select_fcmp_ueq(float %a, float %b) {
  %cmp = fcmp ueq float %a, %b
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

declare void @foo(i1)

; CHECK-LABEL: @select_max_ugt_2_use_cmp(
; CHECK: fcmp ugt
; CHECK-NOT: fcmp
; CHECK: ret
define float @select_max_ugt_2_use_cmp(float %a, float %b) {
  %cmp = fcmp ugt float %a, %b
  call void @foo(i1 %cmp)
  %sel = select i1 %cmp, float %a, float %b
  ret float %sel
}

; CHECK-LABEL: @select_min_uge_2_use_cmp(
; CHECK: fcmp uge
; CHECK-NOT: fcmp
; CHECK: ret
define float @select_min_uge_2_use_cmp(float %a, float %b) {
  %cmp = fcmp uge float %a, %b
  call void @foo(i1 %cmp)
  %sel = select i1 %cmp, float %b, float %a
  ret float %sel
}
