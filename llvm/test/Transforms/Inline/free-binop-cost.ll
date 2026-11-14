; RUN: opt -passes="print<inline-cost>" -disable-output %s 2>&1 | FileCheck %s

; CHECK:      Analyzing call of inner_sel... (caller:outer_sel)
; CHECK:      NumInstructionsSimplified: 3
; CHECK:      Cost: -35
; CHECK:      Analyzing call of inner_and... (caller:outer_and)
; CHECK:      NumInstructionsSimplified: 3
; CHECK:      Cost: -35

define i1 @outer_sel(i1 %x) {
  %r = call i1 @inner_sel(i1 %x)
  ret i1 %r
}

define i1 @outer_and(i1 %x) {
  %r = call i1 @inner_and(i1 %x)
  ret i1 %r
}

define i1 @inner_sel(i1 %x) {
  %wc = call i1 @llvm.experimental.widenable.condition()
  %a = select i1 %x, i1 %wc, i1 false
  ret i1 %a
}

define i1 @inner_and(i1 %x) {
  %wc = call i1 @llvm.experimental.widenable.condition()
  %a = and i1 %x, %wc
  ret i1 %a
}
