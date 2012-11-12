; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s

; There shouldn't be a unconditional branch at end of bb52.
; rdar://7184787

@posed = external global i64                      ; <i64*> [#uses=1]

define i1 @ab_bb52(i64 %.reload78, i64* %.out, i64* %.out1) nounwind {
newFuncRoot:
  br label %bb52

bb52.bb55_crit_edge.exitStub:                     ; preds = %bb52
  store i64 %0, i64* %.out
  store i64 %2, i64* %.out1
  ret i1 true

bb52.bb53_crit_edge.exitStub:                     ; preds = %bb52
  store i64 %0, i64* %.out
  store i64 %2, i64* %.out1
  ret i1 false

bb52:                                             ; preds = %newFuncRoot
; CHECK: movne
; CHECK: moveq
; CHECK: pop
  %0 = load i64* @posed, align 4                  ; <i64> [#uses=3]
  %1 = sub i64 %0, %.reload78                     ; <i64> [#uses=1]
  %2 = ashr i64 %1, 1                             ; <i64> [#uses=3]
  %3 = icmp eq i64 %2, 0                          ; <i1> [#uses=1]
  br i1 %3, label %bb52.bb55_crit_edge.exitStub, label %bb52.bb53_crit_edge.exitStub
}
