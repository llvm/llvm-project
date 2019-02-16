; RUN: llc < %s -mtriple=s390x-linux-gnu -debug-only=systemz-isel -o - 2>&1 | \
; RUN:   FileCheck %s

; REQUIRES: asserts
;
; Check that some debug output is printed without problems.
; CHECK: SystemZAddressingMode
; CHECK: Base t5: i64,ch = load<(load 8 from %ir.0)>
; CHECK: Index
; CHECK: Disp

define void @fun(i64* %ptr) {
entry:
  %0 = bitcast i64* %ptr to i32**
  %1 = load i32*, i32** %0, align 8
  %xpv_pv = getelementptr inbounds i32, i32* %1
  store i32 0, i32* %xpv_pv
  ret void
}
