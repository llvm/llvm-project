; RUN: llc < %s -mtriple=s390x-linux-gnu -debug-only=systemz-isel -o - 2>&1 | \
; RUN:   FileCheck %s

; REQUIRES: asserts
;
; Check that some debug output is printed without problems.
; CHECK: SystemZAddressingMode
; CHECK: Base t5: i64,ch = load<(load (s64) from %ir.0)>
; CHECK: Index
; CHECK: Disp

define void @fun(ptr %ptr) {
entry:
  %0 = bitcast ptr %ptr to ptr
  %1 = load ptr, ptr %0, align 8
  %xpv_pv = getelementptr inbounds i32, ptr %1
  store i32 0, ptr %xpv_pv
  ret void
}
