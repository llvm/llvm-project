; The SelectionDAG list scheduler used to assert, or crash in release builds,
; on a pseudo probe when it ran with a hazard recognizer, as it does for ARM
; cores with an itinerary such as Cortex-A8.

; RUN: llc -mtriple=armv7 -mcpu=cortex-a8 %s -o - | FileCheck %s

; CHECK-LABEL: f:
; CHECK:       add r0, r0, r1
; CHECK-NEXT:  bx lr

define i32 @f(i32 %a, i32 %b) {
entry:
  call void @llvm.pseudoprobe(i64 1, i64 1, i32 0, i64 -1)
  %r = add i32 %a, %b
  ret i32 %r
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)
