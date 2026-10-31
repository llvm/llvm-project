; The SelectionDAG list scheduler used to assert, or crash in release builds,
; on a pseudo probe when it ran with a hazard recognizer, as it does for ARM
; cores with an itinerary such as Cortex-A8.

; RUN: llc -mtriple=armv7 -mcpu=cortex-a8 -stop-after=finalize-isel %s -o - | FileCheck %s

; CHECK-LABEL: name: f
; CHECK-DAG: ADDrr
; CHECK-DAG: PSEUDO_PROBE 1, 1, 0, 0
; CHECK: BX_RET

define i32 @f(i32 %a, i32 %b) {
entry:
  call void @llvm.pseudoprobe(i64 1, i64 1, i32 0, i64 -1)
  %r = add i32 %a, %b
  ret i32 %r
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)
