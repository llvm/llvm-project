; RUN: llc -O1 -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - \
; RUN:  | FileCheck -check-prefix=A9-CHECK %s

; RUN: llc -O1 -mtriple=arm-eabi -mcpu=swift %s -o - \
; RUN:  | FileCheck -check-prefix=SWIFT-CHECK %s

; Check that swift doesn't use vmov.32. <rdar://problem/10453003>.

define <2 x i32> @testuvec(<2 x i32> %A, <2 x i32> %B) nounwind {
entry:
  %div = udiv <2 x i32> %A, %B
  ret <2 x i32> %div
; A9-CHECK: vmov.32
; vmov.32 should not be used to get a lane:
; vmov.32 <dst>, <src>[<lane>].
; but vmov.32 <dst>[<lane>], <src> is fine.
; SWIFT-CHECK-NOT: vmov.32 {{r[0-9]+}}, {{d[0-9]\[[0-9]+\]}}
}
