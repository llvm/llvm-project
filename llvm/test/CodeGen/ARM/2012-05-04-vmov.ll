; RUN: llc -O1 -march=arm -mcpu=cortex-a9 < %s | FileCheck -check-prefix=A9-CHECK %s
; RUN: llc -O1 -march=arm -mcpu=swift < %s | FileCheck -check-prefix=SWIFT-CHECK %s
; Check that swift doesn't use vmov.32. <rdar://problem/10453003>.

define <2 x i32> @testuvec(<2 x i32> %A, <2 x i32> %B) nounwind {
entry:
  %div = udiv <2 x i32> %A, %B
  ret <2 x i32> %div
; A9-CHECK: vmov.32
; SWIFT-CHECK-NOT: vmov.32
}
