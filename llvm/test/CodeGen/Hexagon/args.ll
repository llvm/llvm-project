; RUN: llc -march=hexagon -mcpu=hexagonv4 -disable-hexagon-misched < %s | FileCheck %s
; CHECK: r[[T0:[0-9]+]] = #7
; CHECK: memw(r29 + #0) = r[[T0]]
; CHECK: r5 = #6
; CHECK: r0 = #1
; CHECK: r1 = #2
; CHECK: r2 = #3
; CHECK: r3 = #4
; CHECK: r4 = #5


define void @foo() nounwind {
entry:
  call void @bar(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret void
}

declare void @bar(i32, i32, i32, i32, i32, i32, i32)
