; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r5:4 = combine(#6,#5)
; CHECK: r3:2 = combine(#4,#3)
; CHECK: r1:0 = combine(#2,#1)
; CHECK: memw(r29+#0) = #7


define void @foo() nounwind {
entry:
  call void @bar(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret void
}

declare void @bar(i32, i32, i32, i32, i32, i32, i32)
