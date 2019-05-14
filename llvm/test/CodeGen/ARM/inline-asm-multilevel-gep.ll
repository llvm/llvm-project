; RUN: llc < %s -mtriple armv7-linux-gnueabi | FileCheck %s

; @foo is a 2d array of i32s, ex.
; i32 foo [2][2]
@foo = internal global [2 x [2 x i32]] zeroinitializer, align 4

define void @bar() {
; access foo[1][1]
; CHECK: @ foo+12
  tail call void asm sideeffect "@ ${0:c}", "i"(i32* getelementptr inbounds ([2 x [2 x i32]], [2 x [2 x i32]]* @foo, i64 0, i64 1, i64 1))
  ret void
}
