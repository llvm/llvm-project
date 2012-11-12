; RUN: llc < %s -mtriple=arm-linux-gnueabi -O0 | FileCheck %s
; run with -O0 to avoid arm global merging.

@c = internal global i8 0, align 1
@x = internal global i32 0, align 4

; CHECK: .lcomm c,1
; .lcomm doesn't support alignment.
; CHECK: .local x
; CHECK-NEXT: .comm x,4,4

define i32 @foo() nounwind {
  ret i32 sub (i32 ptrtoint (i8* @c to i32), i32 ptrtoint (i32* @x to i32))
}
