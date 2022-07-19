; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s

@d = global i32 0, align 4

; Verify the sar happens before ecx is clobbered with the parameter being
; passed to fn3
; CHECK-LABEL: fn4
; CHECK: movb d, %cl
; CHECK: sarl %cl
; CHECK: movl $2, %ecx
define i32 @fn4(i32 %i) #0 {
entry:
  %0 = load i32, ptr @d, align 4
  %shr = ashr i32 %i, %0
  tail call fastcc void @fn3(i32 2, i32 5, i32 %shr, i32 %i)
  %cmp = icmp slt i32 %shr, 1
  %. = zext i1 %cmp to i32
  ret i32 %.
}

declare void @fn3(i32 %p1, i32 %p2, i32 %p3, i32 %p4) #0

attributes #0 = { nounwind }
