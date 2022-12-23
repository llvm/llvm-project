; RUN: opt < %s -passes=gvn -S | FileCheck %s

define i32 @main(ptr %p, i32 %x, i32 %y) {
block1:
    %cmp = icmp eq i32 %x, %y
	br i1 %cmp , label %block2, label %block3

block2:
 %a = load ptr, ptr %p
 br label %block4

block3:
  %b = load ptr, ptr %p
  br label %block4

block4:
; CHECK-NOT: %existingPHI = phi
; CHECK: %DEAD = phi
  %existingPHI = phi ptr [ %a, %block2 ], [ %b, %block3 ] 
  %DEAD = load ptr, ptr %p
  %c = load i32, ptr %DEAD
  %d = load i32, ptr %existingPHI
  %e = add i32 %c, %d
  ret i32 %e
}
