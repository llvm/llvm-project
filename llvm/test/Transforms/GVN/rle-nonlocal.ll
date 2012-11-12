; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

define i32 @main(i32** %p) {
block1:
	br i1 true, label %block2, label %block3

block2:
 %a = load i32** %p
 br label %block4

block3:
  %b = load i32** %p
  br label %block4

block4:
; CHECK-NOT: %existingPHI = phi
; CHECK: %DEAD = phi
  %existingPHI = phi i32* [ %a, %block2 ], [ %b, %block3 ] 
  %DEAD = load i32** %p
  %c = load i32* %DEAD
  %d = load i32* %existingPHI
  %e = add i32 %c, %d
  ret i32 %e
}
