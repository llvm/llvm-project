; RUN: opt < %s -basic-aa -gvn -S | FileCheck %s

define i32 @main(i32** %p, i32 %x, i32 %y) {
block1:
    %cmp = icmp eq i32 %x, %y
	br i1 %cmp , label %block2, label %block3

block2:
 %a = load i32*, i32** %p
 br label %block4

block3:
  %b = load i32*, i32** %p
  br label %block4

block4:
; CHECK-NOT: %existingPHI = phi
; CHECK: %DEAD = phi
  %existingPHI = phi i32* [ %a, %block2 ], [ %b, %block3 ] 
  %DEAD = load i32*, i32** %p
  %c = load i32, i32* %DEAD
  %d = load i32, i32* %existingPHI
  %e = add i32 %c, %d
  ret i32 %e
}
