; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep {%DEAD = phi i32. }

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
  %DEAD = load i32** %p
  %c = load i32* %DEAD
  ret i32 %c
}
