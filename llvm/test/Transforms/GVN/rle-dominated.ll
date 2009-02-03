; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep load | count 2

define i32 @main(i32** %p) {
block1:
  %z = load i32** %p
	br i1 true, label %block2, label %block3

block2:
 %a = load i32** %p
 br label %block4

block3:
  %b = load i32** %p
  br label %block4

block4:
  %c = load i32** %p
  %d = load i32* %c
  ret i32 %d
}
