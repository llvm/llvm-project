; RUN: opt < %s -basicaa -gvn -S | grep "DEAD = phi i32 "

define i32 @main(i32* %p) {
block1:
  %z = load i32* %p
	br i1 true, label %block2, label %block3

block2:
 br label %block4

block3:
  %b = bitcast i32 0 to i32
  store i32 %b, i32* %p
  br label %block4

block4:
  %DEAD = load i32* %p
  ret i32 %DEAD
}
