; RUN: opt < %s -gvn -enable-pre -S | grep "b.pre"

define i32 @main(i32 %p) {
block1:
  
	br i1 true, label %block2, label %block3

block2:
 %a = add i32 %p, 1
 br label %block4

block3:
  br label %block4

block4:
  %b = add i32 %p, 1
  ret i32 %b
}
