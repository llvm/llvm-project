; RUN: opt < %s -gvn -S | not grep "%z2 ="
; RUN: opt < %s -passes=gvn -S | not grep "%z2 ="

define i32 @main() {
block1:
	%z1 = bitcast i32 0 to i32
	br label %block2
block2:
  %z2 = bitcast i32 0 to i32
  ret i32 %z2
}
