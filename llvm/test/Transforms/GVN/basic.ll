; RUN: llvm-as < %s | opt -gvn | llvm-dis | not grep {%z2 =}

define i32 @main() {
block1:
	%z1 = bitcast i32 0 to i32
	br label %block2
block2:
  %z2 = bitcast i32 0 to i32
  ret i32 %z2
}
