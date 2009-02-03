; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm | grep cmpne | count 1
; RUN: llvm-as < %s | llc -march=arm | grep bx | count 2

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) {
	switch i32 %c, label %cond_next [
		 i32 1, label %cond_true
		 i32 7, label %cond_true
	]

cond_true:
	%tmp12 = add i32 %a, 1
	%tmp1518 = add i32 %tmp12, %b
	ret i32 %tmp1518

cond_next:
	%tmp15 = add i32 %b, %a
	ret i32 %tmp15
}
