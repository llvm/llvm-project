; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm | grep bx | count 1

define i32 @t1(i32 %a, i32 %b) {
	%tmp2 = icmp eq i32 %a, 0
	br i1 %tmp2, label %cond_false, label %cond_true

cond_true:
	%tmp5 = add i32 %b, 1
	ret i32 %tmp5

cond_false:
	%tmp7 = add i32 %b, -1
	ret i32 %tmp7
}
