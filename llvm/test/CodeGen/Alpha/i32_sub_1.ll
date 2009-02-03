; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-as < %s | llc -march=alpha | grep -i {subl \$16,1,\$0}


define i32 @foo(i32 signext %x) signext {
entry:
	%tmp.1 = add i32 %x, -1		; <int> [#uses=1]
	ret i32 %tmp.1
}
