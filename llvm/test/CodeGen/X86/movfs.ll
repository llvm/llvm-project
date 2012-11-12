; RUN: llc < %s -march=x86 | grep fs

define i32 @foo() nounwind readonly {
entry:
	%tmp = load i32* addrspace(257)* getelementptr (i32* addrspace(257)* inttoptr (i32 72 to i32* addrspace(257)*), i32 31)		; <i32*> [#uses=1]
	%tmp1 = load i32* %tmp		; <i32> [#uses=1]
	ret i32 %tmp1
}
