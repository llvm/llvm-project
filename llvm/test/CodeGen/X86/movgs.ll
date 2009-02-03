; RUN: llvm-as < %s | llc -march=x86 | grep gs

define i32 @foo() nounwind readonly {
entry:
	%tmp = load i32* addrspace(256)* getelementptr (i32* addrspace(256)* inttoptr (i32 72 to i32* addrspace(256)*), i32 31)		; <i32*> [#uses=1]
	%tmp1 = load i32* %tmp		; <i32> [#uses=1]
	ret i32 %tmp1
}
