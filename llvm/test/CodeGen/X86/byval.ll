; RUN: llvm-as < %s | llc -march=x86-64 | grep {movq	8(%rsp), %rax}
; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep {movl	8(%esp), %edx} %t
; RUN: grep {movl	4(%esp), %eax} %t

%struct.s = type { i64, i64, i64 }

define i64 @f(%struct.s* byval %a) {
entry:
	%tmp2 = getelementptr %struct.s* %a, i32 0, i32 0
	%tmp3 = load i64* %tmp2, align 8
	ret i64 %tmp3
}
