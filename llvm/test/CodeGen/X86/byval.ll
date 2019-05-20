; RUN: llc < %s -mtriple=x86_64-linux | FileCheck -check-prefix=X86-64 %s
; Win64 has not supported byval yet.
; RUN: llc < %s -mtriple=i686-- | FileCheck -check-prefix=X86 %s

; X86: movl	4(%esp), %eax
; X86: movl	8(%esp), %edx

; X86-64: movq	8(%rsp), %rax

%struct.s = type { i64, i64, i64 }

define i64 @f(%struct.s* byval %a) {
entry:
	%tmp2 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 0
	%tmp3 = load i64, i64* %tmp2, align 8
	ret i64 %tmp3
}
