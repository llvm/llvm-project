; RUN: llc < %s -mtriple=armv6-linux-gnu -target-abi=apcs | FileCheck %s

@b = external global ptr

define i64 @t(i64 %a) nounwind readonly {
entry:
; CHECK: push {r4, r5, lr}
; CHECK: pop {r4, r5, pc}
        call void asm sideeffect "", "~{r4},~{r5}"() nounwind
	%0 = load ptr, ptr @b, align 4
	%1 = load i64, ptr %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
