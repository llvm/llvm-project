; RUN: llc < %s -mtriple=armv6-linux-gnu -target-abi=apcs | FileCheck %s

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
; CHECK: push {r4, r5, lr}
; CHECK: pop {r4, r5, pc}
        call void asm sideeffect "", "~{r4},~{r5}"() nounwind
	%0 = load i64*, i64** @b, align 4
	%1 = load i64, i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
