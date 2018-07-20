; RUN: llc %s -o /dev/null -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -debug-only=arm-ldst-opt 2> %t
; RUN: FileCheck %s < %t
; REQUIRES: asserts
; PR8113: ARMLoadStoreOptimizer must preserve memoperands.

@b = external global i64*

; CHECK: Formed {{.*}} t2LDRD{{.*}} (load 4 from %ir.0), (load 4 from %ir.0 + 4)
define i64 @t(i64 %a) nounwind readonly {
entry:
	%0 = load i64*, i64** @b, align 4
	%1 = load i64, i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
