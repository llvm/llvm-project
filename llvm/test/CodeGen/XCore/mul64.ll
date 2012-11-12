; RUN: llc < %s -march=xcore | FileCheck %s
; RUN: llc < %s -march=xcore -regalloc=basic | FileCheck %s
define i64 @umul_lohi(i32 %a, i32 %b) {
entry:
	%0 = zext i32 %a to i64
	%1 = zext i32 %b to i64
	%2 = mul i64 %1, %0
	ret i64 %2
}
; CHECK: umul_lohi:
; CHECK: ldc [[REG:r[0-9]+]], 0
; CHECK-NEXT: lmul {{.*}}, [[REG]], [[REG]]
; CHECK-NEXT: retsp 0

define i64 @smul_lohi(i32 %a, i32 %b) {
entry:
	%0 = sext i32 %a to i64
	%1 = sext i32 %b to i64
	%2 = mul i64 %1, %0
	ret i64 %2
}
; CHECK: smul_lohi:
; CHECK: ldc
; CHECK-NEXT: mov
; CHECK-NEXT: maccs
; CHECK: retsp 0

define i64 @mul64(i64 %a, i64 %b) {
entry:
	%0 = mul i64 %a, %b
	ret i64 %0
}
; CHECK: mul64:
; CHECK: ldc
; CHECK-NEXT: lmul
; CHECK-NEXT: mul
; CHECK-NEXT: lmul

define i64 @mul64_2(i64 %a, i32 %b) {
entry:
	%0 = zext i32 %b to i64
	%1 = mul i64 %a, %0
	ret i64 %1
}
; CHECK: mul64_2:
; CHECK: ldc
; CHECK-NEXT: lmul
; CHECK-NEXT: mul
; CHECK-NEXT: add r1,
; CHECK: retsp 0
