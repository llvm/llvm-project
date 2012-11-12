; RUN: llc -march=x86-64 < %s | FileCheck %s -check-prefix=CHECK-64

define void @a(i64* nocapture %s, i64* nocapture %t, i64 %a, i64 %b, i64 %c) nounwind {
entry:
 %0 = zext i64 %a to i128
 %1 = zext i64 %b to i128
 %2 = add i128 %1, %0
 %3 = zext i64 %c to i128
 %4 = shl i128 %3, 64
 %5 = add i128 %4, %2
 %6 = lshr i128 %5, 64
 %7 = trunc i128 %6 to i64
 store i64 %7, i64* %s, align 8
 %8 = trunc i128 %2 to i64
 store i64 %8, i64* %t, align 8
 ret void

; CHECK-64: addq
; CHECK-64: adcq $0
}
