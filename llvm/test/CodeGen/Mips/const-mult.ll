; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=CHECK
; RUN: llc -march=mips64el < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK64

; CHECK-LABEL: mul5_32:
; CHECK: sll $[[R0:[0-9]+]], $4, 2
; CHECK: addu ${{[0-9]+}}, $[[R0]], $4

define i32 @mul5_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, 5
  ret i32 %mul
}

; CHECK-LABEL:     mul27_32:
; CHECK-DAG: sll $[[R0:[0-9]+]], $4, 2
; CHECK-DAG: addu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK-DAG: sll $[[R2:[0-9]+]], $4, 5
; CHECK:     subu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i32 @mul27_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, 27
  ret i32 %mul
}

; CHECK-LABEL:     muln2147483643_32:
; CHECK-DAG: sll $[[R0:[0-9]+]], $4, 2
; CHECK-DAG: addu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK-DAG: sll $[[R2:[0-9]+]], $4, 31
; CHECK:     addu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i32 @muln2147483643_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, -2147483643
  ret i32 %mul
}

; CHECK64-LABEL:     muln9223372036854775805_64:
; CHECK64-DAG: dsll $[[R0:[0-9]+]], $4, 1
; CHECK64-DAG: daddu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK64-DAG: dsll $[[R2:[0-9]+]], $4, 63
; CHECK64:     daddu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i64 @muln9223372036854775805_64(i64 signext %a) {
entry:
  %mul = mul nsw i64 %a, -9223372036854775805
  ret i64 %mul
}
