; RUN: llc -mtriple=thumb-eabi < %s | FileCheck %s

define i1 @t1(i64 %x) {
; CHECK-LABEL: t1:
; CHECK: lsrs  r0, r1, #31
  %B = icmp slt i64 %x, 0
  ret i1 %B
}

define i1 @t2(i64 %x) {
; CHECK-LABEL: t2:
; CHECK: movs  r0, #0
; CHECK: subs  r0, r0, r1
; CHECK: adcs  r0, r1
  %tmp = icmp ult i64 %x, 4294967296
  ret i1 %tmp
}

define i1 @t3(i32 %x) {
; CHECK-LABEL: t3:
; CHECK: movs  r0, #0
  %tmp = icmp ugt i32 %x, -1
  ret i1 %tmp
}

; CHECK-NOT: cmp
