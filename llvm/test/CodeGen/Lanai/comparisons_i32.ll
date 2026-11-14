; RUN: llc < %s | FileCheck %s

; Test that basic 32-bit integer comparison operations assemble as expected.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; CHECK-LABEL: eq_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: seq
define i32 @eq_i32(i32 %x, i32 %y) {
  %a = icmp eq i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sne
define i32 @ne_i32(i32 %x, i32 %y) {
  %a = icmp ne i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: slt
define i32 @slt_i32(i32 %x, i32 %y) {
  %a = icmp slt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sle
define i32 @sle_i32(i32 %x, i32 %y) {
  %a = icmp sle i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sult
define i32 @ult_i32(i32 %x, i32 %y) {
  %a = icmp ult i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sule
define i32 @ule_i32(i32 %x, i32 %y) {
  %a = icmp ule i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sgt
define i32 @sgt_i32(i32 %x, i32 %y) {
  %a = icmp sgt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sge
define i32 @sge_i32(i32 %x, i32 %y) {
  %a = icmp sge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: sugt
define i32 @ugt_i32(i32 %x, i32 %y) {
  %a = icmp ugt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i32:
; CHECK: sub.f %r{{[0-9]+}}, %r{{[0-9]+}}, %r0
; CHECK-NEXT: suge
define i32 @uge_i32(i32 %x, i32 %y) {
  %a = icmp uge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
