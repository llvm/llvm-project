; RUN: llc < %s | FileCheck %s
; RUN: llc -function-sections < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: .globl f1
; CHECK-NEXT: .prefalign 4
define void @f1() {
  ret void
}

; CHECK: .globl f2
; CHECK-NOT: .prefalign
; CHECK-NOT: .p2align
define void @f2() prefalign(1) {
  ret void
}

; CHECK: .globl f3
; CHECK-NEXT: .p2align 1
; CHECK-NEXT: .prefalign 2
define void @f3() align 2 prefalign(4) {
  ret void
}

;; When align >= prefalign, the .prefalign would be redundant.
; CHECK: .globl f4
; CHECK-NEXT: .p2align 5
; CHECK-NOT: .prefalign
define void @f4() align 32 prefalign(16) {
  ret void
}

; CHECK: .globl f5
; CHECK-NEXT: .p2align 4
; CHECK-NOT: .prefalign
define void @f5() align 16 prefalign(16) {
  ret void
}
