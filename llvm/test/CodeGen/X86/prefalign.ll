; RUN: llc < %s | FileCheck --check-prefixes=CHECK,NOFS %s
; RUN: llc -function-sections < %s | FileCheck --check-prefixes=CHECK,FS %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: .globl f1
; NOFS-NEXT: .p2align 4
; FS-NEXT: .prefalign 16
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
; NOFS-NEXT: .p2align 2
; FS-NEXT: .p2align 1
; FS-NEXT: .prefalign 4
define void @f3() align 2 prefalign(4) {
  ret void
}
