; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

; CHECK: D a1
; CHECK-NEXT: d a2
; CHECK-NEXT: T f1
; CHECK-NEXT: t f2
; CHECK-NEXT: W f3
; CHECK-NEXT: U f4
; CHECK-NEXT: w f5
; CHECK-NEXT: D g1
; CHECK-NEXT: d g2
; CHECK-NEXT: C g3
; CHECK-NOT: g4
; CHECK-NEXT: T global_asm_sym
; CHECK-NEXT: t local_asm_sym
; CHECK-NEXT: U undef_asm_sy

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global global_asm_sym"
module asm "global_asm_sym:"
module asm "local_asm_sym:"
module asm ".long undef_asm_sym"

@g1 = global i32 42
@g2 = internal global i32 42
@g3 = common global i32 0
@g4 = private global i32 42

@a1 = alias i32, i32* @g1
@a2 = internal alias i32, i32* @g1

define void @f1() {
  call void @f5()
  ret void
}

define internal void @f2() {
  ret void
}

define linkonce_odr void @f3() {
  ret void
}

declare void @f4()

declare extern_weak void @f5()
