target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s --check-prefixes=CHECK,ICBT
; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s --check-prefixes=CHECK,NOICBT

define void @test1(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 0, i32 3, i32 1)
  ret void

; CHECK-LABEL: @test1
; CHECK: dcbt
}

declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @test2(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 1, i32 3, i32 1)
  ret void

; CHECK-LABEL: @test2
; CHECK: dcbtst
}

define void @test3(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 0, i32 3, i32 0)
  ret void

; CHECK-LABEL: @test3
; ICBT: icbt
; NOICBT-NOT: icbt
; NOICBT-NOT: dcbt
; NOICBT: blr
}

; Instruction prefetch with a write hint used to crash instruction selection;
; icbt has no read/write variants, so it is used for both hints.
define void @test4(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 1, i32 0, i32 0)
  ret void

; CHECK-LABEL: @test4
; ICBT: icbt
; NOICBT-NOT: icbt
; NOICBT-NOT: dcbt
; NOICBT: blr
}
