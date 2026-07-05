; Test the ICBT instruction on POWER8
; Copied from the ppc64-prefetch.ll test
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @test(ptr %a, ...) nounwind {
entry:
  call void @llvm.prefetch(ptr %a, i32 0, i32 3, i32 0)
  ret void

; CHECK-LABEL: @test
; CHECK: icbt
}


