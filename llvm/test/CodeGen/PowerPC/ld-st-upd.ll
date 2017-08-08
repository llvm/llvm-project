; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc-unknown-linux-gnu"

; Function Attrs: nounwind
define i32* @test4(i32* readonly %X, i32* nocapture %dest) #0 {
  %Y = getelementptr i32, i32* %X, i64 4
  %A = load i32, i32* %Y, align 4
  store i32 %A, i32* %dest, align 4
  ret i32* %Y

; CHECK-LABEL: @test4
; CHECK: lwzu [[REG1:[0-9]+]], 16(3)
; CHECK: stw [[REG1]], 0(4)
; CHECK: blr
}

attributes #0 = { nounwind }

