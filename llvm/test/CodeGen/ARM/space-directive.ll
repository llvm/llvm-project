; RUN: llc -mtriple=armv7 -o - %s | FileCheck %s

define i32 @test_space() minsize {
; CHECK-LABEL: test_space:
; CHECK: ldr {{r[0-9]+}}, [[CPENTRY:.?LCPI[0-9]+_[0-9]+]]
; CHECK: b [[PAST_CP:.?LBB[0-9]+_[0-9]+]]

; CHECK: [[CPENTRY]]:
; CHECK-NEXT: 12345678

; CHECK: [[PAST_CP]]:
; CHECK: .zero 10000
  %addr = inttoptr i32 12345678 to ptr
  %val = load i32, ptr %addr
  call i32 @llvm.arm.space(i32 10000, i32 undef)
  ret i32 %val
}

declare i32 @llvm.arm.space(i32, i32)
