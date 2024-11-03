; RUN: llc -mtriple=spirv32-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; CHECK:      OpCapability ExpectAssumeKHR
; CHECK-NEXT: OpExtension "SPV_KHR_expect_assume"

declare i32 @llvm.expect.i32(i32, i32)
declare i32 @getOne()

; CHECK-DAG: %2 = OpTypeInt 32 0
; CHECK-DAG: %6 = OpFunctionParameter %2
; CHECK-DAG: %9 = OpIMul %2 %6 %8
; CHECK-DAG: %10 = OpExpectKHR %2 %9 %6

define i32 @test(i32 %x) {
  %one = call i32 @getOne()
  %val = mul i32 %x, %one
  %v = call i32 @llvm.expect.i32(i32 %val, i32 %x)
  ret i32 %v
}
