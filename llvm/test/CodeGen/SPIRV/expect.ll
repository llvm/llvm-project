; RUN: llc -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=CHECK,EXT %s
; RUN: llc -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=CHECK,EXT %s
; RUN: llc -verify-machineinstrs -mtriple=spirv32-unknown-unknown < %s | FileCheck --check-prefixes=CHECK,NOEXT %s
; RUN: llc -verify-machineinstrs -mtriple=spirv64-unknown-unknown < %s | FileCheck --check-prefixes=CHECK,NOEXT %s

; EXT:      OpCapability ExpectAssumeKHR
; EXT-NEXT: OpExtension "SPV_KHR_expect_assume"
; NOEXT-NOT:  OpCapability ExpectAssumeKHR
; NOEXT-NOT:  OpExtension "SPV_KHR_expect_assume"

declare i32 @llvm.expect.i32(i32, i32)
declare i32 @getOne()

; CHECK-DAG: %2 = OpTypeInt 32 0
; CHECK-DAG: %6 = OpFunctionParameter %2
; CHECK-DAG: %9 = OpIMul %2 %6 %8
; EXT-DAG:   %10 = OpExpectKHR %2 %9 %6
; NOEXT-NOT: %10 = OpExpectKHR %2 %9 %6

define i32 @test(i32 %x) {
  %one = call i32 @getOne()
  %val = mul i32 %x, %one
  %v = call i32 @llvm.expect.i32(i32 %val, i32 %x)
  ret i32 %v
}
