; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s

; Given both Arm and Thumb functions in the same compilation unit, we should
; get both arm and thumb thunks.

define i32 @test1(i32 %a, i32 %b) {
  ret i32 %a
}

define i32 @test2(i32 %a, i32 %b) "target-features"="+thumb-mode" {
  ret i32 %a
}

; CHECK: test1
; CHECK: test2
; CHECK: __llvm_slsblr_thunk_arm_sp
; CHECK: __llvm_slsblr_thunk_thumb_sp