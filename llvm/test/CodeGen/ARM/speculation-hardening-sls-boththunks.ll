; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi -stop-after=arm-sls-hardening %s -o - | FileCheck %s

; Given both Arm and Thumb functions in the same compilation unit, we should
; get both arm and thumb thunks.

define i32 @test1(i32 %a, i32 %b) {
  ret i32 %a
}

define i32 @test2(i32 %a, i32 %b) "target-features"="+thumb-mode" {
  ret i32 %a
}

; CHECK: define i32 @test1(i32 %a, i32 %b) #0
; CHECK: define i32 @test2(i32 %a, i32 %b) #1
; CHECK: define linkonce_odr hidden void @__llvm_slsblr_thunk_arm_sp() #2 comdat
; CHECK: define linkonce_odr hidden void @__llvm_slsblr_thunk_thumb_sp() #3 comdat

; CHECK: attributes #0 = { "target-features"="+harden-sls-retbr,+harden-sls-blr" }
; CHECK: attributes #1 = { "target-features"="+thumb-mode,+harden-sls-retbr,+harden-sls-blr" }
; CHECK: attributes #2 = { naked nounwind }
; CHECK: attributes #3 = { naked nounwind "target-features"="+thumb-mode" }

