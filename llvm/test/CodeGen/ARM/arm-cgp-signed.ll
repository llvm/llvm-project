; RUN: llc -mtriple=thumbv7m -arm-disable-cgp=false %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv8m.main -arm-disable-cgp=false %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7 %s -arm-disable-cgp=false -o - | FileCheck %s
; RUN: llc -mtriple=armv8 %s -arm-disable-cgp=false -o - | FileCheck %s

; Test to check that ARMCodeGenPrepare doesn't optimised away sign extends.
; CHECK-LABEL: test_signed_load:
; CHECK: uxth
define i16 @test_signed_load(i16* %ptr) {
  %load = load i16, i16* %ptr
  %conv0 = zext i16 %load to i32
  %conv1 = sext i16 %load to i32
  %cmp = icmp eq i32 %conv0, %conv1
  %conv2 = zext i1 %cmp to i16
  ret i16 %conv2
}

; Don't allow sign bit generating opcodes.
; CHECK-LABEL: test_ashr:
; CHECK: sxth
define i16 @test_ashr(i16 zeroext %arg) {
  %ashr = ashr i16 %arg, 1
  %cmp = icmp eq i16 %ashr, 0
  %conv = zext i1 %cmp to i16
  ret i16 %conv 
}

; CHECK-LABEL: test_sdiv:
; CHECK: sxth
define i16 @test_sdiv(i16 zeroext %arg) {
  %sdiv = sdiv i16 %arg, 2
  %cmp = icmp ne i16 %sdiv, 0
  %conv = zext i1 %cmp to i16
  ret i16 %conv 
}

; CHECK-LABEL: test_srem
; CHECK: sxth
define i16 @test_srem(i16 zeroext %arg) {
  %srem = srem i16 %arg, 4
  %cmp = icmp ne i16 %srem, 0
  %conv = zext i1 %cmp to i16
  ret i16 %conv 
}

