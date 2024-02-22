; Autoupgrade i64 overloaded @llvm.ptrauth intrinsics to non-overloaded ones.

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare i64 @llvm.ptrauth.auth.i64(i64, i32, i64)
declare i64 @llvm.ptrauth.sign.i64(i64, i32, i64)
declare i64 @llvm.ptrauth.sign.generic.i64(i64, i64)
declare i64 @llvm.ptrauth.resign.i64(i64, i32, i64, i32, i64)
declare i64 @llvm.ptrauth.strip.i64(i64, i32)
declare i64 @llvm.ptrauth.blend.i64(i64, i64)

; CHECK-LABEL: @test_auth(
define i64 @test_auth(i64 %a0, i64 %a1) {
  %tmp0 = call i64 @llvm.ptrauth.auth.i64(i64 %a0, i32 0, i64 %a1)
  ; CHECK: %tmp0 = call i64 @llvm.ptrauth.auth(i64 %a0, i32 0, i64 %a1)
  ret i64 %tmp0
}

; CHECK-LABEL: @test_sign(
define i64 @test_sign(i64 %a0, i64 %a1) {
  %tmp0 = call i64 @llvm.ptrauth.sign.i64(i64 %a0, i32 0, i64 %a1)
  ; CHECK: %tmp0 = call i64 @llvm.ptrauth.sign(i64 %a0, i32 0, i64 %a1)
  ret i64 %tmp0
}

; CHECK-LABEL: @test_strip(
define i64 @test_strip(i64 %a0) {
  %tmp0 = call i64 @llvm.ptrauth.strip.i64(i64 %a0, i32 0)
  ; CHECK: %tmp0 = call i64 @llvm.ptrauth.strip(i64 %a0, i32 0)
  ret i64 %tmp0
}

; CHECK-LABEL: @test_blend(
define i64 @test_blend(i64 %a0, i64 %a1) {
  %tmp0 = call i64 @llvm.ptrauth.blend.i64(i64 %a0, i64 %a1)
  ; CHECK: %tmp0 = call i64 @llvm.ptrauth.blend(i64 %a0, i64 %a1)
  ret i64 %tmp0
}

; CHECK-LABEL: @test_resign(
define i64 @test_resign(i64 %a0, i64 %a1, i64 %a2) {
  %tmp0 = call i64 @llvm.ptrauth.resign.i64(i64 %a0, i32 0, i64 %a1, i32 0, i64 %a2)
  ; CHECK: %tmp0 = call i64 @llvm.ptrauth.resign(i64 %a0, i32 0, i64 %a1, i32 0, i64 %a2)
  ret i64 %tmp0
}

; CHECK: declare i64 @llvm.ptrauth.auth(i64, i32 immarg, i64)
; CHECK: declare i64 @llvm.ptrauth.sign(i64, i32 immarg, i64)
; CHECK: declare i64 @llvm.ptrauth.sign.generic(i64, i64)
; CHECK: declare i64 @llvm.ptrauth.resign(i64, i32 immarg, i64, i32 immarg, i64)
; CHECK: declare i64 @llvm.ptrauth.strip(i64, i32 immarg)
; CHECK: declare i64 @llvm.ptrauth.blend(i64, i64)
