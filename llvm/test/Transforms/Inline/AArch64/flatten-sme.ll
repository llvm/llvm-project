; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+sme -S -passes=always-inline | FileCheck %s
; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+sme -S -passes=inline | FileCheck %s

; Test that flatten attribute respects ABI restrictions for SME.
; Streaming callee cannot be inlined into non-streaming caller.
; new_za callee cannot be inlined at all.

define internal i32 @streaming_callee() "aarch64_pstate_sm_enabled" {
  ret i32 42
}

define internal i32 @new_za_callee() "aarch64_new_za" {
  ret i32 100
}

define internal i32 @normal_callee() {
  ret i32 50
}

; Streaming callee -> non-streaming caller: should NOT be inlined (ABI violation).
define i32 @test_streaming_not_inlined() flatten {
; CHECK-LABEL: @test_streaming_not_inlined(
; CHECK: call i32 @streaming_callee()
; CHECK: ret i32
  %r = call i32 @streaming_callee()
  ret i32 %r
}

; new_za callee: should NOT be inlined (ABI violation - callee allocates new ZA).
define i32 @test_new_za_not_inlined() flatten {
; CHECK-LABEL: @test_new_za_not_inlined(
; CHECK: call i32 @new_za_callee()
; CHECK: ret i32
  %r = call i32 @new_za_callee()
  ret i32 %r
}

; Streaming caller -> streaming callee: should be inlined (compatible).
define i32 @test_streaming_to_streaming() flatten "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @test_streaming_to_streaming(
; CHECK-NOT: call i32 @streaming_callee
; CHECK: ret i32 42
  %r = call i32 @streaming_callee()
  ret i32 %r
}

; Non-streaming caller -> non-streaming callee: should be inlined.
define i32 @test_normal_inlined() flatten {
; CHECK-LABEL: @test_normal_inlined(
; CHECK-NOT: call i32 @normal_callee
; CHECK: ret i32 50
  %r = call i32 @normal_callee()
  ret i32 %r
}
