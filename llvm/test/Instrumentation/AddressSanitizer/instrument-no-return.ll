; RUN: opt < %s -asan -asan-module -S | FileCheck %s
; AddressSanitizer must insert __asan_handle_no_return
; before every noreturn call or invoke.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @MyNoReturnFunc(i32) noreturn

define i32 @Call1(i8* nocapture %arg) uwtable sanitize_address {
entry:
  call void @MyNoReturnFunc(i32 1) noreturn  ; The call insn has noreturn attr.
; CHECK:        @Call1
; CHECK:        call void @__asan_handle_no_return
; CHECK-NEXT:   call void @MyNoReturnFunc
; CHECK-NEXT: unreachable
  unreachable
}

define i32 @Call2(i8* nocapture %arg) uwtable sanitize_address {
entry:
  call void @MyNoReturnFunc(i32 1)  ; No noreturn attribure on the call.
; CHECK:        @Call2
; CHECK:        call void @__asan_handle_no_return
; CHECK-NEXT:   call void @MyNoReturnFunc
; CHECK-NEXT: unreachable
  unreachable
}

declare i32 @__gxx_personality_v0(...)

define i64 @Invoke1(i8** %esc) nounwind uwtable ssp sanitize_address personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @MyNoReturnFunc(i32 1)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i64 0

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  ret i64 1
}
; CHECK: @Invoke1
; CHECK:        call void @__asan_handle_no_return
; CHECK-NEXT:   invoke void @MyNoReturnFunc
; CHECK: ret i64 0
; CHECK: ret i64 1
