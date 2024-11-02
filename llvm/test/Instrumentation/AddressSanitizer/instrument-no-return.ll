; RUN: opt < %s -passes=asan -S | FileCheck %s
; AddressSanitizer must insert __asan_handle_no_return
; before noreturn calls that aren't inserted by sanitizers.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @NormalFunc()
declare void @NoReturnFunc() noreturn

; Instrument noreturn callsites (regardless of function)
define i32 @Call1() sanitize_address {
  call void @NormalFunc() noreturn
  unreachable
}
; CHECK-LABEL:  @Call1
; CHECK-NEXT:   call void @__asan_handle_no_return
; CHECK-NEXT:   call void @NormalFunc

; Instrument calls to noreturn functions (regardless of callsite)
define i32 @Call2() sanitize_address {
  call void @NoReturnFunc()
  unreachable
}
; CHECK-LABEL:  @Call2
; CHECK-NEXT:   call void @__asan_handle_no_return
; CHECK-NEXT:   call void @NoReturnFunc

; Do *not* instrument callsites marked !nosanitize
define i32 @Call3() sanitize_address {
  call void @NoReturnFunc() noreturn, !nosanitize !{}
  unreachable
}
; CHECK-LABEL:  @Call3
; CHECK-NOT:    call void @__asan_handle_no_return
; CHECK:        call void @NoReturnFunc

; Do *not* instrument functions without ASan
define i32 @Call4() {
  call void @NoReturnFunc() noreturn
  unreachable
}
; CHECK-LABEL:  @Call4
; CHECK-NOT:    call void @__asan_handle_no_return
; CHECK:        call void @NoReturnFunc

declare i32 @__gxx_personality_v0(...)

define i64 @Invoke1() nounwind uwtable ssp sanitize_address personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @NoReturnFunc()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i64 0

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  ret i64 1
}
; CHECK-LABEL:  @Invoke1
; CHECK:        call void @__asan_handle_no_return
; CHECK-NEXT:   invoke void @NoReturnFunc
; CHECK: ret i64 0
; CHECK: ret i64 1
