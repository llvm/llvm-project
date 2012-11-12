; RUN: opt < %s -asan -S | FileCheck %s
; AddressSanitizer must insert __asan_handle_no_return
; before every noreturn call.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @MyNoReturnFunc(i32) noreturn

define i32 @_Z5ChildPv(i8* nocapture %arg) uwtable address_safety {
entry:
  call void @MyNoReturnFunc(i32 1) noreturn
  unreachable
}

; CHECK:        call void @__asan_handle_no_return
; CHECK-NEXT:   call void @MyNoReturnFunc
