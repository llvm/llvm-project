; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test an indirect call: even without the function prototype we can check that
; a C ABI call has some kind of extension.

define void @caller(ptr %fptr) {
  call void %fptr(i32 0)
  ret void
}

; CHECK: ERROR:  (C ABI violiation) missing extension attribute on arg 0.
; CHECK: Callee: -
; CHECK: Caller: void @caller(ptr)
; CHECK:         call void %fptr(i32 0)
