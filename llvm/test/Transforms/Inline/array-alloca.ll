; RUN: opt -inline -S < %s | FileCheck %s
%struct.A = type { i32 }

define void @callee1(i32 %M) {
entry:
  %vla = alloca i32, i32 %M, align 16
  ret void
}

define void @callee2(i32 %M) {
entry:
  %vla = alloca %struct.A, i32 %M, align 16
  ret void
}

define void @callee3(i128 %M) {
entry:
  %vla = alloca i32, i128 %M, align 16
  ret void
}

; CHECK-LABEL: @caller
define void @caller() #0 {
entry:
  call void @caller()
; CHECK-NOT: call void @callee1
  call void @callee1(i32 256)
; CHECK: call void @callee2
  call void @callee2(i32 4096)
; CHECK: call void @callee3
; This is to test that there is no overflow in computing allocated size
; call void @callee3(i128 0x8000000000000000);
  call void @callee3(i128 9223372036854775808);
  ret void
}

