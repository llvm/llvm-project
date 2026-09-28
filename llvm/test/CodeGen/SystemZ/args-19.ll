; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension of an outgoing i16 call argument.

define void @caller() {
  call void @bar_Struct(i16 123)
  ret void
}

define internal void @bar_Struct(i16 %Arg) { ret void }

declare void @ExtFun(ptr %FunPtr)
define void @foo() {
  call void @ExtFun(ptr @bar_Struct)
  ret void
}

; CHECK: ERROR:  (C ABI violiation) missing extension attribute on arg 0.
; CHECK: Callee: void @bar_Struct(i16)
; CHECK: Caller: void @caller()
; CHECK:         call void @bar_Struct(i16 123)
; CHECK: UNREACHABLE executed
