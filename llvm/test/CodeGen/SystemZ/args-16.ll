; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension of an i16 return value.

define internal i16 @callee_MissingRetAttr() {
  ret i16 -1
}

declare void @ExtFun(ptr %FunPtr)
define void @foo() {
  call void @ExtFun(ptr @callee_MissingRetAttr)
  ret void
}

; CHECK: ERROR:  (C ABI violiation) missing extension attribute on arg 0.
; CHECK: Returning from function: i16 @callee_MissingRetAttr()
; CHECK: UNREACHABLE executed

