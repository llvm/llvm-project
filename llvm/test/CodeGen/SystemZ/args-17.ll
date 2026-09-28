; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension of an i8 return value.

define i8 @callee_MissingRetAttr() {
  ret i8 -1
}

; CHECK: ERROR:  (C ABI violiation) missing extension attribute on arg 0.
; CHECK: Returning from function: i8 @callee_MissingRetAttr()
; CHECK: UNREACHABLE executed
