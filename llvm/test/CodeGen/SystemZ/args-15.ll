; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension of an i32 return value.

define i32 @callee_MissingRetAttr() {
  ret i32 -1
}

; CHECK: Narrow integer argument must have a valid extension type.
