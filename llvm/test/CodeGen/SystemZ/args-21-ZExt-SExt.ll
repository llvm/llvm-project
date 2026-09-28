; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of a call making the wrong kind of extension.

define internal i64 @fun(i32 zeroext %Arg) {
  %S = zext i32 %Arg to i64
  ret i64 %S
}

define i64 @foo(i32 %Arg) {
  %res = call i64 @fun(i32 signext %Arg)
  ret i64 %res
}

; CHECK: ERROR : SExt and ZExt are incompatible (arg 0).
; CHECK: Callee: i64 @fun(i32 zeroext)
; CHECK: Caller: i64 @foo(i32)
