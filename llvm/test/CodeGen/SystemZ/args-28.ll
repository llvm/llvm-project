; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test varargs: call with missing extension type on a vararg.

define i64 @caller(i32 %Arg) {
  %res = call i64 @bar(i32 zeroext %Arg, i32 0)
  ret i64 %res
}

define i64 @bar(i32 zeroext %Arg, ...) {
  %S = zext i32 %Arg to i64
  ret i64 %S
}

; CHECK: ERROR:  (C ABI violiation) missing extension attribute on arg 1.
; CHECK: Callee: i64 @bar(i32 zeroext, ...)
; CHECK: Caller: i64 @caller(i32)
; CHECK:         %res = call i64 @bar(i32 zeroext %Arg, i32 0)
