; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test varargs: call with wrong extension type on a fixed argument.

define i64 @caller(i32 %Arg) {
  %res = call i64 @bar(i32 signext %Arg, i32 zeroext 0)
  ret i64 %res
}

define i64 @bar(i32 zeroext %Arg, ...) {
  %S = zext i32 %Arg to i64
  ret i64 %S
}

; CHECK: ERROR : Missing ZExt on arg 0.
; CHECK: Callee: i64 @bar(i32 zeroext, ...)
; CHECK: Caller: i64 @caller(i32)
; CHECK:         %res = call i64 @bar(i32 signext %Arg, i32 zeroext 0)
