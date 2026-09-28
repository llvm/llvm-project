; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test varargs: call with the wrong extension kind on an argument, and also
; test that the split i128 arg (z10) does not mess things up.

define i64 @caller(i32 %Arg) {
  %res = call i64 @bar(i32 noext %Arg, i128 0, i32 signext %Arg)
  ret i64 %res
}

define i64 @bar(i32 noext %A0, i128 %A1, i32 zeroext %A2, ...) {
  %S = zext i32 %A2 to i64
  ret i64 %S
}

; CHECK: ERROR : Missing ZExt on arg 2.
; CHECK: Callee: i64 @bar(i32 noext, i128, i32 zeroext, ...)
; CHECK: Caller: i64 @caller(i32)
; CHECK:         %res = call i64 @bar(i32 noext %Arg, i128 0, i32 signext %Arg)
