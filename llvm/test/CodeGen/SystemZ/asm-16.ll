; Test the "M" constraint (0x7fffffff)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

; Test 1 below the valid value.
define i32 @f1() {
; CHECK-LABEL: f1:
; CHECK: iilf [[REG:%r[0-5]]], 2147483646
; CHECK: blah %r2 [[REG]]
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rM" (i32 2147483646)
  ret i32 %val
}

; Test the first valid value.
define i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: blah %r2 2147483647
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rM" (i32 2147483647)
  ret i32 %val
}

; Test 1 above the valid value.
define i32 @f3() {
; CHECK-LABEL: f3:
; CHECK: llilh [[REG:%r[0-5]]], 32768
; CHECK: blah %r2 [[REG]]
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rM" (i32 2147483648)
  ret i32 %val
}
