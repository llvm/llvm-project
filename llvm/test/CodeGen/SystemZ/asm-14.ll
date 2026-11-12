; Test the "L" constraint (20-bit signed constants).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

; Test 1 below the first valid value.
define i32 @f1() {
; CHECK-LABEL: f1:
; CHECK: iilf [[REG:%r[0-5]]], 4294443007
; CHECK: blah %r2 [[REG]]
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rL" (i32 -524289)
  ret i32 %val
}

; Test the first valid value.
define i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: blah %r2 -524288
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rL" (i32 -524288)
  ret i32 %val
}

; Test the last valid value.
define i32 @f3() {
; CHECK-LABEL: f3:
; CHECK: blah %r2 524287
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rL" (i32 524287)
  ret i32 %val
}

; Test 1 above the last valid value.
define i32 @f4() {
; CHECK-LABEL: f4:
; CHECK: llilh [[REG:%r[0-5]]], 8
; CHECK: blah %r2 [[REG]]
; CHECK: br %r14
  %val = call i32 asm "blah $0 $1", "=&r,rL" (i32 524288)
  ret i32 %val
}
