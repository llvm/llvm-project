; Test the GPR constraint "a", which forbids %r0.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

define i64 @f1() {
; CHECK-LABEL: f1:
; CHECK: lhi %r1, 1
; CHECK: blah %r2 %r1
; CHECK: br %r14
  %val = call i64 asm "blah $0 $1", "=r,a" (i8 1)
  ret i64 %val
}

define i64 @f2() {
; CHECK-LABEL: f2:
; CHECK: lhi %r1, 2
; CHECK: blah %r2 %r1
; CHECK: br %r14
  %val = call i64 asm "blah $0 $1", "=r,a" (i16 2)
  ret i64 %val
}

define i64 @f3() {
; CHECK-LABEL: f3:
; CHECK: lhi %r1, 3
; CHECK: blah %r2 %r1
; CHECK: br %r14
  %val = call i64 asm "blah $0 $1", "=r,a" (i32 3)
  ret i64 %val
}

define i64 @f4() {
; CHECK-LABEL: f4:
; CHECK: lghi %r1, 4
; CHECK: blah %r2 %r1
; CHECK: br %r14
  %val = call i64 asm "blah $0 $1", "=r,a" (i64 4)
  ret i64 %val
}
