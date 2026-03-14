; Test matching operands with the GPR constraint "r".
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

define void @f1(ptr %dst) {
; CHECK-LABEL: f1:
; CHECK: lhi %r0, 100
; CHECK: blah %r0
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = call i32 asm "blah $0", "=r,0" (i8 100)
  store i32 %val, ptr %dst
  ret void
}

define void @f2(ptr %dst) {
; CHECK-LABEL: f2:
; CHECK: lhi %r0, 101
; CHECK: blah %r0
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = call i32 asm "blah $0", "=r,0" (i16 101)
  store i32 %val, ptr %dst
  ret void
}

define void @f3(ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: lhi %r0, 102
; CHECK: blah %r0
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = call i32 asm "blah $0", "=r,0" (i32 102)
  store i32 %val, ptr %dst
  ret void
}

; FIXME: this uses "lhi %r0, 103", but should use "lghi %r0, 103".
define void @f4(ptr %dst) {
; CHECK-LABEL: f4:
; CHECK: blah %r0
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = call i32 asm "blah $0", "=r,0" (i64 103)
  store i32 %val, ptr %dst
  ret void
}

define i64 @f5() {
; CHECK-LABEL: f5:
; CHECK: lghi %r2, 104
; CHECK: blah %r2
; CHECK: br %r14
  %val = call i64 asm "blah $0", "=r,0" (i8 104)
  ret i64 %val
}

define i64 @f6() {
; CHECK-LABEL: f6:
; CHECK: lghi %r2, 105
; CHECK: blah %r2
; CHECK: br %r14
  %val = call i64 asm "blah $0", "=r,0" (i16 105)
  ret i64 %val
}

define i64 @f7() {
; CHECK-LABEL: f7:
; CHECK: lghi %r2, 106
; CHECK: blah %r2
; CHECK: br %r14
  %val = call i64 asm "blah $0", "=r,0" (i32 106)
  ret i64 %val
}

define i64 @f8() {
; CHECK-LABEL: f8:
; CHECK: lghi %r2, 107
; CHECK: blah %r2
; CHECK: br %r14
  %val = call i64 asm "blah $0", "=r,0" (i64 107)
  ret i64 %val
}
