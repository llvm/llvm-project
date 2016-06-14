; Test the "R" asm constraint, which accepts addresses that have a base,
; an index and a 12-bit displacement.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

; Check the lowest range.
define void @f1(i64 %base) {
; CHECK-LABEL: f1:
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %addr = inttoptr i64 %base to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the next lowest byte.
define void @f2(i64 %base) {
; CHECK-LABEL: f2:
; CHECK: aghi %r2, -1
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, -1
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the highest range.
define void @f3(i64 %base) {
; CHECK-LABEL: f3:
; CHECK: blah 4095(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4095
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the next highest byte.
define void @f4(i64 %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4096
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check that indices are allowed
define void @f5(i64 %base, i64 %index) {
; CHECK-LABEL: f5:
; CHECK: blah 0(%r3,%r2)
; CHECK: br %r14
  %add = add i64 %base, %index
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check that indices and displacements are allowed simultaneously
define void @f6(i64 %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: blah 4095(%r3,%r2)
; CHECK: br %r14
  %add = add i64 %base, 4095
  %addi = add i64 %add, %index
  %addr = inttoptr i64 %addi to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check that LAY is used if there is an index but the displacement is too large
define void @f7(i64 %base, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lay %r0, 4096(%r3,%r2)
; CHECK: blah 0(%r0)
; CHECK: br %r14
  %add = add i64 %base, 4096
  %addi = add i64 %add, %index
  %addr = inttoptr i64 %addi to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}
