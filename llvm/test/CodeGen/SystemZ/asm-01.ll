; Test the "Q" asm constraint, which accepts addresses that have a base
; and a 12-bit displacement.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

; Check the lowest range.
define void @f1(i64 %base) {
; CHECK-LABEL: f1:
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %addr = inttoptr i64 %base to ptr
  call void asm "blah $0", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}

; Check the next lowest byte.
define void @f2(i64 %base) {
; CHECK-LABEL: f2:
; CHECK: aghi %r2, -1
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, -1
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}

; Check the highest range.
define void @f3(i64 %base) {
; CHECK-LABEL: f3:
; CHECK: blah 4095(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4095
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}

; Check the next highest byte.
define void @f4(i64 %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4096
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}

; Check that indices aren't allowed
define void @f5(i64 %base, i64 %index) {
; CHECK-LABEL: f5:
; CHECK: agr %r2, %r3
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, %index
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}

; Check A, O and R format flags.
define void @f6(i64 %base) {
; CHECK-LABEL: f6:
; CHECK: blah 111,%r2
; CHECK: br %r14
  %add = add i64 %base, 111
  %addr = inttoptr i64 %add to ptr
  call void asm "blah ${0:O},${0:R}${0:A}", "=*Q" (ptr elementtype(i64) %addr)
  ret void
}
