; Test the "T" asm constraint, which accepts addresses that have a base,
; an index and a 20-bit displacement.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

; Check the lowest range.
define void @f1(i64 %base) {
; CHECK-LABEL: f1:
; CHECK: blah -524288(%r2)
; CHECK: br %r14
  %add = add i64 %base, -524288
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}

; Check the next lowest byte.
define void @f2(i64 %base) {
; CHECK-LABEL: f2:
; CHECK: agfi %r2, -524289
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, -524289
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}

; Check the highest range.
define void @f3(i64 %base) {
; CHECK-LABEL: f3:
; CHECK: blah 524287(%r2)
; CHECK: br %r14
  %add = add i64 %base, 524287
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}

; Check the next highest byte.
define void @f4(i64 %base) {
; CHECK-LABEL: f4:
; CHECK: agfi %r2, 524288
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, 524288
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}

; Check that indices are allowed
define void @f5(i64 %base, i64 %index) {
; CHECK-LABEL: f5:
; CHECK: blah 0(%r3,%r2)
; CHECK: br %r14
  %add = add i64 %base, %index
  %addr = inttoptr i64 %add to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}

; Check that indices and displacements are allowed simultaneously
define void @f6(i64 %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: blah 524287(%r3,%r2)
; CHECK: br %r14
  %add = add i64 %base, 524287
  %addi = add i64 %add, %index
  %addr = inttoptr i64 %addi to ptr
  call void asm "blah $0", "=*T" (ptr elementtype(i64) %addr)
  ret void
}
