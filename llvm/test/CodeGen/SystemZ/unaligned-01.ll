; Check that unaligned accesses are allowed in general.  We check the
; few exceptions (like CRL) in their respective test files.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check that these four byte stores become a single word store.
define void @f1(ptr %ptr) {
; CHECK: f1
; CHECK: iilf [[REG:%r[0-5]]], 66051
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %off1 = getelementptr i8, ptr %ptr, i64 1
  %off2 = getelementptr i8, ptr %ptr, i64 2
  %off3 = getelementptr i8, ptr %ptr, i64 3
  store i8 0, ptr %ptr
  store i8 1, ptr %off1
  store i8 2, ptr %off2
  store i8 3, ptr %off3
  ret void
}

; Check that unaligned 2-byte accesses are allowed.
define i16 @f2(ptr %src, ptr %dst) {
; CHECK-LABEL: f2:
; CHECK: lh %r2, 0(%r2)
; CHECK: sth %r2, 0(%r3)
; CHECK: br %r14
  %val = load i16, ptr %src, align 1
  store i16 %val, ptr %dst, align 1
  ret i16 %val
}

; Check that unaligned 4-byte accesses are allowed.
define i32 @f3(ptr %src1, ptr %src2, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: l %r2, 0(%r2)
; CHECK: s %r2, 0(%r3)
; CHECK: st %r2, 0(%r4)
; CHECK: br %r14
  %val1 = load i32, ptr %src1, align 1
  %val2 = load i32, ptr %src2, align 2
  %sub = sub i32 %val1, %val2
  store i32 %sub, ptr %dst, align 1
  ret i32 %sub
}

; Check that unaligned 8-byte accesses are allowed.
define i64 @f4(ptr %src1, ptr %src2, ptr %dst) {
; CHECK-LABEL: f4:
; CHECK: lg %r2, 0(%r2)
; CHECK: sg %r2, 0(%r3)
; CHECK: stg %r2, 0(%r4)
; CHECK: br %r14
  %val1 = load i64, ptr %src1, align 1
  %val2 = load i64, ptr %src2, align 2
  %sub = sub i64 %val1, %val2
  store i64 %sub, ptr %dst, align 4
  ret i64 %sub
}
