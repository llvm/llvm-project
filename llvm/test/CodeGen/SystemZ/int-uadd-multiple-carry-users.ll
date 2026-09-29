; RUN: llc < %s -mtriple=s390x-linux-gnu -start-after=codegenprepare | FileCheck %s
;
; Test that compilation succeeds with multiple users of the carry resulting
; in a truncation of the SELECT_CCMASK used by GET_CCMASK.

define void @fun() {
; CHECK-LABEL: fun:
bb:
  %0 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 poison, i64 poison)
  %ov = extractvalue { i64, i1 } %0, 1
  %i2 = sext i1 %ov to i64
  %i3 = select i1 %ov, i64 9223372036854775807, i64 -1
  %i4 = sub nsw i64 %i3, %i2
  %i5 = and i64 %i4, %i2
  %i6 = icmp slt i64 %i5, 0
  %i7 = xor i1 %ov, true
  %i8 = select i1 %i6, i1 %i7, i1 false
  %i9 = sext i1 %i8 to i16
  store i16 %i9, ptr poison, align 2
  unreachable
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
