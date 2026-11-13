; RUN: opt -passes=instcombine -S < %s | llc -mtriple=x86_64-unknown-linux-gnu -O2 | FileCheck %s
;
; A pair of 128-bit subtractions should use two sub/sbb pairs after
; InstCombine. See https://github.com/llvm/llvm-project/issues/228028.
;
; CHECK-LABEL: chained_sub_borrow:
; CHECK:       subq %rdx, %rax
; CHECK-NEXT:  sbbq %rcx, %rsi
; CHECK-NEXT:  subq %r8, %rax
; CHECK-NEXT:  sbbq %r9, %rsi

define { i64, i64 } @chained_sub_borrow(i64 %x.lo, i64 %x.hi, i64 %y.lo, i64 %y.hi, i64 %z.lo, i64 %z.hi) {
  %s1 = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %x.lo, i64 %y.lo)
  %b1 = extractvalue { i64, i1 } %s1, 1
  %t.lo = extractvalue { i64, i1 } %s1, 0
  %b1.neg = sext i1 %b1 to i64
  %d1 = sub i64 %x.hi, %y.hi
  %t.hi = add i64 %d1, %b1.neg
  %s2 = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %t.lo, i64 %z.lo)
  %b2 = extractvalue { i64, i1 } %s2, 1
  %r.lo = extractvalue { i64, i1 } %s2, 0
  %b2.neg = sext i1 %b2 to i64
  %d2 = sub i64 %t.hi, %z.hi
  %r.hi = add i64 %d2, %b2.neg
  %r0 = insertvalue { i64, i64 } poison, i64 %r.lo, 0
  %r1 = insertvalue { i64, i64 } %r0, i64 %r.hi, 1
  ret { i64, i64 } %r1
}

declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64)
