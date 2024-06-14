; RUN: llc -stop-after finalize-isel < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; This test checks to make sure the lshr in %then1 block gets expanded using
; GR16_ABCD pattern rather than GR32_ABCD pattern.  By using the 16-bit pattern
; this doesn't make the register liveness information look like the whole
; 32-bit register is a live value, and allows generally better live register
; analysis.
; CHECK-LABEL: bb.1.then1:
; CHECK-NOT:   IMPLICIT_DEF
; CHECK-NOT:   INSERT_SUBREG
; CHECK:       sub_8bit_hi
; CHECK-LABEL: bb.2.endif1:

define i16 @foo4(i32 %prec, ptr%dst, ptr%src) {
entry:
  %cnd = icmp ne i32 %prec, 0
  %t0 = load i16, ptr%src, align 2
  br i1 %cnd, label %then1, label %endif1

then1:
  %shr = lshr i16 %t0, 8
  %conv = trunc i16 %shr to i8
  store i8 %conv, ptr%dst, align 1
  br label %endif1

endif1:
  %t2 = phi i16 [0, %then1], [%t0, %entry]
  ret i16 %t2
}
