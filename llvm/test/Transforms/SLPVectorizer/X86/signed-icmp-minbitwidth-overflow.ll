; Test that the SLPVectorizer does not miscompile signed integer comparisons
; against constants that overflow the reduced bit-width.
;
; The SLPVectorizer's min-bitwidth analysis may attempt to demote the operands
; of a vectorized ICmpInst from a wider type (e.g. i32) to a narrower type
; (e.g. i16).  When a signed predicate is used and a constant operand does NOT
; fit in the signed range of the narrower type, the truncation flips the sign of
; the constant and inverts the comparison, producing wrong code.
;
; Example from https://github.com/llvm/llvm-project/issues/209010 :
;   (i32 %x) < 58593   -- 58593 fits in i32 but NOT in i16 (max signed = 32767)
; Demoting to i16 produces (i16 %x) < -6943 (= 58593 trunc to i16), which is
; semantically incorrect for values of %x near 58593.
;
; RUN: opt -passes=slp-vectorizer -S -mtriple=x86_64-linux-gnu < %s | FileCheck %s
; RUN: opt -passes=slp-vectorizer -S -mtriple=x86_64-linux-gnu -mcpu=x86-64 < %s | FileCheck %s

; CHECK-LABEL: @test_signed_icmp_const_overflow
; The key check is that we do NOT see an i16 comparison with the truncated
; constant.  The operands must remain i32 (or wider).
; CHECK-NOT: icmp {{s[lg][te]}} <{{[0-9]+}} x i16>
; CHECK-NOT: i16 -6943
; CHECK-NOT: i16 -6944
define i32 @test_signed_icmp_const_overflow(ptr %p) {
entry:
  ; Load 8 i32 values.
  %p0  = getelementptr i32, ptr %p, i64 0
  %p1  = getelementptr i32, ptr %p, i64 1
  %p2  = getelementptr i32, ptr %p, i64 2
  %p3  = getelementptr i32, ptr %p, i64 3
  %p4  = getelementptr i32, ptr %p, i64 4
  %p5  = getelementptr i32, ptr %p, i64 5
  %p6  = getelementptr i32, ptr %p, i64 6
  %p7  = getelementptr i32, ptr %p, i64 7
  %v0  = load i32, ptr %p0, align 4
  %v1  = load i32, ptr %p1, align 4
  %v2  = load i32, ptr %p2, align 4
  %v3  = load i32, ptr %p3, align 4
  %v4  = load i32, ptr %p4, align 4
  %v5  = load i32, ptr %p5, align 4
  %v6  = load i32, ptr %p6, align 4
  %v7  = load i32, ptr %p7, align 4

  ; Signed comparisons against 58593 (> INT16_MAX = 32767, so it does NOT fit
  ; in a signed 16-bit integer; truncating to i16 gives -6943, which is wrong).
  %c0  = icmp slt i32 %v0, 58593
  %c1  = icmp slt i32 %v1, 58593
  %c2  = icmp slt i32 %v2, 58593
  %c3  = icmp slt i32 %v3, 58593
  %c4  = icmp slt i32 %v4, 58593
  %c5  = icmp slt i32 %v5, 58593
  %c6  = icmp slt i32 %v6, 58593
  %c7  = icmp slt i32 %v7, 58593

  ; Combine results.
  %r0  = zext i1 %c0 to i32
  %r1  = zext i1 %c1 to i32
  %r2  = zext i1 %c2 to i32
  %r3  = zext i1 %c3 to i32
  %r4  = zext i1 %c4 to i32
  %r5  = zext i1 %c5 to i32
  %r6  = zext i1 %c6 to i32
  %r7  = zext i1 %c7 to i32
  %sum01 = add i32 %r0, %r1
  %sum23 = add i32 %r2, %r3
  %sum45 = add i32 %r4, %r5
  %sum67 = add i32 %r6, %r7
  %sum0123 = add i32 %sum01, %sum23
  %sum4567 = add i32 %sum45, %sum67
  %total   = add i32 %sum0123, %sum4567
  ret i32 %total
}

; Unsigned comparisons against a value that overflows an unsigned 16-bit type
; should still be demotable correctly (0xE4A1 = 58593 fits in u16 = up to 65535).
; This test ensures we don't over-reject unsigned cases.
; CHECK-LABEL: @test_unsigned_icmp_fits_u16
define i32 @test_unsigned_icmp_fits_u16(ptr %p) {
entry:
  %p0  = getelementptr i32, ptr %p, i64 0
  %p1  = getelementptr i32, ptr %p, i64 1
  %p2  = getelementptr i32, ptr %p, i64 2
  %p3  = getelementptr i32, ptr %p, i64 3
  %p4  = getelementptr i32, ptr %p, i64 4
  %p5  = getelementptr i32, ptr %p, i64 5
  %p6  = getelementptr i32, ptr %p, i64 6
  %p7  = getelementptr i32, ptr %p, i64 7
  %v0  = load i32, ptr %p0, align 4
  %v1  = load i32, ptr %p1, align 4
  %v2  = load i32, ptr %p2, align 4
  %v3  = load i32, ptr %p3, align 4
  %v4  = load i32, ptr %p4, align 4
  %v5  = load i32, ptr %p5, align 4
  %v6  = load i32, ptr %p6, align 4
  %v7  = load i32, ptr %p7, align 4

  ; Unsigned comparison: 58593 fits in u16, so demotion to u16 is safe.
  %c0  = icmp ult i32 %v0, 58593
  %c1  = icmp ult i32 %v1, 58593
  %c2  = icmp ult i32 %v2, 58593
  %c3  = icmp ult i32 %v3, 58593
  %c4  = icmp ult i32 %v4, 58593
  %c5  = icmp ult i32 %v5, 58593
  %c6  = icmp ult i32 %v6, 58593
  %c7  = icmp ult i32 %v7, 58593

  %r0  = zext i1 %c0 to i32
  %r1  = zext i1 %c1 to i32
  %r2  = zext i1 %c2 to i32
  %r3  = zext i1 %c3 to i32
  %r4  = zext i1 %c4 to i32
  %r5  = zext i1 %c5 to i32
  %r6  = zext i1 %c6 to i32
  %r7  = zext i1 %c7 to i32
  %sum01 = add i32 %r0, %r1
  %sum23 = add i32 %r2, %r3
  %sum45 = add i32 %r4, %r5
  %sum67 = add i32 %r6, %r7
  %sum0123 = add i32 %sum01, %sum23
  %sum4567 = add i32 %sum45, %sum67
  %total   = add i32 %sum0123, %sum4567
  ret i32 %total
}
