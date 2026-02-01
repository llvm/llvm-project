; RUN: llc -mtriple=mips64el -mcpu=r5900 < %s | FileCheck %s
;
; Test that R5900 FPU comparisons only generate supported instructions.
; R5900 only supports: C.F (0x30), C.EQ (0x32), C.OLT (0x34), C.OLE (0x36)
;
; Unsupported conditions are transformed:
; - Unordered (ULT, ULE, UEQ, etc.) → Ordered equivalents (R5900 has no NaN)
; - Greater-than → Less-than with swapped operands

;-----------------------------------------------------------------------------
; Ordered comparisons
;-----------------------------------------------------------------------------

define i32 @test_oeq(float %a, float %b) {
; CHECK-LABEL: test_oeq:
; CHECK: c.eq.s $f12, $f13
; CHECK-NOT: c.ueq
  %cmp = fcmp oeq float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_olt(float %a, float %b) {
; CHECK-LABEL: test_olt:
; CHECK: c.olt.s $f12, $f13
; CHECK-NOT: c.ult
  %cmp = fcmp olt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ole(float %a, float %b) {
; CHECK-LABEL: test_ole:
; CHECK: c.ole.s $f12, $f13
; CHECK-NOT: c.ule
  %cmp = fcmp ole float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ogt(float %a, float %b) {
; CHECK-LABEL: test_ogt:
; CHECK: c.olt.s $f13, $f12
; Swapped operands: a > b  →  b < a
  %cmp = fcmp ogt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_oge(float %a, float %b) {
; CHECK-LABEL: test_oge:
; CHECK: c.ole.s $f13, $f12
; Swapped operands: a >= b  →  b <= a
  %cmp = fcmp oge float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

;-----------------------------------------------------------------------------
; Unordered comparisons - mapped to ordered (R5900 FPU has no NaN)
;-----------------------------------------------------------------------------

define i32 @test_ueq(float %a, float %b) {
; CHECK-LABEL: test_ueq:
; CHECK: c.eq.s $f12, $f13
; CHECK-NOT: c.ueq
  %cmp = fcmp ueq float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ult(float %a, float %b) {
; CHECK-LABEL: test_ult:
; CHECK: c.olt.s $f12, $f13
; CHECK-NOT: c.ult
  %cmp = fcmp ult float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ule(float %a, float %b) {
; CHECK-LABEL: test_ule:
; CHECK: c.ole.s $f12, $f13
; CHECK-NOT: c.ule
  %cmp = fcmp ule float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ugt(float %a, float %b) {
; CHECK-LABEL: test_ugt:
; CHECK: c.olt.s $f13, $f12
; Swapped operands: a > b  →  b < a
  %cmp = fcmp ugt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_uge(float %a, float %b) {
; CHECK-LABEL: test_uge:
; CHECK: c.ole.s $f13, $f12
; Swapped operands: a >= b  →  b <= a
  %cmp = fcmp uge float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

;-----------------------------------------------------------------------------
; Simple comparisons (non-prefixed)
;-----------------------------------------------------------------------------

define i32 @test_eq(float %a, float %b) {
; CHECK-LABEL: test_eq:
; CHECK: c.eq.s $f12, $f13
  %cmp = fcmp oeq float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_lt(float %a, float %b) {
; CHECK-LABEL: test_lt:
; CHECK: c.olt.s $f12, $f13
  %cmp = fcmp olt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_le(float %a, float %b) {
; CHECK-LABEL: test_le:
; CHECK: c.ole.s $f12, $f13
  %cmp = fcmp ole float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_gt(float %a, float %b) {
; CHECK-LABEL: test_gt:
; CHECK: c.olt.s $f13, $f12
  %cmp = fcmp ogt float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_ge(float %a, float %b) {
; CHECK-LABEL: test_ge:
; CHECK: c.ole.s $f13, $f12
  %cmp = fcmp oge float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

;-----------------------------------------------------------------------------
; Not-equal comparisons
;-----------------------------------------------------------------------------

define i32 @test_one(float %a, float %b) {
; CHECK-LABEL: test_one:
; CHECK: c.eq.s
; Uses EQ and inverts result via bc1f/bc1t
  %cmp = fcmp one float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_une(float %a, float %b) {
; CHECK-LABEL: test_une:
; CHECK: c.eq.s
; Uses EQ and inverts result via bc1f/bc1t
  %cmp = fcmp une float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

;-----------------------------------------------------------------------------
; Ordered/Unordered predicates
;-----------------------------------------------------------------------------

define i32 @test_ord(float %a, float %b) {
; CHECK-LABEL: test_ord:
; R5900 has no NaN, so ordered is always true
; Should optimize or use FCOND_T (inverted F)
  %cmp = fcmp ord float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

define i32 @test_uno(float %a, float %b) {
; CHECK-LABEL: test_uno:
; R5900 has no NaN, so unordered is always false
; CHECK: c.f.s
  %cmp = fcmp uno float %a, %b
  %result = zext i1 %cmp to i32
  ret i32 %result
}

;-----------------------------------------------------------------------------
; Verify no unsupported instructions are generated anywhere
;-----------------------------------------------------------------------------
; CHECK-NOT: c.un.s
; CHECK-NOT: c.ueq.s
; CHECK-NOT: c.ult.s
; CHECK-NOT: c.ule.s
; CHECK-NOT: c.sf.s
; CHECK-NOT: c.ngle.s
; CHECK-NOT: c.seq.s
; CHECK-NOT: c.ngl.s
; CHECK-NOT: c.lt.s
; CHECK-NOT: c.nge.s
; CHECK-NOT: c.le.s
; CHECK-NOT: c.ngt.s
