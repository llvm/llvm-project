; Test 32-bit floating-point comparison.  The tests assume a z10 implementation
; of select, using conditional branches rather than LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-VECTOR %s

declare float @foo()

; Check comparison with registers.
define i64 @f1(i64 %a, i64 %b, float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: cebr %f0, %f2
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the low end of the CEB range.
define i64 @f2(i64 %a, i64 %b, float %f1, float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f2 = load float, float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the high end of the aligned CEB range.
define i64 @f3(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f3:
; CHECK: ceb %f0, 4092(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %f2 = load float, float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r4, 4096
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %f2 = load float, float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check negative displacements, which also need separate address logic.
define i64 @f5(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r4, -4
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %f2 = load float, float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that CEB allows indices.
define i64 @f6(i64 %a, i64 %b, float %f1, float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r5, 2
; CHECK: ceb %f0, 400(%r1,%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %f2 = load float, float *%ptr2
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that comparisons of spilled values can use CEB rather than CEBR.
define float @f7(float *%ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: ceb {{%f[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%ptr0, i64 2
  %ptr2 = getelementptr float, float *%ptr0, i64 4
  %ptr3 = getelementptr float, float *%ptr0, i64 6
  %ptr4 = getelementptr float, float *%ptr0, i64 8
  %ptr5 = getelementptr float, float *%ptr0, i64 10
  %ptr6 = getelementptr float, float *%ptr0, i64 12
  %ptr7 = getelementptr float, float *%ptr0, i64 14
  %ptr8 = getelementptr float, float *%ptr0, i64 16
  %ptr9 = getelementptr float, float *%ptr0, i64 18
  %ptr10 = getelementptr float, float *%ptr0, i64 20

  %val0 = load float, float *%ptr0
  %val1 = load float, float *%ptr1
  %val2 = load float, float *%ptr2
  %val3 = load float, float *%ptr3
  %val4 = load float, float *%ptr4
  %val5 = load float, float *%ptr5
  %val6 = load float, float *%ptr6
  %val7 = load float, float *%ptr7
  %val8 = load float, float *%ptr8
  %val9 = load float, float *%ptr9
  %val10 = load float, float *%ptr10

  %ret = call float @foo()

  %cmp0 = fcmp olt float %ret, %val0
  %cmp1 = fcmp olt float %ret, %val1
  %cmp2 = fcmp olt float %ret, %val2
  %cmp3 = fcmp olt float %ret, %val3
  %cmp4 = fcmp olt float %ret, %val4
  %cmp5 = fcmp olt float %ret, %val5
  %cmp6 = fcmp olt float %ret, %val6
  %cmp7 = fcmp olt float %ret, %val7
  %cmp8 = fcmp olt float %ret, %val8
  %cmp9 = fcmp olt float %ret, %val9
  %cmp10 = fcmp olt float %ret, %val10

  %sel0 = select i1 %cmp0, float %ret, float 0.0
  %sel1 = select i1 %cmp1, float %sel0, float 1.0
  %sel2 = select i1 %cmp2, float %sel1, float 2.0
  %sel3 = select i1 %cmp3, float %sel2, float 3.0
  %sel4 = select i1 %cmp4, float %sel3, float 4.0
  %sel5 = select i1 %cmp5, float %sel4, float 5.0
  %sel6 = select i1 %cmp6, float %sel5, float 6.0
  %sel7 = select i1 %cmp7, float %sel6, float 7.0
  %sel8 = select i1 %cmp8, float %sel7, float 8.0
  %sel9 = select i1 %cmp9, float %sel8, float 9.0
  %sel10 = select i1 %cmp10, float %sel9, float 10.0

  ret float %sel10
}

; Check comparison with zero.
define i64 @f8(i64 %a, i64 %b, float %f) {
; CHECK-LABEL: f8:
; CHECK: ltebr %f0, %f0
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq float %f, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the comparison can be reversed if that allows CEB to be used,
; first with oeq.
define i64 @f9(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f9:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then one.
define i64 @f10(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f10:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: blhr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnlh %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp one float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then olt.
define i64 @f11(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f11:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bhr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnh %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp olt float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ole.
define i64 @f12(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f12:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bher %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnhe %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ole float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then oge.
define i64 @f13(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f13:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bler %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnle %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp oge float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ogt.
define i64 @f14(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f14:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: blr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnl %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ogt float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ueq.
define i64 @f15(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f15:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bnlhr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrlh %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ueq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then une.
define i64 @f16(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f16:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bner %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgre %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp une float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ult.
define i64 @f17(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f17:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bnler %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrle %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ult float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ule.
define i64 @f18(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f18:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bnlr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrl %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ule float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then uge.
define i64 @f19(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f19:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bnhr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrh %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp uge float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; ...then ugt.
define i64 @f20(i64 %a, i64 %b, float %f2, float *%ptr) {
; CHECK-LABEL: f20:
; CHECK: ceb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: bnher %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrhe %r2, %r3
; CHECK: br %r14
  %f1 = load float, float *%ptr
  %cond = fcmp ugt float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
