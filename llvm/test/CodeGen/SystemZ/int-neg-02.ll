; Test negative integer absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test i32->i32 negative absolute using slt.
define i32 @f1(i32 %val) {
; CHECK-LABEL: f1:
; CHECK: lnr %r2, %r2
; CHECK: br %r14
  %cmp = icmp slt i32 %val, 0
  %neg = sub i32 0, %val
  %abs = select i1 %cmp, i32 %neg, i32 %val
  %res = sub i32 0, %abs
  ret i32 %res
}

; Test i32->i32 negative absolute using sle.
define i32 @f2(i32 %val) {
; CHECK-LABEL: f2:
; CHECK: lnr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sle i32 %val, 0
  %neg = sub i32 0, %val
  %abs = select i1 %cmp, i32 %neg, i32 %val
  %res = sub i32 0, %abs
  ret i32 %res
}

; Test i32->i32 negative absolute using sgt.
define i32 @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: lnr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sgt i32 %val, 0
  %neg = sub i32 0, %val
  %abs = select i1 %cmp, i32 %val, i32 %neg
  %res = sub i32 0, %abs
  ret i32 %res
}

; Test i32->i32 negative absolute using sge.
define i32 @f4(i32 %val) {
; CHECK-LABEL: f4:
; CHECK: lnr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sge i32 %val, 0
  %neg = sub i32 0, %val
  %abs = select i1 %cmp, i32 %val, i32 %neg
  %res = sub i32 0, %abs
  ret i32 %res
}

; Test i32->i64 negative absolute.
define i64 @f5(i32 %val) {
; CHECK-LABEL: f5:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %val to i64
  %cmp = icmp slt i64 %ext, 0
  %neg = sub i64 0, %ext
  %abs = select i1 %cmp, i64 %neg, i64 %ext
  %res = sub i64 0, %abs
  ret i64 %res
}

; Test i32->i64 negative absolute that uses an "in-register" form of
; sign extension.
define i64 @f6(i64 %val) {
; CHECK-LABEL: f6:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %trunc = trunc i64 %val to i32
  %ext = sext i32 %trunc to i64
  %cmp = icmp slt i64 %ext, 0
  %neg = sub i64 0, %ext
  %abs = select i1 %cmp, i64 %neg, i64 %ext
  %res = sub i64 0, %abs
  ret i64 %res
}

; Test i64 negative absolute.
define i64 @f7(i64 %val) {
; CHECK-LABEL: f7:
; CHECK: lngr %r2, %r2
; CHECK: br %r14
  %cmp = icmp slt i64 %val, 0
  %neg = sub i64 0, %val
  %abs = select i1 %cmp, i64 %neg, i64 %val
  %res = sub i64 0, %abs
  ret i64 %res
}

; Test another form of f6, which is that produced by InstCombine.
define i64 @f8(i64 %val) {
; CHECK-LABEL: f8:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp slt i64 %shl, 0
  %abs = select i1 %cmp, i64 %neg, i64 %ashr
  %res = sub i64 0, %abs
  ret i64 %res
}

; Try again with sle rather than slt.
define i64 @f9(i64 %val) {
; CHECK-LABEL: f9:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sle i64 %shl, 0
  %abs = select i1 %cmp, i64 %neg, i64 %ashr
  %res = sub i64 0, %abs
  ret i64 %res
}

; Repeat f8 with the operands reversed.
define i64 @f10(i64 %val) {
; CHECK-LABEL: f10:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sgt i64 %shl, 0
  %abs = select i1 %cmp, i64 %ashr, i64 %neg
  %res = sub i64 0, %abs
  ret i64 %res
}

; Try again with sge rather than sgt.
define i64 @f11(i64 %val) {
; CHECK-LABEL: f11:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sge i64 %shl, 0
  %abs = select i1 %cmp, i64 %ashr, i64 %neg
  %res = sub i64 0, %abs
  ret i64 %res
}

; Repeat f8 with the negation coming from swapped operands.
define i64 @f12(i64 %val) {
; CHECK-LABEL: f12:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp slt i64 %shl, 0
  %negabs = select i1 %cmp, i64 %ashr, i64 %neg
  ret i64 %negabs
}

; Likewise f9.
define i64 @f13(i64 %val) {
; CHECK-LABEL: f13:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sle i64 %shl, 0
  %negabs = select i1 %cmp, i64 %ashr, i64 %neg
  ret i64 %negabs
}

; Likewise f10.
define i64 @f14(i64 %val) {
; CHECK-LABEL: f14:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sgt i64 %shl, 0
  %negabs = select i1 %cmp, i64 %neg, i64 %ashr
  ret i64 %negabs
}

; Likewise f11.
define i64 @f15(i64 %val) {
; CHECK-LABEL: f15:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %shl = shl i64 %val, 32
  %ashr = ashr i64 %shl, 32
  %neg = sub i64 0, %ashr
  %cmp = icmp sge i64 %shl, 0
  %negabs = select i1 %cmp, i64 %neg, i64 %ashr
  ret i64 %negabs
}

; Repeat f5 with the comparison on the unextended value.
define i64 @f16(i32 %val) {
; CHECK-LABEL: f16:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %val to i64
  %cmp = icmp slt i32 %val, 0
  %neg = sub i64 0, %ext
  %abs = select i1 %cmp, i64 %neg, i64 %ext
  %res = sub i64 0, %abs
  ret i64 %res
}

; And again with the negation coming from swapped operands.
define i64 @f17(i32 %val) {
; CHECK-LABEL: f17:
; CHECK: lngfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %val to i64
  %cmp = icmp slt i32 %val, 0
  %neg = sub i64 0, %ext
  %abs = select i1 %cmp, i64 %ext, i64 %neg
  ret i64 %abs
}
