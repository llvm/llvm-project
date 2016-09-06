; RUN: llc                               -aarch64-enable-atomic-cfg-tidy=0 -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -fast-isel -fast-isel-abort=1 -aarch64-enable-atomic-cfg-tidy=0 -mtriple=aarch64-apple-darwin < %s | FileCheck %s

define i32 @fcmp_oeq(float %x, float %y) {
; CHECK-LABEL: fcmp_oeq
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.ne {{LBB.+_2}}
  %1 = fcmp oeq float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ogt(float %x, float %y) {
; CHECK-LABEL: fcmp_ogt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.le {{LBB.+_2}}
  %1 = fcmp ogt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_oge(float %x, float %y) {
; CHECK-LABEL: fcmp_oge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.lt {{LBB.+_2}}
  %1 = fcmp oge float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_olt(float %x, float %y) {
; CHECK-LABEL: fcmp_olt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.pl {{LBB.+_2}}
  %1 = fcmp olt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ole(float %x, float %y) {
; CHECK-LABEL: fcmp_ole
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.hi {{LBB.+_2}}
  %1 = fcmp ole float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_one(float %x, float %y) {
; CHECK-LABEL: fcmp_one
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.mi
; CHECK-NEXT:  b.gt
  %1 = fcmp one float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ord(float %x, float %y) {
; CHECK-LABEL: fcmp_ord
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.vs {{LBB.+_2}}
  %1 = fcmp ord float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uno(float %x, float %y) {
; CHECK-LABEL: fcmp_uno
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.vs {{LBB.+_2}}
  %1 = fcmp uno float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ueq(float %x, float %y) {
; CHECK-LABEL: fcmp_ueq
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.eq {{LBB.+_2}}
; CHECK-NEXT:  b.vs {{LBB.+_2}}
  %1 = fcmp ueq float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ugt(float %x, float %y) {
; CHECK-LABEL: fcmp_ugt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.ls {{LBB.+_2}}
  %1 = fcmp ugt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uge(float %x, float %y) {
; CHECK-LABEL: fcmp_uge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.mi {{LBB.+_2}}
  %1 = fcmp uge float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ult(float %x, float %y) {
; CHECK-LABEL: fcmp_ult
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.ge {{LBB.+_2}}
  %1 = fcmp ult float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ule(float %x, float %y) {
; CHECK-LABEL: fcmp_ule
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.gt {{LBB.+_2}}
  %1 = fcmp ule float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_une(float %x, float %y) {
; CHECK-LABEL: fcmp_une
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  b.eq {{LBB.+_2}}
  %1 = fcmp une float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_eq
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.ne {{LBB.+_2}}
  %1 = icmp eq i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ne(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ne
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.eq {{LBB.+_2}}
  %1 = icmp ne i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ugt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ugt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.ls {{LBB.+_2}}
  %1 = icmp ugt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_uge(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_uge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.lo {{LBB.+_2}}
  %1 = icmp uge i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ult(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ult
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.hs {{LBB.+_2}}
  %1 = icmp ult i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ule(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ule
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.hi {{LBB.+_2}}
  %1 = icmp ule i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sgt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sgt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.le {{LBB.+_2}}
  %1 = icmp sgt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sge(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.lt {{LBB.+_2}}
  %1 = icmp sge i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_slt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_slt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.ge {{LBB.+_2}}
  %1 = icmp slt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sle(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sle
; CHECK:       cmp w0, w1
; CHECK-NEXT:  b.gt {{LBB.+_2}}
  %1 = icmp sle i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

