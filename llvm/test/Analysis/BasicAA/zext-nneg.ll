; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

;; Simple case: a zext nneg can be replaced with a sext. Make sure BasicAA
;; understands that.
define void @t1(i32 %a, i32 %b) {
; CHECK-LABEL: Function: t1
; CHECK: NoAlias: float* %gep1, float* %gep2

  %1 = alloca [8 x float], align 4
  %or1 = or i32 %a, 1
  %2 = sext i32 %or1 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %2

  %shl1 = shl i32 %b, 1
  %3 = zext nneg i32 %shl1 to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %3

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}

;; A (zext nneg (sext V)) is equivalent to a (zext (sext V)) as long as the
;; total number of zext+sext bits is the same for both.
define void @t2(i8 %a, i8 %b) {
; CHECK-LABEL: Function: t2
; CHECK: NoAlias: float* %gep1, float* %gep2
  %1 = alloca [8 x float], align 4
  %or1 = or i8 %a, 1
  %2 = sext i8 %or1 to i32
  %3 = zext i32 %2 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %3

  %shl1 = shl i8 %b, 1
  %4 = sext i8 %shl1 to i16
  %5 = zext nneg i16 %4 to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %5

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}

;; Here the %a and %b are knowably non-equal. In this cases we can distribute
;; the zext, preserving the nneg flag, through the shl because it has a nsw flag
define void @t3(i8 %v) {
; CHECK-LABEL: Function: t3
; CHECK: NoAlias: <2 x float>* %gep1, <2 x float>* %gep2
  %a = or i8 %v, 1
  %b = and i8 %v, 2

  %1 = alloca [8 x float], align 4
  %or1 = shl nuw nsw i8 %a, 1
  %2 = zext nneg i8 %or1 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %2

  %m = mul nsw nuw i8 %b, 2
  %3 = sext i8 %m to i16
  %4 = zext i16 %3 to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %4

  load <2 x float>, ptr %gep1
  load <2 x float>, ptr %gep2
  ret void
}

;; This is the same as above, but this time the shl does not have the nsw flag.
;; the nneg cannot be kept on the zext.
define void @t4(i8 %v) {
; CHECK-LABEL: Function: t4
; CHECK: MayAlias: <2 x float>* %gep1, <2 x float>* %gep2
  %a = or i8 %v, 1
  %b = and i8 %v, 2

  %1 = alloca [8 x float], align 4
  %or1 = shl nuw i8 %a, 1
  %2 = zext nneg i8 %or1 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %2

  %m = mul nsw nuw i8 %b, 2
  %3 = sext i8 %m to i16
  %4 = zext i16 %3 to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %4

  load <2 x float>, ptr %gep1
  load <2 x float>, ptr %gep2
  ret void
}

;; Verify a zext nneg and a zext are understood as the same
define void @t5(ptr %p, i16 %i) {
; CHECK-LABEL: Function: t5
; CHECK: NoAlias: i32* %pi, i32* %pi.next
  %i1 = zext nneg i16 %i to i32
  %pi = getelementptr i32, ptr %p, i32 %i1

  %i.next = add i16 %i, 1
  %i.next2 = zext i16 %i.next to i32
  %pi.next = getelementptr i32, ptr %p, i32 %i.next2

  load i32, ptr %pi
  load i32, ptr %pi.next
  ret void
}

;; This is not very idiomatic, but still possible, verify the nneg is propagated
;; outward. and that no alias is correctly identified.
define void @t6(i8 %a) {
; CHECK-LABEL: Function: t6
; CHECK: NoAlias: float* %gep1, float* %gep2
  %1 = alloca [8 x float], align 4
  %a.add = add i8 %a, 1
  %2 = zext nneg i8 %a.add to i16
  %3 = sext i16 %2 to i32
  %4 = zext i32 %3 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %4

  %5 = sext i8 %a to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %5

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}

;; This is even less idiomatic, but still possible, verify the nneg is not
;; propagated inward. and that may alias is correctly identified.
define void @t7(i8 %a) {
; CHECK-LABEL: Function: t7
; CHECK: MayAlias: float* %gep1, float* %gep2
  %1 = alloca [8 x float], align 4
  %a.add = add i8 %a, 1
  %2 = zext i8 %a.add to i16
  %3 = sext i16 %2 to i32
  %4 = zext nneg i32 %3 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %4

  %5 = sext i8 %a to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %5

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}

;; Verify the nneg survives an implicit trunc of fewer bits then the zext.
define void @t8(i8 %a) {
; CHECK-LABEL: Function: t8
; CHECK: NoAlias: float* %gep1, float* %gep2
  %1 = alloca [8 x float], align 4
  %a.add = add i8 %a, 1
  %2 = zext nneg i8 %a.add to i128
  %gep1 = getelementptr inbounds float, ptr %1, i128 %2

  %3 = sext i8 %a to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %3

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}

;; Ensure that the nneg is never propagated past this trunc and that these
;; casted values are understood as non-equal.
define void @t9(i8 %a) {
; CHECK-LABEL: Function: t9
; CHECK: MayAlias: float* %gep1, float* %gep2
  %1 = alloca [8 x float], align 4
  %a.add = add i8 %a, 1
  %2 = zext i8 %a.add to i16
  %3 = trunc i16 %2 to i1
  %4 = zext nneg i1 %3 to i64
  %gep1 = getelementptr inbounds float, ptr %1, i64 %4

  %5 = sext i8 %a to i64
  %gep2 = getelementptr inbounds float, ptr %1, i64 %5

  load float, ptr %gep1
  load float, ptr %gep2
  ret void
}
