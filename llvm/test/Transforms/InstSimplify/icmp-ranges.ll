; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

; Cycle through all pairs of predicates to test
; simplification of range-intersection or range-union.

; eq
; x == 13 && x == 17

define i1 @and_eq_eq(i8 %x) {
; CHECK-LABEL: @and_eq_eq(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x != 17

define i1 @and_eq_ne(i8 %x) {
; CHECK-LABEL: @and_eq_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x >=s 17

define i1 @and_eq_sge(i8 %x) {
; CHECK-LABEL: @and_eq_sge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x >s 17

define i1 @and_eq_sgt(i8 %x) {
; CHECK-LABEL: @and_eq_sgt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x <=s 17

define i1 @and_eq_sle(i8 %x) {
; CHECK-LABEL: @and_eq_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x <s 17

define i1 @and_eq_slt(i8 %x) {
; CHECK-LABEL: @and_eq_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x >=u 17

define i1 @and_eq_uge(i8 %x) {
; CHECK-LABEL: @and_eq_uge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x >u 17

define i1 @and_eq_ugt(i8 %x) {
; CHECK-LABEL: @and_eq_ugt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x <=u 17

define i1 @and_eq_ule(i8 %x) {
; CHECK-LABEL: @and_eq_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 13 && x <u 17

define i1 @and_eq_ult(i8 %x) {
; CHECK-LABEL: @and_eq_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ne
; x != 13 && x == 17

define i1 @and_ne_eq(i8 %x) {
; CHECK-LABEL: @and_ne_eq(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x != 17

define i1 @and_ne_ne(i8 %x) {
; CHECK-LABEL: @and_ne_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x >=s 17

define i1 @and_ne_sge(i8 %x) {
; CHECK-LABEL: @and_ne_sge(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x >s 17

define i1 @and_ne_sgt(i8 %x) {
; CHECK-LABEL: @and_ne_sgt(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x <=s 17

define i1 @and_ne_sle(i8 %x) {
; CHECK-LABEL: @and_ne_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x <s 17

define i1 @and_ne_slt(i8 %x) {
; CHECK-LABEL: @and_ne_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x >=u 17

define i1 @and_ne_uge(i8 %x) {
; CHECK-LABEL: @and_ne_uge(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x >u 17

define i1 @and_ne_ugt(i8 %x) {
; CHECK-LABEL: @and_ne_ugt(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x <=u 17

define i1 @and_ne_ule(i8 %x) {
; CHECK-LABEL: @and_ne_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 13 && x <u 17

define i1 @and_ne_ult(i8 %x) {
; CHECK-LABEL: @and_ne_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sge
; x >=s 13 && x == 17

define i1 @and_sge_eq(i8 %x) {
; CHECK-LABEL: @and_sge_eq(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x != 17

define i1 @and_sge_ne(i8 %x) {
; CHECK-LABEL: @and_sge_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x >=s 17

define i1 @and_sge_sge(i8 %x) {
; CHECK-LABEL: @and_sge_sge(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x >s 17

define i1 @and_sge_sgt(i8 %x) {
; CHECK-LABEL: @and_sge_sgt(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x <=s 17

define i1 @and_sge_sle(i8 %x) {
; CHECK-LABEL: @and_sge_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x <s 17

define i1 @and_sge_slt(i8 %x) {
; CHECK-LABEL: @and_sge_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x >=u 17

define i1 @and_sge_uge(i8 %x) {
; CHECK-LABEL: @and_sge_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x >u 17

define i1 @and_sge_ugt(i8 %x) {
; CHECK-LABEL: @and_sge_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x <=u 17

define i1 @and_sge_ule(i8 %x) {
; CHECK-LABEL: @and_sge_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 13 && x <u 17

define i1 @and_sge_ult(i8 %x) {
; CHECK-LABEL: @and_sge_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sgt
; x >s 13 && x == 17

define i1 @and_sgt_eq(i8 %x) {
; CHECK-LABEL: @and_sgt_eq(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x != 17

define i1 @and_sgt_ne(i8 %x) {
; CHECK-LABEL: @and_sgt_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x >=s 17

define i1 @and_sgt_sge(i8 %x) {
; CHECK-LABEL: @and_sgt_sge(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x >s 17

define i1 @and_sgt_sgt(i8 %x) {
; CHECK-LABEL: @and_sgt_sgt(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x <=s 17

define i1 @and_sgt_sle(i8 %x) {
; CHECK-LABEL: @and_sgt_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x <s 17

define i1 @and_sgt_slt(i8 %x) {
; CHECK-LABEL: @and_sgt_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x >=u 17

define i1 @and_sgt_uge(i8 %x) {
; CHECK-LABEL: @and_sgt_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x >u 17

define i1 @and_sgt_ugt(i8 %x) {
; CHECK-LABEL: @and_sgt_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x <=u 17

define i1 @and_sgt_ule(i8 %x) {
; CHECK-LABEL: @and_sgt_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 13 && x <u 17

define i1 @and_sgt_ult(i8 %x) {
; CHECK-LABEL: @and_sgt_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sle
; x <=s 13 && x == 17

define i1 @and_sle_eq(i8 %x) {
; CHECK-LABEL: @and_sle_eq(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sle i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x != 17

define i1 @and_sle_ne(i8 %x) {
; CHECK-LABEL: @and_sle_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x >=s 17

define i1 @and_sle_sge(i8 %x) {
; CHECK-LABEL: @and_sle_sge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sle i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x >s 17

define i1 @and_sle_sgt(i8 %x) {
; CHECK-LABEL: @and_sle_sgt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sle i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x <=s 17

define i1 @and_sle_sle(i8 %x) {
; CHECK-LABEL: @and_sle_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x <s 17

define i1 @and_sle_slt(i8 %x) {
; CHECK-LABEL: @and_sle_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x >=u 17

define i1 @and_sle_uge(i8 %x) {
; CHECK-LABEL: @and_sle_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x >u 17

define i1 @and_sle_ugt(i8 %x) {
; CHECK-LABEL: @and_sle_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x <=u 17

define i1 @and_sle_ule(i8 %x) {
; CHECK-LABEL: @and_sle_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 13 && x <u 17

define i1 @and_sle_ult(i8 %x) {
; CHECK-LABEL: @and_sle_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; slt
; x <s 13 && x == 17

define i1 @and_slt_eq(i8 %x) {
; CHECK-LABEL: @and_slt_eq(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp slt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x != 17

define i1 @and_slt_ne(i8 %x) {
; CHECK-LABEL: @and_slt_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x >=s 17

define i1 @and_slt_sge(i8 %x) {
; CHECK-LABEL: @and_slt_sge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp slt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x >s 17

define i1 @and_slt_sgt(i8 %x) {
; CHECK-LABEL: @and_slt_sgt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp slt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x <=s 17

define i1 @and_slt_sle(i8 %x) {
; CHECK-LABEL: @and_slt_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x <s 17

define i1 @and_slt_slt(i8 %x) {
; CHECK-LABEL: @and_slt_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x >=u 17

define i1 @and_slt_uge(i8 %x) {
; CHECK-LABEL: @and_slt_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x >u 17

define i1 @and_slt_ugt(i8 %x) {
; CHECK-LABEL: @and_slt_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x <=u 17

define i1 @and_slt_ule(i8 %x) {
; CHECK-LABEL: @and_slt_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 13 && x <u 17

define i1 @and_slt_ult(i8 %x) {
; CHECK-LABEL: @and_slt_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; uge
; x >=u 13 && x == 17

define i1 @and_uge_eq(i8 %x) {
; CHECK-LABEL: @and_uge_eq(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x != 17

define i1 @and_uge_ne(i8 %x) {
; CHECK-LABEL: @and_uge_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x >=s 17

define i1 @and_uge_sge(i8 %x) {
; CHECK-LABEL: @and_uge_sge(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x >s 17

define i1 @and_uge_sgt(i8 %x) {
; CHECK-LABEL: @and_uge_sgt(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x <=s 17

define i1 @and_uge_sle(i8 %x) {
; CHECK-LABEL: @and_uge_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x <s 17

define i1 @and_uge_slt(i8 %x) {
; CHECK-LABEL: @and_uge_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x >=u 17

define i1 @and_uge_uge(i8 %x) {
; CHECK-LABEL: @and_uge_uge(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x >u 17

define i1 @and_uge_ugt(i8 %x) {
; CHECK-LABEL: @and_uge_ugt(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x <=u 17

define i1 @and_uge_ule(i8 %x) {
; CHECK-LABEL: @and_uge_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 13 && x <u 17

define i1 @and_uge_ult(i8 %x) {
; CHECK-LABEL: @and_uge_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ugt
; x >u 13 && x == 17

define i1 @and_ugt_eq(i8 %x) {
; CHECK-LABEL: @and_ugt_eq(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x != 17

define i1 @and_ugt_ne(i8 %x) {
; CHECK-LABEL: @and_ugt_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x >=s 17

define i1 @and_ugt_sge(i8 %x) {
; CHECK-LABEL: @and_ugt_sge(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x >s 17

define i1 @and_ugt_sgt(i8 %x) {
; CHECK-LABEL: @and_ugt_sgt(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x <=s 17

define i1 @and_ugt_sle(i8 %x) {
; CHECK-LABEL: @and_ugt_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x <s 17

define i1 @and_ugt_slt(i8 %x) {
; CHECK-LABEL: @and_ugt_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x >=u 17

define i1 @and_ugt_uge(i8 %x) {
; CHECK-LABEL: @and_ugt_uge(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x >u 17

define i1 @and_ugt_ugt(i8 %x) {
; CHECK-LABEL: @and_ugt_ugt(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x <=u 17

define i1 @and_ugt_ule(i8 %x) {
; CHECK-LABEL: @and_ugt_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 13 && x <u 17

define i1 @and_ugt_ult(i8 %x) {
; CHECK-LABEL: @and_ugt_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ule
; x <=u 13 && x == 17

define i1 @and_ule_eq(i8 %x) {
; CHECK-LABEL: @and_ule_eq(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ule i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x != 17

define i1 @and_ule_ne(i8 %x) {
; CHECK-LABEL: @and_ule_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x >=s 17

define i1 @and_ule_sge(i8 %x) {
; CHECK-LABEL: @and_ule_sge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ule i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x >s 17

define i1 @and_ule_sgt(i8 %x) {
; CHECK-LABEL: @and_ule_sgt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ule i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x <=s 17

define i1 @and_ule_sle(i8 %x) {
; CHECK-LABEL: @and_ule_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x <s 17

define i1 @and_ule_slt(i8 %x) {
; CHECK-LABEL: @and_ule_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x >=u 17

define i1 @and_ule_uge(i8 %x) {
; CHECK-LABEL: @and_ule_uge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ule i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x >u 17

define i1 @and_ule_ugt(i8 %x) {
; CHECK-LABEL: @and_ule_ugt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ule i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x <=u 17

define i1 @and_ule_ule(i8 %x) {
; CHECK-LABEL: @and_ule_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 13 && x <u 17

define i1 @and_ule_ult(i8 %x) {
; CHECK-LABEL: @and_ule_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ult
; x <u 13 && x == 17

define i1 @and_ult_eq(i8 %x) {
; CHECK-LABEL: @and_ult_eq(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x != 17

define i1 @and_ult_ne(i8 %x) {
; CHECK-LABEL: @and_ult_ne(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x >=s 17

define i1 @and_ult_sge(i8 %x) {
; CHECK-LABEL: @and_ult_sge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x >s 17

define i1 @and_ult_sgt(i8 %x) {
; CHECK-LABEL: @and_ult_sgt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x <=s 17

define i1 @and_ult_sle(i8 %x) {
; CHECK-LABEL: @and_ult_sle(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x <s 17

define i1 @and_ult_slt(i8 %x) {
; CHECK-LABEL: @and_ult_slt(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x >=u 17

define i1 @and_ult_uge(i8 %x) {
; CHECK-LABEL: @and_ult_uge(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x >u 17

define i1 @and_ult_ugt(i8 %x) {
; CHECK-LABEL: @and_ult_ugt(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x <=u 17

define i1 @and_ult_ule(i8 %x) {
; CHECK-LABEL: @and_ult_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 13 && x <u 17

define i1 @and_ult_ult(i8 %x) {
; CHECK-LABEL: @and_ult_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; eq
; x == 23 && x == 17

define i1 @and_eq_eq_swap(i8 %x) {
; CHECK-LABEL: @and_eq_eq_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x != 17

define i1 @and_eq_ne_swap(i8 %x) {
; CHECK-LABEL: @and_eq_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x >=s 17

define i1 @and_eq_sge_swap(i8 %x) {
; CHECK-LABEL: @and_eq_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x >s 17

define i1 @and_eq_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_eq_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x <=s 17

define i1 @and_eq_sle_swap(i8 %x) {
; CHECK-LABEL: @and_eq_sle_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x <s 17

define i1 @and_eq_slt_swap(i8 %x) {
; CHECK-LABEL: @and_eq_slt_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x >=u 17

define i1 @and_eq_uge_swap(i8 %x) {
; CHECK-LABEL: @and_eq_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x >u 17

define i1 @and_eq_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_eq_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x <=u 17

define i1 @and_eq_ule_swap(i8 %x) {
; CHECK-LABEL: @and_eq_ule_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x == 23 && x <u 17

define i1 @and_eq_ult_swap(i8 %x) {
; CHECK-LABEL: @and_eq_ult_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp eq i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ne
; x != 23 && x == 17

define i1 @and_ne_eq_swap(i8 %x) {
; CHECK-LABEL: @and_ne_eq_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x != 17

define i1 @and_ne_ne_swap(i8 %x) {
; CHECK-LABEL: @and_ne_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x >=s 17

define i1 @and_ne_sge_swap(i8 %x) {
; CHECK-LABEL: @and_ne_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x >s 17

define i1 @and_ne_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_ne_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x <=s 17

define i1 @and_ne_sle_swap(i8 %x) {
; CHECK-LABEL: @and_ne_sle_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x <s 17

define i1 @and_ne_slt_swap(i8 %x) {
; CHECK-LABEL: @and_ne_slt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x >=u 17

define i1 @and_ne_uge_swap(i8 %x) {
; CHECK-LABEL: @and_ne_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x >u 17

define i1 @and_ne_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_ne_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x <=u 17

define i1 @and_ne_ule_swap(i8 %x) {
; CHECK-LABEL: @and_ne_ule_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x != 23 && x <u 17

define i1 @and_ne_ult_swap(i8 %x) {
; CHECK-LABEL: @and_ne_ult_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sge
; x >=s 23 && x == 17

define i1 @and_sge_eq_swap(i8 %x) {
; CHECK-LABEL: @and_sge_eq_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sge i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x != 17

define i1 @and_sge_ne_swap(i8 %x) {
; CHECK-LABEL: @and_sge_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x >=s 17

define i1 @and_sge_sge_swap(i8 %x) {
; CHECK-LABEL: @and_sge_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x >s 17

define i1 @and_sge_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_sge_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x <=s 17

define i1 @and_sge_sle_swap(i8 %x) {
; CHECK-LABEL: @and_sge_sle_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sge i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x <s 17

define i1 @and_sge_slt_swap(i8 %x) {
; CHECK-LABEL: @and_sge_slt_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sge i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x >=u 17

define i1 @and_sge_uge_swap(i8 %x) {
; CHECK-LABEL: @and_sge_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x >u 17

define i1 @and_sge_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_sge_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x <=u 17

define i1 @and_sge_ule_swap(i8 %x) {
; CHECK-LABEL: @and_sge_ule_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sge i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=s 23 && x <u 17

define i1 @and_sge_ult_swap(i8 %x) {
; CHECK-LABEL: @and_sge_ult_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sge i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sgt
; x >s 23 && x == 17

define i1 @and_sgt_eq_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_eq_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sgt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x != 17

define i1 @and_sgt_ne_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x >=s 17

define i1 @and_sgt_sge_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x >s 17

define i1 @and_sgt_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x <=s 17

define i1 @and_sgt_sle_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_sle_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x <s 17

define i1 @and_sgt_slt_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_slt_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sgt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x >=u 17

define i1 @and_sgt_uge_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x >u 17

define i1 @and_sgt_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x <=u 17

define i1 @and_sgt_ule_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_ule_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >s 23 && x <u 17

define i1 @and_sgt_ult_swap(i8 %x) {
; CHECK-LABEL: @and_sgt_ult_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; sle
; x <=s 23 && x == 17

define i1 @and_sle_eq_swap(i8 %x) {
; CHECK-LABEL: @and_sle_eq_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x != 17

define i1 @and_sle_ne_swap(i8 %x) {
; CHECK-LABEL: @and_sle_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x >=s 17

define i1 @and_sle_sge_swap(i8 %x) {
; CHECK-LABEL: @and_sle_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x >s 17

define i1 @and_sle_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_sle_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x <=s 17

define i1 @and_sle_sle_swap(i8 %x) {
; CHECK-LABEL: @and_sle_sle_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x <s 17

define i1 @and_sle_slt_swap(i8 %x) {
; CHECK-LABEL: @and_sle_slt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x >=u 17

define i1 @and_sle_uge_swap(i8 %x) {
; CHECK-LABEL: @and_sle_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x >u 17

define i1 @and_sle_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_sle_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x <=u 17

define i1 @and_sle_ule_swap(i8 %x) {
; CHECK-LABEL: @and_sle_ule_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=s 23 && x <u 17

define i1 @and_sle_ult_swap(i8 %x) {
; CHECK-LABEL: @and_sle_ult_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; slt
; x <s 23 && x == 17

define i1 @and_slt_eq_swap(i8 %x) {
; CHECK-LABEL: @and_slt_eq_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x != 17

define i1 @and_slt_ne_swap(i8 %x) {
; CHECK-LABEL: @and_slt_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x >=s 17

define i1 @and_slt_sge_swap(i8 %x) {
; CHECK-LABEL: @and_slt_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x >s 17

define i1 @and_slt_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_slt_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x <=s 17

define i1 @and_slt_sle_swap(i8 %x) {
; CHECK-LABEL: @and_slt_sle_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x <s 17

define i1 @and_slt_slt_swap(i8 %x) {
; CHECK-LABEL: @and_slt_slt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x >=u 17

define i1 @and_slt_uge_swap(i8 %x) {
; CHECK-LABEL: @and_slt_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x >u 17

define i1 @and_slt_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_slt_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x <=u 17

define i1 @and_slt_ule_swap(i8 %x) {
; CHECK-LABEL: @and_slt_ule_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <s 23 && x <u 17

define i1 @and_slt_ult_swap(i8 %x) {
; CHECK-LABEL: @and_slt_ult_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; uge
; x >=u 23 && x == 17

define i1 @and_uge_eq_swap(i8 %x) {
; CHECK-LABEL: @and_uge_eq_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp uge i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x != 17

define i1 @and_uge_ne_swap(i8 %x) {
; CHECK-LABEL: @and_uge_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x >=s 17

define i1 @and_uge_sge_swap(i8 %x) {
; CHECK-LABEL: @and_uge_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x >s 17

define i1 @and_uge_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_uge_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x <=s 17

define i1 @and_uge_sle_swap(i8 %x) {
; CHECK-LABEL: @and_uge_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x <s 17

define i1 @and_uge_slt_swap(i8 %x) {
; CHECK-LABEL: @and_uge_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x >=u 17

define i1 @and_uge_uge_swap(i8 %x) {
; CHECK-LABEL: @and_uge_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x >u 17

define i1 @and_uge_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_uge_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x <=u 17

define i1 @and_uge_ule_swap(i8 %x) {
; CHECK-LABEL: @and_uge_ule_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp uge i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >=u 23 && x <u 17

define i1 @and_uge_ult_swap(i8 %x) {
; CHECK-LABEL: @and_uge_ult_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp uge i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ugt
; x >u 23 && x == 17

define i1 @and_ugt_eq_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_eq_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ugt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x != 17

define i1 @and_ugt_ne_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x >=s 17

define i1 @and_ugt_sge_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x >s 17

define i1 @and_ugt_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x <=s 17

define i1 @and_ugt_sle_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x <s 17

define i1 @and_ugt_slt_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x >=u 17

define i1 @and_ugt_uge_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x >u 17

define i1 @and_ugt_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x <=u 17

define i1 @and_ugt_ule_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_ule_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x >u 23 && x <u 17

define i1 @and_ugt_ult_swap(i8 %x) {
; CHECK-LABEL: @and_ugt_ult_swap(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ule
; x <=u 23 && x == 17

define i1 @and_ule_eq_swap(i8 %x) {
; CHECK-LABEL: @and_ule_eq_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x != 17

define i1 @and_ule_ne_swap(i8 %x) {
; CHECK-LABEL: @and_ule_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x >=s 17

define i1 @and_ule_sge_swap(i8 %x) {
; CHECK-LABEL: @and_ule_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x >s 17

define i1 @and_ule_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_ule_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x <=s 17

define i1 @and_ule_sle_swap(i8 %x) {
; CHECK-LABEL: @and_ule_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x <s 17

define i1 @and_ule_slt_swap(i8 %x) {
; CHECK-LABEL: @and_ule_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x >=u 17

define i1 @and_ule_uge_swap(i8 %x) {
; CHECK-LABEL: @and_ule_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x >u 17

define i1 @and_ule_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_ule_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x <=u 17

define i1 @and_ule_ule_swap(i8 %x) {
; CHECK-LABEL: @and_ule_ule_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <=u 23 && x <u 17

define i1 @and_ule_ult_swap(i8 %x) {
; CHECK-LABEL: @and_ule_ult_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; ult
; x <u 23 && x == 17

define i1 @and_ult_eq_swap(i8 %x) {
; CHECK-LABEL: @and_ult_eq_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x != 17

define i1 @and_ult_ne_swap(i8 %x) {
; CHECK-LABEL: @and_ult_ne_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x >=s 17

define i1 @and_ult_sge_swap(i8 %x) {
; CHECK-LABEL: @and_ult_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x >s 17

define i1 @and_ult_sgt_swap(i8 %x) {
; CHECK-LABEL: @and_ult_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x <=s 17

define i1 @and_ult_sle_swap(i8 %x) {
; CHECK-LABEL: @and_ult_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x <s 17

define i1 @and_ult_slt_swap(i8 %x) {
; CHECK-LABEL: @and_ult_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x >=u 17

define i1 @and_ult_uge_swap(i8 %x) {
; CHECK-LABEL: @and_ult_uge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x >u 17

define i1 @and_ult_ugt_swap(i8 %x) {
; CHECK-LABEL: @and_ult_ugt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = and i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x <=u 17

define i1 @and_ult_ule_swap(i8 %x) {
; CHECK-LABEL: @and_ult_ule_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; x <u 23 && x <u 17

define i1 @and_ult_ult_swap(i8 %x) {
; CHECK-LABEL: @and_ult_ult_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = and i1 %a, %b
  ret i1 %c
}

; eq
; x == 13 || x == 17

define i1 @or_eq_eq(i8 %x) {
; CHECK-LABEL: @or_eq_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x != 17

define i1 @or_eq_ne(i8 %x) {
; CHECK-LABEL: @or_eq_ne(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x >=s 17

define i1 @or_eq_sge(i8 %x) {
; CHECK-LABEL: @or_eq_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x >s 17

define i1 @or_eq_sgt(i8 %x) {
; CHECK-LABEL: @or_eq_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x <=s 17

define i1 @or_eq_sle(i8 %x) {
; CHECK-LABEL: @or_eq_sle(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x <s 17

define i1 @or_eq_slt(i8 %x) {
; CHECK-LABEL: @or_eq_slt(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x >=u 17

define i1 @or_eq_uge(i8 %x) {
; CHECK-LABEL: @or_eq_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x >u 17

define i1 @or_eq_ugt(i8 %x) {
; CHECK-LABEL: @or_eq_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x <=u 17

define i1 @or_eq_ule(i8 %x) {
; CHECK-LABEL: @or_eq_ule(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 13 || x <u 17

define i1 @or_eq_ult(i8 %x) {
; CHECK-LABEL: @or_eq_ult(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ne
; x != 13 || x == 17

define i1 @or_ne_eq(i8 %x) {
; CHECK-LABEL: @or_ne_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x != 17

define i1 @or_ne_ne(i8 %x) {
; CHECK-LABEL: @or_ne_ne(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x >=s 17

define i1 @or_ne_sge(i8 %x) {
; CHECK-LABEL: @or_ne_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x >s 17

define i1 @or_ne_sgt(i8 %x) {
; CHECK-LABEL: @or_ne_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x <=s 17

define i1 @or_ne_sle(i8 %x) {
; CHECK-LABEL: @or_ne_sle(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x <s 17

define i1 @or_ne_slt(i8 %x) {
; CHECK-LABEL: @or_ne_slt(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x >=u 17

define i1 @or_ne_uge(i8 %x) {
; CHECK-LABEL: @or_ne_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x >u 17

define i1 @or_ne_ugt(i8 %x) {
; CHECK-LABEL: @or_ne_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x <=u 17

define i1 @or_ne_ule(i8 %x) {
; CHECK-LABEL: @or_ne_ule(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 13 || x <u 17

define i1 @or_ne_ult(i8 %x) {
; CHECK-LABEL: @or_ne_ult(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sge
; x >=s 13 || x == 17

define i1 @or_sge_eq(i8 %x) {
; CHECK-LABEL: @or_sge_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x != 17

define i1 @or_sge_ne(i8 %x) {
; CHECK-LABEL: @or_sge_ne(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sge i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x >=s 17

define i1 @or_sge_sge(i8 %x) {
; CHECK-LABEL: @or_sge_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x >s 17

define i1 @or_sge_sgt(i8 %x) {
; CHECK-LABEL: @or_sge_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x <=s 17

define i1 @or_sge_sle(i8 %x) {
; CHECK-LABEL: @or_sge_sle(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sge i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x <s 17

define i1 @or_sge_slt(i8 %x) {
; CHECK-LABEL: @or_sge_slt(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sge i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x >=u 17

define i1 @or_sge_uge(i8 %x) {
; CHECK-LABEL: @or_sge_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x >u 17

define i1 @or_sge_ugt(i8 %x) {
; CHECK-LABEL: @or_sge_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x <=u 17

define i1 @or_sge_ule(i8 %x) {
; CHECK-LABEL: @or_sge_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 13 || x <u 17

define i1 @or_sge_ult(i8 %x) {
; CHECK-LABEL: @or_sge_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sgt
; x >s 13 || x == 17

define i1 @or_sgt_eq(i8 %x) {
; CHECK-LABEL: @or_sgt_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x != 17

define i1 @or_sgt_ne(i8 %x) {
; CHECK-LABEL: @or_sgt_ne(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x >=s 17

define i1 @or_sgt_sge(i8 %x) {
; CHECK-LABEL: @or_sgt_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x >s 17

define i1 @or_sgt_sgt(i8 %x) {
; CHECK-LABEL: @or_sgt_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x <=s 17

define i1 @or_sgt_sle(i8 %x) {
; CHECK-LABEL: @or_sgt_sle(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sgt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x <s 17

define i1 @or_sgt_slt(i8 %x) {
; CHECK-LABEL: @or_sgt_slt(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sgt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x >=u 17

define i1 @or_sgt_uge(i8 %x) {
; CHECK-LABEL: @or_sgt_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x >u 17

define i1 @or_sgt_ugt(i8 %x) {
; CHECK-LABEL: @or_sgt_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x <=u 17

define i1 @or_sgt_ule(i8 %x) {
; CHECK-LABEL: @or_sgt_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 13 || x <u 17

define i1 @or_sgt_ult(i8 %x) {
; CHECK-LABEL: @or_sgt_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sle
; x <=s 13 || x == 17

define i1 @or_sle_eq(i8 %x) {
; CHECK-LABEL: @or_sle_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x != 17

define i1 @or_sle_ne(i8 %x) {
; CHECK-LABEL: @or_sle_ne(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x >=s 17

define i1 @or_sle_sge(i8 %x) {
; CHECK-LABEL: @or_sle_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x >s 17

define i1 @or_sle_sgt(i8 %x) {
; CHECK-LABEL: @or_sle_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x <=s 17

define i1 @or_sle_sle(i8 %x) {
; CHECK-LABEL: @or_sle_sle(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x <s 17

define i1 @or_sle_slt(i8 %x) {
; CHECK-LABEL: @or_sle_slt(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x >=u 17

define i1 @or_sle_uge(i8 %x) {
; CHECK-LABEL: @or_sle_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x >u 17

define i1 @or_sle_ugt(i8 %x) {
; CHECK-LABEL: @or_sle_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x <=u 17

define i1 @or_sle_ule(i8 %x) {
; CHECK-LABEL: @or_sle_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 13 || x <u 17

define i1 @or_sle_ult(i8 %x) {
; CHECK-LABEL: @or_sle_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sle i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; slt
; x <s 13 || x == 17

define i1 @or_slt_eq(i8 %x) {
; CHECK-LABEL: @or_slt_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x != 17

define i1 @or_slt_ne(i8 %x) {
; CHECK-LABEL: @or_slt_ne(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x >=s 17

define i1 @or_slt_sge(i8 %x) {
; CHECK-LABEL: @or_slt_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x >s 17

define i1 @or_slt_sgt(i8 %x) {
; CHECK-LABEL: @or_slt_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x <=s 17

define i1 @or_slt_sle(i8 %x) {
; CHECK-LABEL: @or_slt_sle(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x <s 17

define i1 @or_slt_slt(i8 %x) {
; CHECK-LABEL: @or_slt_slt(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x >=u 17

define i1 @or_slt_uge(i8 %x) {
; CHECK-LABEL: @or_slt_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x >u 17

define i1 @or_slt_ugt(i8 %x) {
; CHECK-LABEL: @or_slt_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x <=u 17

define i1 @or_slt_ule(i8 %x) {
; CHECK-LABEL: @or_slt_ule(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 13 || x <u 17

define i1 @or_slt_ult(i8 %x) {
; CHECK-LABEL: @or_slt_ult(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp slt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; uge
; x >=u 13 || x == 17

define i1 @or_uge_eq(i8 %x) {
; CHECK-LABEL: @or_uge_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x != 17

define i1 @or_uge_ne(i8 %x) {
; CHECK-LABEL: @or_uge_ne(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp uge i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x >=s 17

define i1 @or_uge_sge(i8 %x) {
; CHECK-LABEL: @or_uge_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x >s 17

define i1 @or_uge_sgt(i8 %x) {
; CHECK-LABEL: @or_uge_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x <=s 17

define i1 @or_uge_sle(i8 %x) {
; CHECK-LABEL: @or_uge_sle(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp uge i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x <s 17

define i1 @or_uge_slt(i8 %x) {
; CHECK-LABEL: @or_uge_slt(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp uge i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x >=u 17

define i1 @or_uge_uge(i8 %x) {
; CHECK-LABEL: @or_uge_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x >u 17

define i1 @or_uge_ugt(i8 %x) {
; CHECK-LABEL: @or_uge_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp uge i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x <=u 17

define i1 @or_uge_ule(i8 %x) {
; CHECK-LABEL: @or_uge_ule(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp uge i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 13 || x <u 17

define i1 @or_uge_ult(i8 %x) {
; CHECK-LABEL: @or_uge_ult(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp uge i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ugt
; x >u 13 || x == 17

define i1 @or_ugt_eq(i8 %x) {
; CHECK-LABEL: @or_ugt_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x != 17

define i1 @or_ugt_ne(i8 %x) {
; CHECK-LABEL: @or_ugt_ne(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x >=s 17

define i1 @or_ugt_sge(i8 %x) {
; CHECK-LABEL: @or_ugt_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x >s 17

define i1 @or_ugt_sgt(i8 %x) {
; CHECK-LABEL: @or_ugt_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x <=s 17

define i1 @or_ugt_sle(i8 %x) {
; CHECK-LABEL: @or_ugt_sle(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ugt i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x <s 17

define i1 @or_ugt_slt(i8 %x) {
; CHECK-LABEL: @or_ugt_slt(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ugt i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x >=u 17

define i1 @or_ugt_uge(i8 %x) {
; CHECK-LABEL: @or_ugt_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x >u 17

define i1 @or_ugt_ugt(i8 %x) {
; CHECK-LABEL: @or_ugt_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 13
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x <=u 17

define i1 @or_ugt_ule(i8 %x) {
; CHECK-LABEL: @or_ugt_ule(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 13 || x <u 17

define i1 @or_ugt_ult(i8 %x) {
; CHECK-LABEL: @or_ugt_ult(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ugt i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ule
; x <=u 13 || x == 17

define i1 @or_ule_eq(i8 %x) {
; CHECK-LABEL: @or_ule_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x != 17

define i1 @or_ule_ne(i8 %x) {
; CHECK-LABEL: @or_ule_ne(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x >=s 17

define i1 @or_ule_sge(i8 %x) {
; CHECK-LABEL: @or_ule_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x >s 17

define i1 @or_ule_sgt(i8 %x) {
; CHECK-LABEL: @or_ule_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x <=s 17

define i1 @or_ule_sle(i8 %x) {
; CHECK-LABEL: @or_ule_sle(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x <s 17

define i1 @or_ule_slt(i8 %x) {
; CHECK-LABEL: @or_ule_slt(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x >=u 17

define i1 @or_ule_uge(i8 %x) {
; CHECK-LABEL: @or_ule_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x >u 17

define i1 @or_ule_ugt(i8 %x) {
; CHECK-LABEL: @or_ule_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x <=u 17

define i1 @or_ule_ule(i8 %x) {
; CHECK-LABEL: @or_ule_ule(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 13 || x <u 17

define i1 @or_ule_ult(i8 %x) {
; CHECK-LABEL: @or_ule_ult(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ule i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ult
; x <u 13 || x == 17

define i1 @or_ult_eq(i8 %x) {
; CHECK-LABEL: @or_ult_eq(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x != 17

define i1 @or_ult_ne(i8 %x) {
; CHECK-LABEL: @or_ult_ne(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x >=s 17

define i1 @or_ult_sge(i8 %x) {
; CHECK-LABEL: @or_ult_sge(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x >s 17

define i1 @or_ult_sgt(i8 %x) {
; CHECK-LABEL: @or_ult_sgt(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x <=s 17

define i1 @or_ult_sle(i8 %x) {
; CHECK-LABEL: @or_ult_sle(
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x <s 17

define i1 @or_ult_slt(i8 %x) {
; CHECK-LABEL: @or_ult_slt(
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x >=u 17

define i1 @or_ult_uge(i8 %x) {
; CHECK-LABEL: @or_ult_uge(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x >u 17

define i1 @or_ult_ugt(i8 %x) {
; CHECK-LABEL: @or_ult_ugt(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 13
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x <=u 17

define i1 @or_ult_ule(i8 %x) {
; CHECK-LABEL: @or_ult_ule(
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 13 || x <u 17

define i1 @or_ult_ult(i8 %x) {
; CHECK-LABEL: @or_ult_ult(
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ult i8 %x, 13
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; eq
; x == 23 || x == 17

define i1 @or_eq_eq_swap(i8 %x) {
; CHECK-LABEL: @or_eq_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x != 17

define i1 @or_eq_ne_swap(i8 %x) {
; CHECK-LABEL: @or_eq_ne_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x >=s 17

define i1 @or_eq_sge_swap(i8 %x) {
; CHECK-LABEL: @or_eq_sge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x >s 17

define i1 @or_eq_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_eq_sgt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x <=s 17

define i1 @or_eq_sle_swap(i8 %x) {
; CHECK-LABEL: @or_eq_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x <s 17

define i1 @or_eq_slt_swap(i8 %x) {
; CHECK-LABEL: @or_eq_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x >=u 17

define i1 @or_eq_uge_swap(i8 %x) {
; CHECK-LABEL: @or_eq_uge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x >u 17

define i1 @or_eq_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_eq_ugt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x <=u 17

define i1 @or_eq_ule_swap(i8 %x) {
; CHECK-LABEL: @or_eq_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x == 23 || x <u 17

define i1 @or_eq_ult_swap(i8 %x) {
; CHECK-LABEL: @or_eq_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp eq i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp eq i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ne
; x != 23 || x == 17

define i1 @or_ne_eq_swap(i8 %x) {
; CHECK-LABEL: @or_ne_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x != 17

define i1 @or_ne_ne_swap(i8 %x) {
; CHECK-LABEL: @or_ne_ne_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x >=s 17

define i1 @or_ne_sge_swap(i8 %x) {
; CHECK-LABEL: @or_ne_sge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x >s 17

define i1 @or_ne_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_ne_sgt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x <=s 17

define i1 @or_ne_sle_swap(i8 %x) {
; CHECK-LABEL: @or_ne_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x <s 17

define i1 @or_ne_slt_swap(i8 %x) {
; CHECK-LABEL: @or_ne_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x >=u 17

define i1 @or_ne_uge_swap(i8 %x) {
; CHECK-LABEL: @or_ne_uge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x >u 17

define i1 @or_ne_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_ne_ugt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ne i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x <=u 17

define i1 @or_ne_ule_swap(i8 %x) {
; CHECK-LABEL: @or_ne_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x != 23 || x <u 17

define i1 @or_ne_ult_swap(i8 %x) {
; CHECK-LABEL: @or_ne_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ne i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ne i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sge
; x >=s 23 || x == 17

define i1 @or_sge_eq_swap(i8 %x) {
; CHECK-LABEL: @or_sge_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x != 17

define i1 @or_sge_ne_swap(i8 %x) {
; CHECK-LABEL: @or_sge_ne_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x >=s 17

define i1 @or_sge_sge_swap(i8 %x) {
; CHECK-LABEL: @or_sge_sge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x >s 17

define i1 @or_sge_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_sge_sgt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x <=s 17

define i1 @or_sge_sle_swap(i8 %x) {
; CHECK-LABEL: @or_sge_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x <s 17

define i1 @or_sge_slt_swap(i8 %x) {
; CHECK-LABEL: @or_sge_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x >=u 17

define i1 @or_sge_uge_swap(i8 %x) {
; CHECK-LABEL: @or_sge_uge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x >u 17

define i1 @or_sge_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_sge_ugt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x <=u 17

define i1 @or_sge_ule_swap(i8 %x) {
; CHECK-LABEL: @or_sge_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=s 23 || x <u 17

define i1 @or_sge_ult_swap(i8 %x) {
; CHECK-LABEL: @or_sge_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sge i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sgt
; x >s 23 || x == 17

define i1 @or_sgt_eq_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x != 17

define i1 @or_sgt_ne_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_ne_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x >=s 17

define i1 @or_sgt_sge_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_sge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x >s 17

define i1 @or_sgt_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_sgt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x <=s 17

define i1 @or_sgt_sle_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x <s 17

define i1 @or_sgt_slt_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x >=u 17

define i1 @or_sgt_uge_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_uge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x >u 17

define i1 @or_sgt_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_ugt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x <=u 17

define i1 @or_sgt_ule_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >s 23 || x <u 17

define i1 @or_sgt_ult_swap(i8 %x) {
; CHECK-LABEL: @or_sgt_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sgt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp sgt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; sle
; x <=s 23 || x == 17

define i1 @or_sle_eq_swap(i8 %x) {
; CHECK-LABEL: @or_sle_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x != 17

define i1 @or_sle_ne_swap(i8 %x) {
; CHECK-LABEL: @or_sle_ne_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sle i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x >=s 17

define i1 @or_sle_sge_swap(i8 %x) {
; CHECK-LABEL: @or_sle_sge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sle i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x >s 17

define i1 @or_sle_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_sle_sgt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sle i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x <=s 17

define i1 @or_sle_sle_swap(i8 %x) {
; CHECK-LABEL: @or_sle_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x <s 17

define i1 @or_sle_slt_swap(i8 %x) {
; CHECK-LABEL: @or_sle_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x >=u 17

define i1 @or_sle_uge_swap(i8 %x) {
; CHECK-LABEL: @or_sle_uge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sle i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x >u 17

define i1 @or_sle_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_sle_ugt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp sle i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x <=u 17

define i1 @or_sle_ule_swap(i8 %x) {
; CHECK-LABEL: @or_sle_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=s 23 || x <u 17

define i1 @or_sle_ult_swap(i8 %x) {
; CHECK-LABEL: @or_sle_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp sle i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp sle i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; slt
; x <s 23 || x == 17

define i1 @or_slt_eq_swap(i8 %x) {
; CHECK-LABEL: @or_slt_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x != 17

define i1 @or_slt_ne_swap(i8 %x) {
; CHECK-LABEL: @or_slt_ne_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp slt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x >=s 17

define i1 @or_slt_sge_swap(i8 %x) {
; CHECK-LABEL: @or_slt_sge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp slt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x >s 17

define i1 @or_slt_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_slt_sgt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp slt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x <=s 17

define i1 @or_slt_sle_swap(i8 %x) {
; CHECK-LABEL: @or_slt_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x <s 17

define i1 @or_slt_slt_swap(i8 %x) {
; CHECK-LABEL: @or_slt_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x >=u 17

define i1 @or_slt_uge_swap(i8 %x) {
; CHECK-LABEL: @or_slt_uge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp slt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x >u 17

define i1 @or_slt_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_slt_ugt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp slt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x <=u 17

define i1 @or_slt_ule_swap(i8 %x) {
; CHECK-LABEL: @or_slt_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <s 23 || x <u 17

define i1 @or_slt_ult_swap(i8 %x) {
; CHECK-LABEL: @or_slt_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp slt i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp slt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; uge
; x >=u 23 || x == 17

define i1 @or_uge_eq_swap(i8 %x) {
; CHECK-LABEL: @or_uge_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x != 17

define i1 @or_uge_ne_swap(i8 %x) {
; CHECK-LABEL: @or_uge_ne_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x >=s 17

define i1 @or_uge_sge_swap(i8 %x) {
; CHECK-LABEL: @or_uge_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x >s 17

define i1 @or_uge_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_uge_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x <=s 17

define i1 @or_uge_sle_swap(i8 %x) {
; CHECK-LABEL: @or_uge_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x <s 17

define i1 @or_uge_slt_swap(i8 %x) {
; CHECK-LABEL: @or_uge_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x >=u 17

define i1 @or_uge_uge_swap(i8 %x) {
; CHECK-LABEL: @or_uge_uge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x >u 17

define i1 @or_uge_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_uge_ugt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x <=u 17

define i1 @or_uge_ule_swap(i8 %x) {
; CHECK-LABEL: @or_uge_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >=u 23 || x <u 17

define i1 @or_uge_ult_swap(i8 %x) {
; CHECK-LABEL: @or_uge_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp uge i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp uge i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ugt
; x >u 23 || x == 17

define i1 @or_ugt_eq_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp eq i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x != 17

define i1 @or_ugt_ne_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_ne_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ne i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x >=s 17

define i1 @or_ugt_sge_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x >s 17

define i1 @or_ugt_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x <=s 17

define i1 @or_ugt_sle_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x <s 17

define i1 @or_ugt_slt_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x >=u 17

define i1 @or_ugt_uge_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_uge_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp uge i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x >u 17

define i1 @or_ugt_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_ugt_swap(
; CHECK-NEXT:    [[B:%.*]] = icmp ugt i8 %x, 17
; CHECK-NEXT:    ret i1 [[B]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x <=u 17

define i1 @or_ugt_ule_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ule i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x >u 23 || x <u 17

define i1 @or_ugt_ult_swap(i8 %x) {
; CHECK-LABEL: @or_ugt_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ugt i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp ult i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ugt i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ule
; x <=u 23 || x == 17

define i1 @or_ule_eq_swap(i8 %x) {
; CHECK-LABEL: @or_ule_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x != 17

define i1 @or_ule_ne_swap(i8 %x) {
; CHECK-LABEL: @or_ule_ne_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ule i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x >=s 17

define i1 @or_ule_sge_swap(i8 %x) {
; CHECK-LABEL: @or_ule_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x >s 17

define i1 @or_ule_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_ule_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x <=s 17

define i1 @or_ule_sle_swap(i8 %x) {
; CHECK-LABEL: @or_ule_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x <s 17

define i1 @or_ule_slt_swap(i8 %x) {
; CHECK-LABEL: @or_ule_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x >=u 17

define i1 @or_ule_uge_swap(i8 %x) {
; CHECK-LABEL: @or_ule_uge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ule i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x >u 17

define i1 @or_ule_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_ule_ugt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ule i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x <=u 17

define i1 @or_ule_ule_swap(i8 %x) {
; CHECK-LABEL: @or_ule_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <=u 23 || x <u 17

define i1 @or_ule_ult_swap(i8 %x) {
; CHECK-LABEL: @or_ule_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ule i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ule i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; ult
; x <u 23 || x == 17

define i1 @or_ult_eq_swap(i8 %x) {
; CHECK-LABEL: @or_ult_eq_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp eq i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x != 17

define i1 @or_ult_ne_swap(i8 %x) {
; CHECK-LABEL: @or_ult_ne_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ult i8 %x, 23
  %b = icmp ne i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x >=s 17

define i1 @or_ult_sge_swap(i8 %x) {
; CHECK-LABEL: @or_ult_sge_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sge i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x >s 17

define i1 @or_ult_sgt_swap(i8 %x) {
; CHECK-LABEL: @or_ult_sgt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sgt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sgt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x <=s 17

define i1 @or_ult_sle_swap(i8 %x) {
; CHECK-LABEL: @or_ult_sle_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp sle i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp sle i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x <s 17

define i1 @or_ult_slt_swap(i8 %x) {
; CHECK-LABEL: @or_ult_slt_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    [[B:%.*]] = icmp slt i8 %x, 17
; CHECK-NEXT:    [[C:%.*]] = or i1 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp slt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x >=u 17

define i1 @or_ult_uge_swap(i8 %x) {
; CHECK-LABEL: @or_ult_uge_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ult i8 %x, 23
  %b = icmp uge i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x >u 17

define i1 @or_ult_ugt_swap(i8 %x) {
; CHECK-LABEL: @or_ult_ugt_swap(
; CHECK-NEXT:    ret i1 true
;
  %a = icmp ult i8 %x, 23
  %b = icmp ugt i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x <=u 17

define i1 @or_ult_ule_swap(i8 %x) {
; CHECK-LABEL: @or_ult_ule_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ule i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; x <u 23 || x <u 17

define i1 @or_ult_ult_swap(i8 %x) {
; CHECK-LABEL: @or_ult_ult_swap(
; CHECK-NEXT:    [[A:%.*]] = icmp ult i8 %x, 23
; CHECK-NEXT:    ret i1 [[A]]
;
  %a = icmp ult i8 %x, 23
  %b = icmp ult i8 %x, 17
  %c = or i1 %a, %b
  ret i1 %c
}

; Special case - slt is uge
; x <u 31 && x <s 0

define i1 @empty2(i32 %x) {
; CHECK-LABEL: @empty2(
; CHECK-NEXT:    ret i1 false
;
  %a = icmp ult i32 %x, 31
  %b = icmp slt i32 %x, 0
  %c = and i1 %a, %b
  ret i1 %c
}

