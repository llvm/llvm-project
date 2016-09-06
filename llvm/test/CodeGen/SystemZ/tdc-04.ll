; Test the Test Data Class instruction logic operation conversion from
; signbit extraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;

; Extract sign bit.
define i32 @f1(float %x) {
; CHECK-LABEL: f1
; CHECK: tceb %f0, 1365
  %cast = bitcast float %x to i32
  %res = icmp slt i32 %cast, 0
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Extract negated sign bit.
define i32 @f2(float %x) {
; CHECK-LABEL: f2
; CHECK: tceb %f0, 2730
  %cast = bitcast float %x to i32
  %res = icmp sgt i32 %cast, -1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Extract sign bit.
define i32 @f3(double %x) {
; CHECK-LABEL: f3
; CHECK: tcdb %f0, 1365
  %cast = bitcast double %x to i64
  %res = icmp slt i64 %cast, 0
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Extract negated sign bit.
define i32 @f4(double %x) {
; CHECK-LABEL: f4
; CHECK: tcdb %f0, 2730
  %cast = bitcast double %x to i64
  %res = icmp sgt i64 %cast, -1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Extract sign bit.
define i32 @f5(fp128 %x) {
; CHECK-LABEL: f5
; CHECK: tcxb %f0, 1365
  %cast = bitcast fp128 %x to i128
  %res = icmp slt i128 %cast, 0
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Extract negated sign bit.
define i32 @f6(fp128 %x) {
; CHECK-LABEL: f6
; CHECK: tcxb %f0, 2730
  %cast = bitcast fp128 %x to i128
  %res = icmp sgt i128 %cast, -1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Wrong const.
define i32 @f7(float %x) {
; CHECK-LABEL: f7
; CHECK-NOT: tceb
  %cast = bitcast float %x to i32
  %res = icmp slt i32 %cast, -1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Wrong pred.
define i32 @f8(float %x) {
; CHECK-LABEL: f8
; CHECK-NOT: tceb
  %cast = bitcast float %x to i32
  %res = icmp eq i32 %cast, 0
  %xres = zext i1 %res to i32
  ret i32 %xres
}
