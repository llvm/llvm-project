; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
;
; FIXME: two consecutive immediate adds not fused in i16/i8 functions.

declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i8 @llvm.ctlz.i8(i8, i1)

define i64 @f0(i64 %arg) {
; CHECK-LABEL: f0:
; CHECK-LABEL: %bb.0:
; CHECK-NOT:   %bb.1:
; CHECK: flogr
  %1 = tail call i64 @llvm.ctlz.i64(i64 %arg, i1 false)
  ret i64 %1
}

define i64 @f1(i64 %arg) {
; CHECK-LABEL: f1:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: flogr
; CHECK-NEXT: # kill
; CHECK-NEXT: br %r14
  %1 = tail call i64 @llvm.ctlz.i64(i64 %arg, i1 true)
  ret i64 %1
}

define i32 @f2(i32 %arg) {
; CHECK-LABEL: f2:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: llgfr %r0, %r2
; CHECK-NEXT: flogr %r2, %r0
; CHECK-NEXT: aghi  %r2, -32
; CHECK-NEXT: # kill
; CHECK-NEXT: br %r14
  %1 = tail call i32 @llvm.ctlz.i32(i32 %arg, i1 false)
  ret i32 %1
}

define i32 @f3(i32 %arg) {
; CHECK-LABEL: f3:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: llgfr %r0, %r2
; CHECK-NEXT: flogr %r2, %r0
; CHECK-NEXT: aghi  %r2, -32
; CHECK-NEXT: # kill
; CHECK-NEXT: br %r14
  %1 = tail call i32 @llvm.ctlz.i32(i32 %arg, i1 true)
  ret i32 %1
}

define i16 @f4(i16 %arg) {
; CHECK-LABEL: f4:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: # kill
; CHECK-NEXT: llghr %r0, %r2
; CHECK-NEXT: flogr %r0, %r0
; CHECK-NEXT: aghi  %r0, -32
; CHECK-NEXT: ahik  %r2, %r0, -16
; CHECK-NEXT: br %r14
  %1 = tail call i16 @llvm.ctlz.i16(i16 %arg, i1 false)
  ret i16 %1
}

define i16 @f5(i16 %arg) {
; CHECK-LABEL: f5:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: # kill
; CHECK-NEXT: llghr %r0, %r2
; CHECK-NEXT: flogr %r0, %r0
; CHECK-NEXT: aghi  %r0, -32
; CHECK-NEXT: ahik  %r2, %r0, -16
; CHECK-NEXT: br %r14
  %1 = tail call i16 @llvm.ctlz.i16(i16 %arg, i1 true)
  ret i16 %1
}

define i8 @f6(i8 %arg) {
; CHECK-LABEL: f6:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: # kill
; CHECK-NEXT: llgcr %r0, %r2
; CHECK-NEXT: flogr %r0, %r0
; CHECK-NEXT: aghi  %r0, -32
; CHECK-NEXT: ahik  %r2, %r0, -24
; CHECK-NEXT: br %r14
  %1 = tail call i8 @llvm.ctlz.i8(i8 %arg, i1 false)
  ret i8 %1
}

define i8 @f7(i8 %arg) {
; CHECK-LABEL: f7:
; CHECK-LABEL: %bb.0:
; CHECK-NEXT: # kill
; CHECK-NEXT: llgcr %r0, %r2
; CHECK-NEXT: flogr %r0, %r0
; CHECK-NEXT: aghi  %r0, -32
; CHECK-NEXT: ahik  %r2, %r0, -24
; CHECK-NEXT: br %r14
  %1 = tail call i8 @llvm.ctlz.i8(i8 %arg, i1 true)
  ret i8 %1
}
