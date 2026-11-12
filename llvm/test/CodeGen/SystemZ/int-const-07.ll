; Test loading of 128-bit constants in vector registers on z13
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Constant zero.
define i128 @f1() {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vgbm %v0, 0
; CHECK-NEXT:    vst %v0, 0(%r2), 3
; CHECK-NEXT:    br %r14
  ret i128 0
}

; Constant created using VBGM.
define i128 @f2() {
; CHECK-LABEL: f2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vgbm %v0, 1
; CHECK-NEXT:    vst %v0, 0(%r2), 3
; CHECK-NEXT:    br %r14
  ret i128 255
}

; Constant created using VREPIB.
define i128 @f3() {
; CHECK-LABEL: f3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vrepib %v0, 1
; CHECK-NEXT:    vst %v0, 0(%r2), 3
; CHECK-NEXT:    br %r14
  ret i128 1334440654591915542993625911497130241
}

; Constant loaded from literal pool.
define i128 @f4() {
; CHECK-LABEL: .LCPI3_0:
; CHECK-NEXT:    .quad   54210108624275221
; CHECK-NEXT:    .quad   -5527149226598858752
; CHECK-LABEL: f4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    larl %r1, .LCPI3_0
; CHECK-NEXT:    vl %v0, 0(%r1), 3
; CHECK-NEXT:    vst %v0, 0(%r2), 3
; CHECK-NEXT:    br %r14
  ret i128 1000000000000000000000000000000000000
}
