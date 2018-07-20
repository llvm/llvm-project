; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @nop_test() {
entry:
; CHECK-LABEL: nop_test:

; 2
; CHECK:      bcr 0, %r0

; 4
; CHECK:      bc 0, 0

; 6
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:

; 8
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bcr 0, %r0

; 10
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bc 0, 0

; 12
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:

; 14
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bcr 0, %r0

; 16
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bc 0, 0

; 18
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:

; 20
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bcr 0, %r0

; 22
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bc 0, 0

; 24
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:

; 26
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bcr 0, %r0

; 28
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      bc 0, 0

; 30
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:
; CHECK:      brcl 0, [[LAB:.Ltmp[0-9]+]]
; CHECK-NEXT: [[LAB]]:

  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  0, i32  0)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  2, i32  2)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  4, i32  4)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  6, i32  6)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  8, i32  8)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 10, i32 10)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 12, i32 12)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 14)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 16)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 18, i32 18)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 20, i32 20)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 22, i32 22)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 24, i32 24)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 26, i32 26)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 28, i32 28)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 30, i32 30)
; Add an extra stackmap with a zero-length shadow to thwart the shadow
; optimization. This will force all bytes of the previous shadow to be
; padded with nops.
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 31, i32 0)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
