; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @nop_test() {
entry:
; CHECK-LABEL: nop_test:

; 2
; CHECK:      bcr 0, %r0

; 4
; CHECK:      bc 0, 0

; 6
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]

; 8
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bcr 0, %r0

; 10
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bc 0, 0

; 12
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]

; 14
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bcr 0, %r0

; 16
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bc 0, 0

; 18
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]

; 20
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bcr 0, %r0

; 22
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bc 0, 0

; 24
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]

; 26
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bcr 0, %r0

; 28
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      bc 0, 0

; 30
; CHECK:      .Ltmp
; CHECK-NEXT: [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]
; CHECK:      [[LAB:.Ltmp[0-9]+]]:
; CHECK-NEXT: brcl 0, [[LAB]]

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
