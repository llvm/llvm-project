; RUN: llc < %s -mtriple=bpfel -mcpu=v4 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb -mcpu=v4 -verify-machineinstrs | FileCheck %s

; A load from a constant global is replaced by the constant. On v4 the
; load may be a sign-extending one (LDSX), and the constant must be
; sign-extended as well.

%struct.S = type { i8, i8, i16, i32 }

@g = internal constant %struct.S { i8 -1, i8 127, i16 -3, i32 -4 }, align 4

define i64 @sext_i8_i64() {
  %v = load i8, ptr @g, align 4
  %s = sext i8 %v to i64
  ret i64 %s
}
; CHECK-LABEL: sext_i8_i64:
; CHECK: r0 = -1
; CHECK-NEXT: exit

define i32 @sext_i8_i32() {
  %v = load i8, ptr @g, align 4
  %s = sext i8 %v to i32
  ret i32 %s
}
; CHECK-LABEL: sext_i8_i32:
; CHECK: w0 = -1
; CHECK-NEXT: exit

define i64 @sext_i8_i64_pos() {
  %p = getelementptr inbounds i8, ptr @g, i64 1
  %v = load i8, ptr %p, align 1
  %s = sext i8 %v to i64
  ret i64 %s
}
; CHECK-LABEL: sext_i8_i64_pos:
; CHECK: r0 = 127
; CHECK-NEXT: exit

define i64 @sext_i16_i64() {
  %p = getelementptr inbounds i8, ptr @g, i64 2
  %v = load i16, ptr %p, align 2
  %s = sext i16 %v to i64
  ret i64 %s
}
; CHECK-LABEL: sext_i16_i64:
; CHECK: r0 = -3
; CHECK-NEXT: exit

define i32 @sext_i16_i32() {
  %p = getelementptr inbounds i8, ptr @g, i64 2
  %v = load i16, ptr %p, align 2
  %s = sext i16 %v to i32
  ret i32 %s
}
; CHECK-LABEL: sext_i16_i32:
; CHECK: w0 = -3
; CHECK-NEXT: exit

define i64 @sext_i32_i64() {
  %p = getelementptr inbounds i8, ptr @g, i64 4
  %v = load i32, ptr %p, align 4
  %s = sext i32 %v to i64
  ret i64 %s
}
; CHECK-LABEL: sext_i32_i64:
; CHECK: r0 = -4
; CHECK-NEXT: exit

define i64 @zext_i8_i64() {
  %v = load i8, ptr @g, align 4
  %s = zext i8 %v to i64
  ret i64 %s
}
; CHECK-LABEL: zext_i8_i64:
; CHECK: r0 = 255
; CHECK-NEXT: exit
