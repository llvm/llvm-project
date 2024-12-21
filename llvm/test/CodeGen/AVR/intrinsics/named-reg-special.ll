; RUN: llc -O0 < %s -mtriple=avr | FileCheck %s

; CHECK-LABEL: read_sp:
; CHECK: in r24, 61
; CHECK: in r25, 62
define i16 @read_sp() {
entry:
  %sp = call i16 @llvm.read_register.i16(metadata !0)
  ret i16 %sp
}

; CHECK-LABEL: read_r0:
; CHECK: mov r24, r0
define i8 @read_r0() {
entry:
  %r0 = call i8 @llvm.read_register.i8(metadata !1)
  ret i8 %r0
}

; CHECK-LABEL: read_r1:
; CHECK: mov r24, r1
define i8 @read_r1() {
entry:
  %r1 = call i8 @llvm.read_register.i8(metadata !2)
  ret i8 %r1
}

; CHECK-LABEL: read_r1r0:
; CHECK: mov r24, r0
; CHECK: mov r25, r1
define i16 @read_r1r0() {
entry:
  %r1r0 = call i16 @llvm.read_register.i16(metadata !1)
  ret i16 %r1r0
}

declare i16 @llvm.read_register.i16(metadata)
declare i8 @llvm.read_register.i8(metadata)

!0 = !{!"sp"}
!1 = !{!"r0"}
!2 = !{!"r1"}
