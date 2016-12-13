; RUN: llc < %s -march=avr | FileCheck %s
; XFAIL: *

; Multibyte references

; CHECK-LABEL: multibyte_i16
define void @multibyte_i16(i16 %a) {
entry:
; CHECK: instr r24 r25
  call void asm sideeffect "instr ${0:A} ${0:B}", "r"(i16 %a)
; CHECK: instr r25 r24
  call void asm sideeffect "instr ${0:B} ${0:A}", "r"(i16 %a)
  ret void
}

; CHECK-LABEL: multibyte_i32
define void @multibyte_i32(i32 %a) {
entry:
; CHECK: instr r22 r23 r24 r25
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "r"(i32 %a)
; CHECK: instr r25 r24 r23 r22
  call void asm sideeffect "instr ${0:D} ${0:C} ${0:B} ${0:A}", "r"(i32 %a)
  ret void
}

; CHECK-LABEL: multibyte_alternative_name
define void @multibyte_alternative_name(i16* %p) {
entry:
; CHECK: instr Z
  call void asm sideeffect "instr ${0:a}", "e" (i16* %p)
  ret void
}

; CHECK-LABEL: multibyte_a_i32
define void @multibyte_a_i32() {
entry:
  %a = alloca i32
  %0 = load i32, i32* %a
; CHECK: instr r20 r21 r22 r23
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "a"(i32 %0)
  ret void
}

@c = internal global i32 0

; CHECK-LABEL: multibyte_b_i32
define void @multibyte_b_i32() {
entry:
  %0 = load i32, i32* @c
; CHECK: instr r28 r29 r30 r31
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "b"(i32 %0)
  ret void
}

; CHECK-LABEL: multibyte_d_i32
define void @multibyte_d_i32() {
entry:
  %a = alloca i32
  %0 = load i32, i32* %a
; CHECK: instr r18 r19 r24 r25
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "d"(i32 %0)
  ret void
}

; CHECK-LABEL: multibyte_e_i32
define void @multibyte_e_i32() {
entry:
  %a = alloca i32
  %0 = load i32, i32* %a
; CHECK: instr r26 r27 r30 r31
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "e"(i32 %0)
  ret void
}

; CHECK-LABEL: multibyte_l_i32
define void @multibyte_l_i32() {
entry:
  %a = alloca i32
  %0 = load i32, i32* %a
; CHECK: instr r12 r13 r14 r15
  call void asm sideeffect "instr ${0:A} ${0:B} ${0:C} ${0:D}", "l"(i32 %0)
  ret void
}

; CHECK-LABEL: multibyte_a_i16
define void @multibyte_a_i16() {
entry:
  %a = alloca i16
  %0 = load i16, i16* %a
; CHECK: instr r22 r23
  call void asm sideeffect "instr ${0:A} ${0:B}", "a"(i16 %0)
  ret void
}

; CHECK-LABEL: multibyte_b_i16
define void @multibyte_b_i16() {
entry:
  %a = alloca i16
  %0 = load i16, i16* %a
; CHECK: instr r30 r31
  call void asm sideeffect "instr ${0:A} ${0:B}", "b"(i16 %0)
  ret void
}

; CHECK-LABEL: multibyte_d_i16
define void @multibyte_d_i16() {
entry:
  %a = alloca i16
  %0 = load i16, i16* %a
; CHECK: instr r24 r25
  call void asm sideeffect "instr ${0:A} ${0:B}", "d"(i16 %0)
  ret void
}

; CHECK-LABEL: multibyte_e_i16
define void @multibyte_e_i16() {
entry:
  %a = alloca i16
  %0 = load i16, i16* %a
; CHECK: instr r30 r31
  call void asm sideeffect "instr ${0:A} ${0:B}", "e"(i16 %0)
  ret void
}

; CHECK-LABEL: multibyte_l_i16
define void @multibyte_l_i16() {
entry:
  %a = alloca i16
  %0 = load i16, i16* %a
; CHECK: instr r14 r15
  call void asm sideeffect "instr ${0:A} ${0:B}", "l"(i16 %0)
  ret void
}


