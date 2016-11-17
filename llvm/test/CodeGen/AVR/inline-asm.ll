; RUN: llc < %s -march=avr | FileCheck %s
; XFAIL: *

;CHECK-LABEL: no_operands:
define void @no_operands() {
  ;CHECK: add r24, r22
  call void asm sideeffect "add r24, r22", ""() nounwind
  ret void
}

;CHECK-LABEL: input_operand:
define void @input_operand(i8 %a) {
  ;CHECK: add r24, r24
  call void asm sideeffect "add $0, $0", "r"(i8 %a) nounwind
  ret void
}

;CHECK-LABEL: simple_upper_regs:
define void @simple_upper_regs(i8 %p0, i8 %p1, i8 %p2, i8 %p3,
                               i8 %p4, i8 %p5, i8 %p6, i8 %p7) {
  ;CHECK: some_instr r17, r22, r20, r18, r16, r19, r21, r23
  call void asm sideeffect "some_instr $0, $1, $2, $3, $4, $5, $6, $7",
                           "a,a,a,a,a,a,a,a" (i8 %p0, i8 %p1, i8 %p2, i8 %p3,
                                              i8 %p4, i8 %p5, i8 %p6, i8 %p7) nounwind
  ret void
}

;CHECK-LABEL: upper_regs:
define void @upper_regs(i8 %p0) {
  ;CHECK: some_instr r24
  call void asm sideeffect "some_instr $0", "d" (i8 %p0) nounwind
  ret void
}

;CHECK-LABEL: lower_regs:
define void @lower_regs(i8 %p0) {
  ;CHECK: some_instr r15
  call void asm sideeffect "some_instr $0", "l" (i8 %p0) nounwind
  ret void
}

;CHECK-LABEL: special_upper_regs:
define void @special_upper_regs(i8 %p0, i8 %p1, i8 %p2, i8 %p3) {
  ;CHECK: some_instr r24,r28,r26,r30
  call void asm sideeffect "some_instr $0,$1,$2,$3", "w,w,w,w" (i8 %p0, i8 %p1, i8 %p2, i8 %p3) nounwind
  ret void
}

;CHECK-LABEL: xyz_reg:
define void @xyz_reg(i16 %var) {
  ;CHECK: some_instr r26, r28, r30
  call void asm sideeffect "some_instr $0, $1, $2", "x,y,z" (i16 %var, i16 %var, i16 %var) nounwind
  ret void
}

;TODO
; How to use SP reg properly in inline asm??
; define void @sp_reg(i16 %var) 

;CHECK-LABEL: ptr_reg:
define void @ptr_reg(i16 %var0, i16 %var1, i16 %var2) {
  ;CHECK: some_instr r28, r26, r30
  call void asm sideeffect "some_instr $0, $1, $2", "e,e,e" (i16 %var0, i16 %var1, i16 %var2) nounwind
  ret void
}

;CHECK-LABEL: base_ptr_reg:
define void @base_ptr_reg(i16 %var0, i16 %var1) {
  ;CHECK: some_instr r28, r30
  call void asm sideeffect "some_instr $0, $1", "b,b" (i16 %var0, i16 %var1) nounwind
  ret void
}

;CHECK-LABEL: input_output_operand:
define i8 @input_output_operand(i8 %a, i8 %b) {
  ;CHECK: add r24, r24
  %1 = call i8 asm "add $0, $1", "=r,r"(i8 %a) nounwind
  ret i8 %1
}

;CHECK-LABEL: temp_reg:
define void @temp_reg(i8 %a) {
  ;CHECK: some_instr r0
  call void asm sideeffect "some_instr $0", "t" (i8 %a) nounwind
  ret void
}

;CHECK-LABEL: int_0_63:
define void @int_0_63() {
  ;CHECK: some_instr 5
  call void asm sideeffect "some_instr $0", "I" (i8 5) nounwind
  ret void
}

;CHECK-LABEL: int_minus63_0:
define void @int_minus63_0() {
  ;CHECK: some_instr -5
  call void asm sideeffect "some_instr $0", "J" (i8 -5) nounwind
  ret void
}

;CHECK-LABEL: int_2_2:
define void @int_2_2() {
  ;CHECK: some_instr 2
  call void asm sideeffect "some_instr $0", "K" (i8 2) nounwind
  ret void
}

;CHECK-LABEL: int_0_0:
define void @int_0_0() {
  ;CHECK: some_instr 0
  call void asm sideeffect "some_instr $0", "L" (i8 0) nounwind
  ret void
}

;CHECK-LABEL: int_0_255:
define void @int_0_255() {
  ;CHECK: some_instr 254
  call void asm sideeffect "some_instr $0", "M" (i8 254) nounwind
  ret void
}

;CHECK-LABEL: int_minus1_minus1:
define void @int_minus1_minus1() {
  ;CHECK: some_instr -1
  call void asm sideeffect "some_instr $0", "N" (i8 -1) nounwind
  ret void
}

;CHECK-LABEL: int_8_or_16_or_24:
define void @int_8_or_16_or_24() {
  ;CHECK: some_instr 8, 16, 24
  call void asm sideeffect "some_instr $0, $1, $2", "O,O,O" (i8 8, i8 16, i8 24) nounwind
  ret void
}

;CHECK-LABEL: int_1_1:
define void @int_1_1() {
  ;CHECK: some_instr 1
  call void asm sideeffect "some_instr $0", "P" (i8 1) nounwind
  ret void
}

;CHECK-LABEL: int_minus6_5:
define void @int_minus6_5() {
  ;CHECK: some_instr -6
  call void asm sideeffect "some_instr $0", "R" (i8 -6) nounwind
  ret void
}

;CHECK-LABEL: float_0_0:
define void @float_0_0() {
  ;CHECK: some_instr 0
  call void asm sideeffect "some_instr $0", "G" (float 0.0) nounwind
  ret void
}


; Memory constraint

@a = internal global i16 0, align 4
@b = internal global i16 0, align 4

; CHECK-LABEL: mem_global:
define void @mem_global() {
  ;CHECK: some_instr Y, Z
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* @a, i16* @b)
  ret void
}

; CHECK-LABEL: mem_params:
define void @mem_params(i16* %a, i16* %b) {
  ;CHECK: some_instr Y, Z
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_local:
define void @mem_local() {
  %a = alloca i16
  %b = alloca i16
  ;CHECK: some_instr Y+3, Y+1
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_mixed:
define void @mem_mixed() {
  %a = alloca i16
  %b = alloca i16
  ;CHECK: some_instr Z, Y+3, Y+1
  call void asm "some_instr $0, $1, $2", "=*Q,=*Q,=*Q"(i16* @a, i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_gep:
define i8 @mem_gep(i8* %p) {
entry:
; CHECK: movw r30, r24
  %arrayidx = getelementptr inbounds i8, i8* %p, i16 1
; CHECK: ld r24, Z+1
  %0 = tail call i8 asm sideeffect "ld $0, $1\0A\09", "=r,*Q"(i8* %arrayidx)
  ret i8 %0
}

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


