; RUN: llc < %s -march=avr -mattr=movw | FileCheck %s

; CHECK-LABEL: no_operands:
define void @no_operands() {
  ; CHECK: add {{r[0-9]+}}, {{r[0-9]+}}
  call void asm sideeffect "add r24, r22", ""() nounwind
  ret void
}

; CHECK-LABEL: input_operand:
define void @input_operand(i8 %a) {
  ; CHECK: add {{r[0-9]+}}, {{r[0-9]+}}
  call void asm sideeffect "add $0, $0", "r"(i8 %a) nounwind
  ret void
}

; CHECK-LABEL: simple_upper_regs:
define void @simple_upper_regs(i8 %p0, i8 %p1, i8 %p2, i8 %p3,
                               i8 %p4, i8 %p5, i8 %p6, i8 %p7) {
  ; CHECK: some_instr {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
  call void asm sideeffect "some_instr $0, $1, $2, $3, $4, $5, $6, $7",
                           "a,a,a,a,a,a,a,a" (i8 %p0, i8 %p1, i8 %p2, i8 %p3,
                                              i8 %p4, i8 %p5, i8 %p6, i8 %p7) nounwind
  ret void
}

; CHECK-LABEL: upper_regs:
define void @upper_regs(i8 %p0) {
  ; CHECK: some_instr {{r[0-9]+}}
  call void asm sideeffect "some_instr $0", "d" (i8 %p0) nounwind
  ret void
}

; CHECK-LABEL: lower_regs:
define void @lower_regs(i8 %p0) {
  ; CHECK: some_instr {{r[0-9]+}}
  call void asm sideeffect "some_instr $0", "l" (i8 %p0) nounwind
  ret void
}

; CHECK-LABEL: special_upper_regs:
define void @special_upper_regs(i8 %p0, i8 %p1, i8 %p2, i8 %p3) {
  ; CHECK: some_instr {{r[0-9]+}},{{r[0-9]+}},{{r[0-9]+}},{{r[0-9]+}}
  call void asm sideeffect "some_instr $0,$1,$2,$3", "w,w,w,w" (i8 %p0, i8 %p1, i8 %p2, i8 %p3) nounwind
  ret void
}

; CHECK-LABEL: xyz_reg:
define void @xyz_reg(i16 %var) {
  ; CHECK: some_instr {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
  call void asm sideeffect "some_instr $0, $1, $2", "x,y,z" (i16 %var, i16 %var, i16 %var) nounwind
  ret void
}

;TODO
; How to use SP reg properly in inline asm??
; define void @sp_reg(i16 %var)

; CHECK-LABEL: ptr_reg:
define void @ptr_reg(i16 %var0, i16 %var1, i16 %var2) {
  ; CHECK: some_instr {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
  call void asm sideeffect "some_instr $0, $1, $2", "e,e,e" (i16 %var0, i16 %var1, i16 %var2) nounwind
  ret void
}

; CHECK-LABEL: base_ptr_reg:
define void @base_ptr_reg(i16 %var0, i16 %var1) {
  ; CHECK: some_instr r28, r30
  call void asm sideeffect "some_instr $0, $1", "b,b" (i16 %var0, i16 %var1) nounwind
  ret void
}

; CHECK-LABEL: input_output_operand:
define i8 @input_output_operand(i8 %a, i8 %b) {
  ; CHECK: add {{r[0-9]+}}, {{r[0-9]+}}
  %1 = call i8 asm "add $0, $1", "=r,r"(i8 %a) nounwind
  ret i8 %1
}

; CHECK-LABEL: temp_reg:
define void @temp_reg(i8 %a) {
  ; CHECK: some_instr {{r[0-9]+}}
  call void asm sideeffect "some_instr $0", "t" (i8 %a) nounwind
  ret void
}

; CHECK-LABEL: int_0_63:
define void @int_0_63() {
  ; CHECK: some_instr 5
  call void asm sideeffect "some_instr $0", "I" (i8 5) nounwind
  ret void
}

; CHECK-LABEL: int_minus63_0:
define void @int_minus63_0() {
  ; CHECK: some_instr -5
  call void asm sideeffect "some_instr $0", "J" (i8 -5) nounwind
  ret void
}

; CHECK-LABEL: int_2_2:
define void @int_2_2() {
  ; CHECK: some_instr 2
  call void asm sideeffect "some_instr $0", "K" (i8 2) nounwind
  ret void
}

; CHECK-LABEL: int_0_0:
define void @int_0_0() {
  ; CHECK: some_instr 0
  call void asm sideeffect "some_instr $0", "L" (i8 0) nounwind
  ret void
}

; CHECK-LABEL: int_0_255:
define void @int_0_255() {
  ; CHECK: some_instr 254
  call void asm sideeffect "some_instr $0", "M" (i8 254) nounwind
  ret void
}

; CHECK-LABEL: int_minus1_minus1:
define void @int_minus1_minus1() {
  ; CHECK: some_instr -1
  call void asm sideeffect "some_instr $0", "N" (i8 -1) nounwind
  ret void
}

; CHECK-LABEL: int_8_or_16_or_24:
define void @int_8_or_16_or_24() {
  ; CHECK: some_instr 8, 16, 24
  call void asm sideeffect "some_instr $0, $1, $2", "O,O,O" (i8 8, i8 16, i8 24) nounwind
  ret void
}

; CHECK-LABEL: int_1_1:
define void @int_1_1() {
  ; CHECK: some_instr 1
  call void asm sideeffect "some_instr $0", "P" (i8 1) nounwind
  ret void
}

; CHECK-LABEL: int_minus6_5:
define void @int_minus6_5() {
  ; CHECK: some_instr -6
  call void asm sideeffect "some_instr $0", "R" (i8 -6) nounwind
  ret void
}

; CHECK-LABEL: float_0_0:
define void @float_0_0() {
  ; CHECK: some_instr 0
  call void asm sideeffect "some_instr $0", "G" (float 0.0) nounwind
  ret void
}


; Memory constraint

@a = internal global i16 0, align 4
@b = internal global i16 0, align 4

; CHECK-LABEL: mem_global:
define void @mem_global() {
  ; CHECK: some_instr {{X|Y|Z}}, {{X|Y|Z}}
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* @a, i16* @b)
  ret void
}

; CHECK-LABEL: mem_params:
define void @mem_params(i16* %a, i16* %b) {
  ; CHECK: some_instr {{X|Y|Z}}, {{X|Y|Z}}
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_local:
define void @mem_local() {
  %a = alloca i16
  %b = alloca i16
  ; CHECK: some_instr {{X|Y|Z}}+3, {{X|Y|Z}}+1
  call void asm "some_instr $0, $1", "=*Q,=*Q"(i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_mixed:
define void @mem_mixed() {
  %a = alloca i16
  %b = alloca i16
  ; CHECK: some_instr {{X|Y|Z}}, {{X|Y|Z}}+3, {{X|Y|Z}}+1
  call void asm "some_instr $0, $1, $2", "=*Q,=*Q,=*Q"(i16* @a, i16* %a, i16* %b)
  ret void
}

; CHECK-LABEL: mem_gep:
define i8 @mem_gep(i8* %p) {
entry:
; CHECK: movw {{r[0-9]+}}, [[REG:r[0-9]+]]
  %arrayidx = getelementptr inbounds i8, i8* %p, i16 1
; CHECK: ld [[REG]], {{X|Y|Z}}+1
  %0 = tail call i8 asm sideeffect "ld $0, $1\0A\09", "=r,*Q"(i8* %arrayidx)
  ret i8 %0
}
