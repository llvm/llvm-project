; RUN: llc < %s -mtriple=mips -mcpu=mips32 -mips-ssection-threshold=8 \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt \
; RUN:   | FileCheck %s

; Test that object with an explicit section that is not .sdata or .sbss are not
; considered in the small data section if they would otherwise qualify to be in
; small data section. Also test that explicitly placing something in the small
; data section uses %gp_rel addressing mode.

@a = constant [2 x i32] zeroinitializer, section ".rodata", align 4
@b = global [4 x i32] zeroinitializer, section ".sdata", align 4
@c = global [4 x i32] zeroinitializer, section ".sbss", align 4

; CHECK-LABEL: g
; CHECK:       lui $[[R:[0-9]+]], %hi(a)
; CHECK:       lw  ${{[0-9]+}}, %lo(a)($[[R]])

define i32 @g() {
entry:
  %0 = load i32, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @a, i32 0, i32 0), align 4
  ret i32 %0
}

; CHECK-LABEL: f:
; CHECK-LABEL: lw ${{[0-9]+}}, %gp_rel(b)($gp)

define i32 @f() {
entry:
  %0 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @b, i32 0, i32 0), align 4
  ret i32 %0
}

; CHECK-LABEL: h:
; CHECK-LABEL: lw ${{[0-9]+}}, %gp_rel(c)($gp)

define i32 @h() {
entry:
  %0 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i32 0), align 4
  ret i32 %0
}


; CHECK:  .type a,@object
; CHECK:  .section  .rodata,"a",@progbits
; CHECK:  .globl  a

; CHECK:  .type b,@object
; CHECK:  .section  .sdata,"aw",@progbits
; CHECK:  .globl  b

; CHECK:  .type c,@object
; CHECK:  .section  .sbss,"aw",@nobits
; CHECK:  .globl  c
