; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

define i8 @ctz_v8i16(<8 x i16> %a) {
; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqw %xmm0, %xmm1
; CHECK-NEXT:    packsswb %xmm1, %xmm1
; CHECK-NEXT:    pandn {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    movdqa %xmm1, -{{[0-9]+}}(%rsp)
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    movl -{{[0-9]+}}(%rsp), %edx
; CHECK-NEXT:    cmpb %cl, %al
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    cmpb %dl, %cl
; CHECK-NEXT:    cmovbel %edx, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movb $8, %al
; CHECK-NEXT:    subb %cl, %al
; CHECK-NEXT:    retq
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16> %a, i1 0)
  ret i8 %res
}

define i16 @ctz_v4i32(<4 x i32> %a) {
; CHECK-LABEL: .LCPI1_0:
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm1
; CHECK-NEXT:    packssdw %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm0
; CHECK-NEXT:    pxor %xmm1, %xmm0
; CHECK-NEXT:    packsswb %xmm0, %xmm0
; CHECK-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; CHECK-NEXT:    movd %xmm0, %eax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $8, %ecx
; CHECK-NEXT:    cmpb %cl, %al
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    shrl $16, %edx
; CHECK-NEXT:    cmpb %dl, %cl
; CHECK-NEXT:    cmoval %ecx, %edx
; CHECK-NEXT:    shrl $24, %eax
; CHECK-NEXT:    cmpb %al, %dl
; CHECK-NEXT:    cmoval %edx, %eax
; CHECK-NEXT:    movb $4, %cl
; CHECK-NEXT:    subb %al, %cl
; CHECK-NEXT:    movzbl %cl, %eax
; CHECK-NEXT:    # kill: def $ax killed $ax killed $eax
; CHECK-NEXT:    retq
  %res = call i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32> %a, i1 0)
  ret i16 %res
}

; ZERO IS POISON

define i8 @ctz_v8i16_poison(<8 x i16> %a) {
; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i16_poison:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqw %xmm0, %xmm1
; CHECK-NEXT:    packsswb %xmm1, %xmm1
; CHECK-NEXT:    pandn {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    movdqa %xmm1, -{{[0-9]+}}(%rsp)
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    movl -{{[0-9]+}}(%rsp), %edx
; CHECK-NEXT:    cmpb %cl, %al
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    cmpb %dl, %cl
; CHECK-NEXT:    cmovbel %edx, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movzbl -{{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    cmpb %al, %cl
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movb $8, %al
; CHECK-NEXT:    subb %cl, %al
; CHECK-NEXT:    retq
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16> %a, i1 1)
  ret i8 %res
}

declare i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16>, i1)
declare i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32>, i1)
