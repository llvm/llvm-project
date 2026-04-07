; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

define i8 @ctz_v8i16(<8 x i16> %a) {
; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 7
; CHECK-NEXT:   .short 6
; CHECK-NEXT:   .short 5
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short 3
; CHECK-NEXT:   .short 2
; CHECK-NEXT:   .short 1
; CHECK-LABEL: ctz_v8i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqw %xmm0, %xmm1
; CHECK-NEXT:    pandn {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    pextrw $1, %xmm1, %ecx
; CHECK-NEXT:    movd %xmm1, %eax
; CHECK-NEXT:    cmpw %cx, %ax
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    pextrw $2, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $3, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $4, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $5, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $6, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $7, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movl $8, %eax
; CHECK-NEXT:    subl %ecx, %eax
; CHECK-NEXT:    # kill: def $al killed $al killed $eax
; CHECK-NEXT:    retq
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16> %a, i1 0)
  ret i8 %res
}

define i16 @ctz_v4i32(<4 x i32> %a) {
; CHECK-LABEL: .LCPI1_0:
; CHECK-NEXT:   .long 4
; CHECK-NEXT:   .long 3
; CHECK-NEXT:   .long 2
; CHECK-NEXT:   .long 1
; CHECK-LABEL: ctz_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm1
; CHECK-NEXT:    pandn {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    movd %xmm1, %eax
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[1,1,1,1]
; CHECK-NEXT:    movd %xmm0, %ecx
; CHECK-NEXT:    cmpl %ecx, %eax
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[2,3,2,3]
; CHECK-NEXT:    movd %xmm0, %eax
; CHECK-NEXT:    cmpl %eax, %ecx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[3,3,3,3]
; CHECK-NEXT:    movd %xmm0, %eax
; CHECK-NEXT:    cmpl %eax, %ecx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movl $4, %eax
; CHECK-NEXT:    subl %ecx, %eax
; CHECK-NEXT:    # kill: def $ax killed $ax killed $eax
; CHECK-NEXT:    retq
  %res = call i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32> %a, i1 0)
  ret i16 %res
}

; ZERO IS POISON

define i8 @ctz_v8i16_poison(<8 x i16> %a) {
; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 7
; CHECK-NEXT:   .short 6
; CHECK-NEXT:   .short 5
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short 3
; CHECK-NEXT:   .short 2
; CHECK-NEXT:   .short 1
; CHECK-LABEL: ctz_v8i16_poison:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpeqw %xmm0, %xmm1
; CHECK-NEXT:    pandn {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    pextrw $1, %xmm1, %ecx
; CHECK-NEXT:    movd %xmm1, %eax
; CHECK-NEXT:    cmpw %cx, %ax
; CHECK-NEXT:    cmoval %eax, %ecx
; CHECK-NEXT:    pextrw $2, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $3, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $4, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $5, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $6, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    pextrw $7, %xmm1, %eax
; CHECK-NEXT:    cmpw %ax, %cx
; CHECK-NEXT:    cmovbel %eax, %ecx
; CHECK-NEXT:    movl $8, %eax
; CHECK-NEXT:    subl %ecx, %eax
; CHECK-NEXT:    # kill: def $al killed $al killed $eax
; CHECK-NEXT:    retq
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16> %a, i1 1)
  ret i8 %res
}

declare i8 @llvm.experimental.cttz.elts.i8.v8i16(<8 x i16>, i1)
declare i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32>, i1)
