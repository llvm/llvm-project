; RUN: llc < %s -mtriple=x86_64-- -mattr=+x87,-sse,-sse2 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i686-- -mattr=+x87,-sse,-sse2 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=X86

define i32 @store_and_fcmp_f32(ptr %p, ptr %q) nounwind {
; X64-LABEL: store_and_fcmp_f32:
; X64:       # %bb.0:
; X64-NEXT:    fldz
; X64-NEXT:    flds (%rdi)
; X64-NEXT:    fsts (%rsi)
; X64-NEXT:    fucompi %st(1), %st
; X64-NEXT:    fstp %st(0)
; X64-NEXT:    sete %al
; X64-NEXT:    setnp %cl
; X64-NEXT:    andb %al, %cl
; X64-NEXT:    movzbl %cl, %eax
; X64-NEXT:    andl $1, %eax
; X64-NEXT:    retq
;
; X86-LABEL: store_and_fcmp_f32:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    fldz
; X86-NEXT:    flds (%eax)
; X86-NEXT:    fsts (%ecx)
; X86-NEXT:    fucompi %st(1), %st
; X86-NEXT:    fstp %st(0)
; X86-NEXT:    sete %al
; X86-NEXT:    setnp %cl
; X86-NEXT:    andb %al, %cl
; X86-NEXT:    movzbl %cl, %eax
; X86-NEXT:    andl $1, %eax
; X86-NEXT:    retl
  %v = load float, ptr %p
  store float %v, ptr %q
  %c = fcmp oeq float %v, 0.0
  %r = zext i1 %c to i32
  ret i32 %r
}
