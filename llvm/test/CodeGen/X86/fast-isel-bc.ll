; RUN: llc < %s -O0 -mattr=+mmx,+sse2 | FileCheck %s
; PR4684

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.8"

declare void @func2(x86_mmx)

define void @func1() nounwind {

; This isn't spectacular, but it's MMX code at -O0...
; CHECK:  movq2dq %mm0, %xmm0
; For now, handling of x86_mmx parameters in fast Isel is unimplemented,
; so we get pretty poor code.  The below is preferable.
; CHEK: movl $2, %eax
; CHEK: movd %rax, %mm0
; CHEK: movd %mm0, %rdi

        %tmp0 = bitcast <2 x i32><i32 0, i32 2> to x86_mmx
        call void @func2(x86_mmx %tmp0)
        ret void
}
