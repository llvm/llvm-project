; RUN: llc < %s | FileCheck %s
; ModuleID = 'asm.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @test1() nounwind {
entry:
; CHECK-LABEL: test1:
; CHECK: movw	%gs:6, %ax
  %asmtmp.i = tail call i16 asm "movw\09%gs:${1:a}, ${0:w}", "=r,ir,~{dirflag},~{fpsr},~{flags}"(i32 6) nounwind ; <i16> [#uses=1]
  %0 = zext i16 %asmtmp.i to i32                  ; <i32> [#uses=1]
  ret i32 %0
}

define zeroext i16 @test2(i32 %address) nounwind {
entry:
; CHECK-LABEL: test2:
; CHECK: movw	%gs:(%eax), %ax
  %asmtmp = tail call i16 asm "movw\09%gs:${1:a}, ${0:w}", "=r,ir,~{dirflag},~{fpsr},~{flags}"(i32 %address) nounwind ; <i16> [#uses=1]
  ret i16 %asmtmp
}

@n = global i32 42                                ; <i32*> [#uses=3]
@y = common global i32 0                          ; <i32*> [#uses=3]

define void @test3() nounwind {
entry:
; CHECK-LABEL: test3:
; CHECK: movl _n, %eax
  call void asm sideeffect "movl ${0:a}, %eax", "ir,~{dirflag},~{fpsr},~{flags},~{eax}"(i32* @n) nounwind
  ret void
}

define void @test4() nounwind {
entry:
; CHECK-LABEL: test4:
; CHECK: movl	L_y$non_lazy_ptr, %ecx
; CHECK: movl (%ecx), %eax
  call void asm sideeffect "movl ${0:a}, %eax", "ir,~{dirflag},~{fpsr},~{flags},~{eax}"(i32* @y) nounwind
  ret void
}
