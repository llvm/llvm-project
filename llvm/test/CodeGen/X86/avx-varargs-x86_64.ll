; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; <rdar://problem/10463281>
; Check that the <8 x float> is passed on the stack.

@x = common global <8 x float> zeroinitializer, align 32
declare i32 @f(i32, ...)

; CHECK-LABEL: test1:
; CHECK: vmovaps	%ymm0, (%rsp)
define void @test1() nounwind uwtable ssp {
entry:
  %0 = load <8 x float>, ptr @x, align 32
  %call = call i32 (i32, ...) @f(i32 1, <8 x float> %0)
  ret void
}
