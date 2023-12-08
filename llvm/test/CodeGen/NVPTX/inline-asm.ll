; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

define float @test(float %x) {
entry:
; CHECK: ex2.approx.ftz.f32 %f{{[0-9]+}}, %f{{[0-9]+}}
  %0 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=f,f"(float %x)
  ret float %0
}

define i32 @foo(i1 signext %cond, i32 %a, i32 %b) #0 {
entry:
; CHECK: selp.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %p{{[0-9]+}}
  %0 = tail call i32 asm "selp.b32 $0, $1, $2, $3;", "=r,r,r,b"(i32 %a, i32 %b, i1 %cond)
  ret i32 %0
}
