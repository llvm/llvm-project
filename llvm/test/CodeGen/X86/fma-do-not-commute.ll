; RUN: llc -fp-contract=fast -mattr=+fma -disable-cgp < %s -o - | FileCheck %s
; Check that the 2nd and 3rd arguments of fmaXXX231 reg1, reg2, mem3 are not commuted.
; <rdar://problem/16800495> 
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; CHECK-LABEL: test1:
; %arg lives in xmm0 and it shouldn't be redefined until it is used in the FMA.
; CHECK-NOT: {{.*}}, %xmm0
; %addr lives in rdi.
; %addr2 lives in rsi.
; CHECK: vmovss (%rdi), [[ADDR:%xmm[0-9]+]]
; The assembly syntax is in the reverse order.
; CHECK: vfmadd231ss (%rsi), [[ADDR]], %xmm0
define void @test1(float* %addr, float* %addr2, float %arg) {
entry:
  br label %loop

loop:
  %sum0 = phi float [ %fma, %loop ], [ %arg, %entry ]
  %addrVal = load float, float* %addr, align 4
  %addr2Val = load float, float* %addr2, align 4
  %fmul = fmul float %addrVal, %addr2Val
  %fma = fadd float %sum0, %fmul
  br i1 true, label %exit, label %loop

exit:
  store float %fma, float* %addr, align 4
  ret void
}
