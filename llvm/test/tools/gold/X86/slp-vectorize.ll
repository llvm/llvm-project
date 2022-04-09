; RUN: llvm-as %s -o %t.o

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o.0.4.opt.bc -o - | FileCheck %s

; test that the vectorizer is run.
; CHECK: fadd <4 x float>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(float* nocapture %x) {
  %tmp = load float, float* %x, align 4
  %add = fadd float %tmp, 1.000000e+00
  store float %add, float* %x, align 4
  %arrayidx1 = getelementptr inbounds float, float* %x, i64 1
  %tmp1 = load float, float* %arrayidx1, align 4
  %add2 = fadd float %tmp1, 1.000000e+00
  store float %add2, float* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds float, float* %x, i64 2
  %tmp2 = load float, float* %arrayidx3, align 4
  %add4 = fadd float %tmp2, 1.000000e+00
  store float %add4, float* %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, float* %x, i64 3
  %tmp3 = load float, float* %arrayidx5, align 4
  %add6 = fadd float %tmp3, 1.000000e+00
  store float %add6, float* %arrayidx5, align 4
  ret void
}
