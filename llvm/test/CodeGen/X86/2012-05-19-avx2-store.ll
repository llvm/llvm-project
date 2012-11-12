; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx2 | FileCheck %s

define void @double_save(<4 x i32>* %Ap, <4 x i32>* %Bp, <8 x i32>* %P) nounwind ssp {
entry:
  ; CHECK: vmovaps
  ; CHECK: vinsertf128 $1, ([[A0:%rdi|%rsi]]),
  ; CHECK: vmovups
  %A = load <4 x i32>* %Ap
  %B = load <4 x i32>* %Bp
  %Z = shufflevector <4 x i32>%A, <4 x i32>%B, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %Z, <8 x i32>* %P, align 16
  ret void
}
