; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define float @t1(i1 %a, float %b, float %c) nounwind {
; CHECK: t1
; CHECK: fcsel	s0, s0, s1, ne
  %sel = select i1 %a, float %b, float %c
  ret float %sel
}
