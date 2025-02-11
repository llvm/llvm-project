; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-prec-divf32=0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-prec-divf32=0 | %ptxas-verify %}

define float @foo(float %a) {
; CHECK: div.approx.f32
  %div = fdiv float %a, 13.0
  ret float %div
}

