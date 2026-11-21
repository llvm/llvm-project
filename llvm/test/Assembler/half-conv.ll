; RUN: opt < %s -O3 -S | FileCheck %s
; RUN: verify-uselistorder %s
; Testing half to float conversion.

define float @abc() nounwind {
entry:
  %a = alloca half, align 2
  %.compoundliteral = alloca float, align 4
  store half 0xH4C8D, ptr %a, align 2
  %tmp = load half, ptr %a, align 2
  %conv = fpext half %tmp to float
; CHECK: f0x4191A000
  ret float %conv
}
