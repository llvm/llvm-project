; RUN: llvm-as < %s | opt -O3 | llvm-dis | FileCheck %s
; Testing half to float conversion.

define float @abc() nounwind {
entry:
  %a = alloca half, align 2
  %.compoundliteral = alloca float, align 4
  store half 0xH4C8D, half* %a, align 2
  %tmp = load half* %a, align 2
  %conv = fpext half %tmp to float
; CHECK: 0x4032340000000000
  ret float %conv
}
