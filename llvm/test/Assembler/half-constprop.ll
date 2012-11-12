; RUN: llvm-as < %s | opt -O3 | llvm-dis | FileCheck %s
; Testing half constant propagation.

define half @abc() nounwind {
entry:
  %a = alloca half, align 2
  %b = alloca half, align 2
  %.compoundliteral = alloca float, align 4
  store half 0xH4200, half* %a, align 2
  store half 0xH4B9A, half* %b, align 2
  %tmp = load half* %a, align 2
  %tmp1 = load half* %b, align 2
  %add = fadd half %tmp, %tmp1
; CHECK: 0xH4C8D
  ret half %add
}

