; RUN: opt < %s -O3 -S | FileCheck %s
; RUN: verify-uselistorder %s
; Testing half constant propagation.

define half @abc() nounwind {
entry:
  %a = alloca half, align 2
  %b = alloca half, align 2
  %.compoundliteral = alloca float, align 4
  store half 0xH4200, ptr %a, align 2
  store half 0xH4B9A, ptr %b, align 2
  %tmp = load half, ptr %a, align 2
  %tmp1 = load half, ptr %b, align 2
  %add = fadd half %tmp, %tmp1
; CHECK: 1.820310e+01
  ret half %add
}

