; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate sp floating point add in V5.

; CHECK: r{{[0-9]+}} = sfadd(r{{[0-9]+}},r{{[0-9]+}})

define i32 @main() nounwind {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  store volatile float 0x402ECCCCC0000000, ptr %a, align 4
  store volatile float 0x4022333340000000, ptr %b, align 4
  %0 = load volatile float, ptr %a, align 4
  %1 = load volatile float, ptr %b, align 4
  %add = fadd float %0, %1
  store float %add, ptr %c, align 4
  ret i32 0
}
