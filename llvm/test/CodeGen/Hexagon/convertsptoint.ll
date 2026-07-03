; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; Check that we generate conversion from single precision floating point
; to 32-bit int value in IEEE complaint mode in V5.

; CHECK: r{{[0-9]+}} = convert_sf2w(r{{[0-9]+}}):chop

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  store i32 0, ptr %retval
  store float 0x402ECCCCC0000000, ptr %a, align 4
  store float 0x4022333340000000, ptr %b, align 4
  %0 = load float, ptr %a, align 4
  %1 = load float, ptr %b, align 4
  %add = fadd float %0, %1
  store volatile float %add, ptr %c, align 4
  %2 = load volatile float, ptr %c, align 4
  %conv = fptosi float %2 to i32
  store i32 %conv, ptr %i, align 4
  %3 = load i32, ptr %i, align 4
  ret i32 %3
}
