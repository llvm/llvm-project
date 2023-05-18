; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate conversion from single precision floating point
; to 64-bit int value in IEEE complaint mode in V5.

; CHECK: r{{[0-9]+}}:{{[0-9]+}} = convert_sf2d(r{{[0-9]+}})

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %i = alloca i64, align 8
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
  %conv = fptosi float %2 to i64
  store i64 %conv, ptr %i, align 8
  %3 = load i64, ptr %i, align 8
  %conv1 = trunc i64 %3 to i32
  ret i32 %conv1
}
