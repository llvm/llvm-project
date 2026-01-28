; RUN: llc -mtriple arm-none-eabi -stop-after=finalize-isel %s -o - | FileCheck %s

define float @func_02(float %x, float %y) strictfp nounwind {
  %call = call float @func_01(float %x) strictfp
  %res = call float @llvm.experimental.constrained.fadd.f32(float %call, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  ret float %res
}
; CHECK-LABEL: name: func_02
; CHECK:       BL @func_01, {{.*}}, implicit-def $fpscr_rm


declare float @func_01(float)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
