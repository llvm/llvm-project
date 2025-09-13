; RUN: llc -mtriple arm-- -mattr=+vfp4 -stop-after=finalize-isel %s -o - | FileCheck %s

define float @func_02(float %x, float %y) strictfp nounwind {
  %call = call float @func_01(float %x) strictfp
  %res = call float @llvm.experimental.constrained.fadd.f32(float %call, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  ret float %res
}
; CHECK-LABEL: func_02
; CHECK:       BL @func_01, {{.*}}, implicit-def $fpscr


declare float @func_01(float)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)

