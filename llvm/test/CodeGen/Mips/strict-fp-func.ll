; RUN: llc -mtriple=mips -stop-after=finalize-isel %s -o - | FileCheck %s

define float @func_02(float %x, float %y) strictfp nounwind {
; CHECK-LABEL: name: func_02
; CHECK:       JAL @func_01, {{.*}}, implicit-def $fcr31
  %call = call float @func_01(float %x) strictfp
  %res = call float @llvm.experimental.constrained.fadd.f32(float %call, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  ret float %res
}

declare float @func_01(float)
