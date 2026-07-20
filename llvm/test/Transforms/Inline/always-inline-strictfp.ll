; RUN: opt -passes=always-inline,verify %s -S | FileCheck %s

define internal <16 x half> @convert_rtp(<16 x double> %x) #0 {
entry:
  %r = tail call <16 x half> @llvm.experimental.constrained.fptrunc.v16f16.v16f64(<16 x double> %x, metadata !"round.upward", metadata !"fpexcept.ignore") #1
  ret <16 x half> %r
}

; CHECK: define <16 x half> @spirv_fconvert_rtp(<16 x double> %x) [[ATTR:#[0-9]+]]
; CHECK: call <16 x half> @llvm.experimental.constrained.fptrunc.v16f16.v16f64({{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
define <16 x half> @spirv_fconvert_rtp(<16 x double> %x) {
entry:
  %c = call <16 x half> @convert_rtp(<16 x double> %x)
  ret <16 x half> %c
}

; CHECK: attributes [[ATTR]] = {{{.*}}strictfp{{.*}}}

declare <16 x half> @llvm.experimental.constrained.fptrunc.v16f16.v16f64(<16 x double>, metadata, metadata)

attributes #0 = { alwaysinline strictfp }
attributes #1 = { strictfp }
