; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme --force-streaming -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

define void @discard_f16(<vscale x 8 x half> %input) strictfp {
; MIR-LABEL: name: discard_f16
; MIR: %{{[0-9]+}}:zpr = FRINTA_ZPmZ_H_UNDEF
; CHECK-LABEL: discard_f16:
; CHECK: frinta {{z[0-9]+}}.h,
; CHECK: ret
  %unused = call <vscale x 8 x half> @llvm.experimental.constrained.round.nxv8f16(<vscale x 8 x half> %input, metadata !"fpexcept.strict") strictfp
  ret void
}

define void @discard_f32(<vscale x 4 x float> %input) strictfp {
; MIR-LABEL: name: discard_f32
; MIR: %{{[0-9]+}}:zpr = FRINTA_ZPmZ_S_UNDEF
; CHECK-LABEL: discard_f32:
; CHECK: frinta {{z[0-9]+}}.s,
; CHECK: ret
  %unused = call <vscale x 4 x float> @llvm.experimental.constrained.round.nxv4f32(<vscale x 4 x float> %input, metadata !"fpexcept.strict") strictfp
  ret void
}

define void @discard_f64(<vscale x 2 x double> %input) strictfp {
; MIR-LABEL: name: discard_f64
; MIR: %{{[0-9]+}}:zpr = FRINTA_ZPmZ_D_UNDEF
; CHECK-LABEL: discard_f64:
; CHECK: frinta {{z[0-9]+}}.d,
; CHECK: ret
  %unused = call <vscale x 2 x double> @llvm.experimental.constrained.round.nxv2f64(<vscale x 2 x double> %input, metadata !"fpexcept.strict") strictfp
  ret void
}

define void @discard_ignore_f32(<vscale x 4 x float> %input) strictfp {
; CHECK-LABEL: discard_ignore_f32:
; CHECK-NOT: frinta
; CHECK: ret
  %unused = call <vscale x 4 x float> @llvm.experimental.constrained.round.nxv4f32(<vscale x 4 x float> %input, metadata !"fpexcept.ignore") strictfp
  ret void
}

define void @discard_ignore_f64(<vscale x 2 x double> %input) strictfp {
; CHECK-LABEL: discard_ignore_f64:
; CHECK-NOT: frinta
; CHECK: ret
  %unused = call <vscale x 2 x double> @llvm.experimental.constrained.round.nxv2f64(<vscale x 2 x double> %input, metadata !"fpexcept.ignore") strictfp
  ret void
}

declare <vscale x 8 x half> @llvm.experimental.constrained.round.nxv8f16(<vscale x 8 x half>, metadata)
declare <vscale x 4 x float> @llvm.experimental.constrained.round.nxv4f32(<vscale x 4 x float>, metadata)
declare <vscale x 2 x double> @llvm.experimental.constrained.round.nxv2f64(<vscale x 2 x double>, metadata)
