; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+sve,+f64mm -asm-verbose=0 < %s -o - 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 2 x double> @fmmla_d(<vscale x 2 x double> %r, <vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
entry:
; CHECK-LABEL: fmmla_d:
; CHECK-NEXT:  fmmla   z0.d, z1.d, z2.d
; CHECK-NEXT:  ret
  %val = tail call <vscale x 2 x double> @llvm.aarch64.sve.fmmla.nxv2f64(<vscale x 2 x double> %r, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %val
}

declare <vscale x 2 x double> @llvm.aarch64.sve.fmmla.nxv2f64(<vscale x 2 x double>,<vscale x 2 x double>,<vscale x 2 x double>)

