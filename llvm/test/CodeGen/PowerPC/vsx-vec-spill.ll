; RUN: llc < %s -mtriple=ppc64-unknown-linux-gnu -mattr=+vsx \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=VSX
; RUN: llc < %s -mtriple=ppc64-unknown-linux-gnu -mattr=-vsx \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=NOVSX

define <2 x double> @interleaving_VSX_VMX(
  <2 x double> %a, <2 x double> %b, <2 x double> %c,
  <2 x double> %d, <2 x double> %e, <2 x double> %f) {
entry:
  tail call void asm sideeffect "# clobbers",
    "~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() nounwind
  tail call void @goo(<2 x double> %a) nounwind
  %add = fadd <2 x double> %a, %b
  %sub = fsub <2 x double> %a, %b
  %mul = fmul <2 x double> %add, %sub
  %add1 = fadd <2 x double> %c, %d
  %sub2 = fsub <2 x double> %c, %d
  %mul3 = fmul <2 x double> %add1, %sub2
  %add4 = fadd <2 x double> %mul, %mul3
  %add5 = fadd <2 x double> %e, %f
  %sub6 = fsub <2 x double> %e, %f
  %mul7 = fmul <2 x double> %add5, %sub6
  %add8 = fadd <2 x double> %add4, %mul7
  ret <2 x double> %add8
; VSX-LABEL: interleaving_VSX_VMX
; VSX-NOT: stvx
; VSX-NOT: lvx

; NOVSX-LABEL: interleaving_VSX_VMX
; NOVSX-NOT: stxvd2x
; NOVSX-NOT: lxvd2x
}

declare void @goo(<2 x double>)
