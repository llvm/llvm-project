; NOTE: This test ensures that for both Big and Little Endian cases a set of
; NOTE: 4 floats is gathered into a v4f32 register using xxmrghd, xvcvdpsp,
; NOTE: and vmrgew.
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -ppc-vsr-nums-as-vr \
; RUN: -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN: | FileCheck %s -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -ppc-vsr-nums-as-vr \
; RUN: -ppc-asm-full-reg-names -mtriple=powerpc64-unknown-linux-gnu < %s \
; RUN: | FileCheck %s -check-prefix=CHECK-BE
define dso_local <4 x float> @vector_gatherf(float* nocapture readonly %a,
float* nocapture readonly %b, float* nocapture readonly %c,
float* nocapture readonly %d) {
; C code from which this IR test case was generated:
; vector float test(float *a, float *b, float *c, float *d) {
;  return (vector float) { *a, *b, *c, *d };
; }
; CHECK-LE-LABEL: vector_gatherf:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-DAG:    lfs f[[REG0:[0-9]+]], 0(r3)
; CHECK-LE-DAG:    lfs f[[REG1:[0-9]+]], 0(r4)
; CHECK-LE-DAG:    lfs f[[REG2:[0-9]+]], 0(r5)
; CHECK-LE-DAG:    lfs f[[REG3:[0-9]+]], 0(r6)
; CHECK-LE-DAG:    xxmrghd vs[[REG4:[0-9]+]], vs[[REG2]], vs[[REG0]]
; CHECK-LE-NEXT:   xvcvdpsp v[[VREG2:[0-9]+]], vs[[REG4]]
; CHECK-LE-NEXT:   xxmrghd vs[[REG5:[0-9]+]], vs[[REG3]], vs[[REG1]]
; CHECK-LE-NEXT:   xvcvdpsp v[[VREG3:[0-9]+]], vs[[REG5]]
; CHECK-LE-NEXT:   vmrgew v[[VREG:[0-9]+]], v[[VREG3]], v[[VREG2]]
; CHECK-LE-NEXT:   blr

; CHECK-BE-LABEL: vector_gatherf:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-DAG:    lfs f[[REG0:[0-9]+]], 0(r3)
; CHECK-BE-DAG:    lfs f[[REG1:[0-9]+]], 0(r4)
; CHECK-BE-DAG:    lfs f[[REG2:[0-9]+]], 0(r5)
; CHECK-BE-DAG:    lfs f[[REG3:[0-9]+]], 0(r6)
; CHECK-BE-DAG:    xxmrghd vs[[REG4:[0-9]+]], vs[[REG0]], vs[[REG2]]
; CHECK-BE-DAG:    xxmrghd vs[[REG5:[0-9]+]], vs[[REG1]], vs[[REG3]]
; CHECK-BE-NEXT:   xvcvdpsp v[[VREG2:[0-9]+]], vs[[REG5]]
; CHECK-BE-NEXT:   xvcvdpsp v[[VREG3:[0-9]+]], vs[[REG4]]
; CHECK-BE-NEXT:   vmrgew v[[VREG:[0-9]+]], v[[VREG3]], v[[VREG2]]
; CHECK-BE-NEXT:   blr
entry:
  %0 = load float, float* %a, align 4
  %vecinit = insertelement <4 x float> undef, float %0, i32 0
  %1 = load float, float* %b, align 4
  %vecinit1 = insertelement <4 x float> %vecinit, float %1, i32 1
  %2 = load float, float* %c, align 4
  %vecinit2 = insertelement <4 x float> %vecinit1, float %2, i32 2
  %3 = load float, float* %d, align 4
  %vecinit3 = insertelement <4 x float> %vecinit2, float %3, i32 3
  ret <4 x float> %vecinit3
}

