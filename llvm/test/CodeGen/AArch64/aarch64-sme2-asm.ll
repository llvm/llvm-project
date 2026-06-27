; RUN: llc < %s -mtriple aarch64-none-linux-gnu -stop-after=finalize-isel | FileCheck %s

define void @UphPNR(target("aarch64.svcount") %predcnt) "target-features"="+sme2" "aarch64_pstate_sm_enabled" {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store (<vscale x 1 x s16>) into %ir.predcnt.addr)
; CHECK:  %1:pnr_p8to15 = COPY %0
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", sideeffect attdialect, reguse:PNR_p8to15, %1
; CHECK:  RET_ReallyLR
  %predcnt.addr = alloca target("aarch64.svcount"), align 2
  store target("aarch64.svcount") %predcnt, ptr %predcnt.addr, align 2
  %0 = load target("aarch64.svcount"), ptr %predcnt.addr, align 2
  call void asm sideeffect "ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", "@3Uph"(target("aarch64.svcount") %0)
  ret void
}

define void @UpaPNR(target("aarch64.svcount") %predcnt) "target-features"="+sme2" "aarch64_pstate_sm_enabled" {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store (<vscale x 1 x s16>) into %ir.predcnt.addr)
; CHECK:  %1:pnr = COPY %0
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", sideeffect attdialect, reguse:PNR, %1
; CHECK:  RET_ReallyLR
  %predcnt.addr = alloca target("aarch64.svcount"), align 2
  store target("aarch64.svcount") %predcnt, ptr %predcnt.addr, align 2
  %0 = load target("aarch64.svcount"), ptr %predcnt.addr, align 2
  call void asm sideeffect "ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", "@3Upa"(target("aarch64.svcount") %0)
  ret void
}

define void @UplPNR(target("aarch64.svcount") %predcnt) "target-features"="+sme2" "aarch64_pstate_sm_enabled" {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store (<vscale x 1 x s16>) into %ir.predcnt.addr)
; CHECK:  %1:pnr_3b = COPY %0
; CHECK:  INLINEASM &"fadd z0.h, $0/m, z0.h, #0.5", sideeffect attdialect, reguse:PNR_3b, %1
; CHECK:  RET_ReallyLR
  %predcnt.addr = alloca target("aarch64.svcount"), align 2
  store target("aarch64.svcount") %predcnt, ptr %predcnt.addr, align 2
  %0 = load target("aarch64.svcount"), ptr %predcnt.addr, align 2
  call void asm sideeffect "fadd z0.h, $0/m, z0.h, #0.5", "@3Upl"(target("aarch64.svcount") %0)
  ret void
}

; Test that the z-register clobbers result in preserving %0 across the inline asm call.
define <2 x float> @sme_nosve_nonstreaming(ptr %in) "target-features"="+sme,-sve" {
entry:
; CHECK-LABEL: name: sme_nosve_nonstreaming
; CHECK:  INLINEASM &"smstart sm; smstop sm;"
; CHECK-SAME: implicit-def $q0
; CHECK-SAME: implicit-def $q1
; CHECK-SAME: implicit-def $q2
; CHECK-SAME: implicit-def $q3
; CHECK-SAME: implicit-def $q4
; CHECK-SAME: implicit-def $q5
; CHECK-SAME: implicit-def $q6
; CHECK-SAME: implicit-def $q7
; CHECK-SAME: implicit-def $q8
; CHECK-SAME: implicit-def $q9
; CHECK-SAME: implicit-def $q10
; CHECK-SAME: implicit-def $q11
; CHECK-SAME: implicit-def $q12
; CHECK-SAME: implicit-def $q13
; CHECK-SAME: implicit-def $q14
; CHECK-SAME: implicit-def $q15
; CHECK-SAME: implicit-def $q16
; CHECK-SAME: implicit-def $q17
; CHECK-SAME: implicit-def $q18
; CHECK-SAME: implicit-def $q19
; CHECK-SAME: implicit-def $q20
; CHECK-SAME: implicit-def $q21
; CHECK-SAME: implicit-def $q22
; CHECK-SAME: implicit-def $q23
; CHECK-SAME: implicit-def $q24
; CHECK-SAME: implicit-def $q25
; CHECK-SAME: implicit-def $q26
; CHECK-SAME: implicit-def $q27
; CHECK-SAME: implicit-def $q28
; CHECK-SAME: implicit-def $q29
; CHECK-SAME: implicit-def $q30
; CHECK-SAME: implicit-def $q31
  %0 = load <2 x float>, ptr %in, align 8
  call void asm sideeffect "smstart sm; smstop sm;", "~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31}"()
  ret <2 x float> %0
}

define <2 x float> @sme_nosve_streaming(ptr %in) "target-features"="+sme,-sve" "aarch64_pstate_sm_enabled" {
entry:
; CHECK-LABEL: name: sme_nosve_streaming
; CHECK:  INLINEASM &"smstart sm; smstop sm;"
; CHECK-SAME: implicit-def $z0
; CHECK-SAME: implicit-def $z1
; CHECK-SAME: implicit-def $z2
; CHECK-SAME: implicit-def $z3
; CHECK-SAME: implicit-def $z4
; CHECK-SAME: implicit-def $z5
; CHECK-SAME: implicit-def $z6
; CHECK-SAME: implicit-def $z7
; CHECK-SAME: implicit-def $z8
; CHECK-SAME: implicit-def $z9
; CHECK-SAME: implicit-def $z10
; CHECK-SAME: implicit-def $z11
; CHECK-SAME: implicit-def $z12
; CHECK-SAME: implicit-def $z13
; CHECK-SAME: implicit-def $z14
; CHECK-SAME: implicit-def $z15
; CHECK-SAME: implicit-def $z16
; CHECK-SAME: implicit-def $z17
; CHECK-SAME: implicit-def $z18
; CHECK-SAME: implicit-def $z19
; CHECK-SAME: implicit-def $z20
; CHECK-SAME: implicit-def $z21
; CHECK-SAME: implicit-def $z22
; CHECK-SAME: implicit-def $z23
; CHECK-SAME: implicit-def $z24
; CHECK-SAME: implicit-def $z25
; CHECK-SAME: implicit-def $z26
; CHECK-SAME: implicit-def $z27
; CHECK-SAME: implicit-def $z28
; CHECK-SAME: implicit-def $z29
; CHECK-SAME: implicit-def $z30
; CHECK-SAME: implicit-def $z31
  %0 = load <2 x float>, ptr %in, align 8
  call void asm sideeffect "smstart sm; smstop sm;", "~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31}"()
  ret <2 x float> %0
}
