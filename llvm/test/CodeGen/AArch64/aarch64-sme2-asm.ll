; RUN: llc < %s -mtriple aarch64-none-linux-gnu -stop-after=finalize-isel | FileCheck %s

define void @UphPNR(target("aarch64.svcount") %predcnt) "target-features"="+sme2" "aarch64_pstate_sm_enabled" {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store (<vscale x 1 x s16>) into %ir.predcnt.addr)
; CHECK:  %1:pnr_p8to15 = COPY %0
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", 1 /* sideeffect attdialect */, {{[0-9]+}} /* reguse:PNR_p8to15 */, %1
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
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", 1 /* sideeffect attdialect */, {{[0-9]+}} /* reguse:PNR */, %1
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
; CHECK:  INLINEASM &"fadd z0.h, $0/m, z0.h, #0.5", 1 /* sideeffect attdialect */, {{[0-9]+}} /* reguse:PNR_3b */, %1
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
; CHECK-SAME: implicit-def early-clobber $q0
; CHECK-SAME: implicit-def early-clobber $q1
; CHECK-SAME: implicit-def early-clobber $q2
; CHECK-SAME: implicit-def early-clobber $q3
; CHECK-SAME: implicit-def early-clobber $q4
; CHECK-SAME: implicit-def early-clobber $q5
; CHECK-SAME: implicit-def early-clobber $q6
; CHECK-SAME: implicit-def early-clobber $q7
; CHECK-SAME: implicit-def early-clobber $q8
; CHECK-SAME: implicit-def early-clobber $q9
; CHECK-SAME: implicit-def early-clobber $q10
; CHECK-SAME: implicit-def early-clobber $q11
; CHECK-SAME: implicit-def early-clobber $q12
; CHECK-SAME: implicit-def early-clobber $q13
; CHECK-SAME: implicit-def early-clobber $q14
; CHECK-SAME: implicit-def early-clobber $q15
; CHECK-SAME: implicit-def early-clobber $q16
; CHECK-SAME: implicit-def early-clobber $q17
; CHECK-SAME: implicit-def early-clobber $q18
; CHECK-SAME: implicit-def early-clobber $q19
; CHECK-SAME: implicit-def early-clobber $q20
; CHECK-SAME: implicit-def early-clobber $q21
; CHECK-SAME: implicit-def early-clobber $q22
; CHECK-SAME: implicit-def early-clobber $q23
; CHECK-SAME: implicit-def early-clobber $q24
; CHECK-SAME: implicit-def early-clobber $q25
; CHECK-SAME: implicit-def early-clobber $q26
; CHECK-SAME: implicit-def early-clobber $q27
; CHECK-SAME: implicit-def early-clobber $q28
; CHECK-SAME: implicit-def early-clobber $q29
; CHECK-SAME: implicit-def early-clobber $q30
; CHECK-SAME: implicit-def early-clobber $q31
  %0 = load <2 x float>, ptr %in, align 8
  call void asm sideeffect "smstart sm; smstop sm;", "~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31}"()
  ret <2 x float> %0
}

define <2 x float> @sme_nosve_streaming(ptr %in) "target-features"="+sme,-sve" "aarch64_pstate_sm_enabled" {
entry:
; CHECK-LABEL: name: sme_nosve_streaming
; CHECK:  INLINEASM &"smstart sm; smstop sm;"
; CHECK-SAME: implicit-def early-clobber $z0
; CHECK-SAME: implicit-def early-clobber $z1
; CHECK-SAME: implicit-def early-clobber $z2
; CHECK-SAME: implicit-def early-clobber $z3
; CHECK-SAME: implicit-def early-clobber $z4
; CHECK-SAME: implicit-def early-clobber $z5
; CHECK-SAME: implicit-def early-clobber $z6
; CHECK-SAME: implicit-def early-clobber $z7
; CHECK-SAME: implicit-def early-clobber $z8
; CHECK-SAME: implicit-def early-clobber $z9
; CHECK-SAME: implicit-def early-clobber $z10
; CHECK-SAME: implicit-def early-clobber $z11
; CHECK-SAME: implicit-def early-clobber $z12
; CHECK-SAME: implicit-def early-clobber $z13
; CHECK-SAME: implicit-def early-clobber $z14
; CHECK-SAME: implicit-def early-clobber $z15
; CHECK-SAME: implicit-def early-clobber $z16
; CHECK-SAME: implicit-def early-clobber $z17
; CHECK-SAME: implicit-def early-clobber $z18
; CHECK-SAME: implicit-def early-clobber $z19
; CHECK-SAME: implicit-def early-clobber $z20
; CHECK-SAME: implicit-def early-clobber $z21
; CHECK-SAME: implicit-def early-clobber $z22
; CHECK-SAME: implicit-def early-clobber $z23
; CHECK-SAME: implicit-def early-clobber $z24
; CHECK-SAME: implicit-def early-clobber $z25
; CHECK-SAME: implicit-def early-clobber $z26
; CHECK-SAME: implicit-def early-clobber $z27
; CHECK-SAME: implicit-def early-clobber $z28
; CHECK-SAME: implicit-def early-clobber $z29
; CHECK-SAME: implicit-def early-clobber $z30
; CHECK-SAME: implicit-def early-clobber $z31
  %0 = load <2 x float>, ptr %in, align 8
  call void asm sideeffect "smstart sm; smstop sm;", "~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31}"()
  ret <2 x float> %0
}
