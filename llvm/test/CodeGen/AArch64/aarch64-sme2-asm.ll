; RUN: llc < %s -mtriple aarch64-none-linux-gnu -mattr=+sme2 -stop-after=finalize-isel | FileCheck %s

define dso_local void @UphPNR(target("aarch64.svcount") %predcnt) {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store unknown-size into %ir.predcnt.addr, align 2)
; CHECK:  %1:pnr_p8to15 = COPY %0
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", 1 /* sideeffect attdialect */, 393225 /* reguse:PNR_p8to15 */, %1
; CHECK:  RET_ReallyLR
  %predcnt.addr = alloca target("aarch64.svcount"), align 2
  store target("aarch64.svcount") %predcnt, ptr %predcnt.addr, align 2
  %0 = load target("aarch64.svcount"), ptr %predcnt.addr, align 2
  call void asm sideeffect "ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", "@3Uph"(target("aarch64.svcount") %0)
  ret void
}

define dso_local void @UpaPNR(target("aarch64.svcount") %predcnt) {
entry:
; CHECK:  %0:ppr = COPY $p0
; CHECK:  STR_PXI %0, %stack.0.predcnt.addr, 0 :: (store unknown-size into %ir.predcnt.addr, align 2)
; CHECK:  %1:pnr = COPY %0
; CHECK:  INLINEASM &"ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", 1 /* sideeffect attdialect */, 262153 /* reguse:PNR */, %1
; CHECK:  RET_ReallyLR
  %predcnt.addr = alloca target("aarch64.svcount"), align 2
  store target("aarch64.svcount") %predcnt, ptr %predcnt.addr, align 2
  %0 = load target("aarch64.svcount"), ptr %predcnt.addr, align 2
  call void asm sideeffect "ld1w {z0.s,z1.s,z2.s,z3.s}, $0/z, [x10]", "@3Upa"(target("aarch64.svcount") %0)
  ret void
}