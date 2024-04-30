; RUN: llc < %s -mtriple aarch64-none-linux-gnu -mattr=+sme2 -stop-after=finalize-isel | FileCheck %s

define void @UphPNR(target("aarch64.svcount") %predcnt) {
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

define void @UpaPNR(target("aarch64.svcount") %predcnt) {
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

define void @UplPNR(target("aarch64.svcount") %predcnt) {
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
