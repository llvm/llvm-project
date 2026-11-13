; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=prolog-epilog < %s | FileCheck %s

; Check that when HLX caller calls a scalar callee with hlxcc attribute,
; caller uses L12 register for a live value across the call.
; CHECK-LABEL: name: foo0
; CHECK: $l12 = LX_L16b_pseudo
; CHECK-NOT: PS_storerl
; CHECK: J2_call @bar0
define inreg <1024 x i16> @foo0(ptr readonly %ptr) {
entry:
  %0 = load <1024 x i16>, ptr %ptr, align 2048
  %call = tail call hexagon_hlxcc ptr @bar0(ptr nonnull %ptr)
  %1 = load <1024 x i16>, ptr %call, align 2048
  %add = add <1024 x i16> %1, %0
  ret <1024 x i16> %add
}

; Check that HLX spills are inserted when scalar caller with hlxcc attribute
; calls a scalar callee
; CHECK-LABEL: name: foo1
; CHECK: PS_storerl {{.*}}$l12
; CHECK: PS_storerl {{.*}}$l13
; CHECK: PS_storerl {{.*}}$l14
; CHECK: PS_storerl {{.*}}$l15
; CHECK: J2_call @bar1, hexagoncsr,
; CHECK: $l12 = PS_loadrl
; CHECK: $l13 = PS_loadrl
; CHECK: $l14 = PS_loadrl
; CHECK: $l15 = PS_loadrl
define hexagon_hlxcc void @foo1(ptr readonly %ptr, ptr writeonly %dst) {
entry:
  %0 = load <1024 x i16>, ptr %ptr, align 2048
  %call = tail call ptr @bar1(ptr nonnull %ptr)
  %1 = load <1024 x i16>, ptr %call, align 2048
  %add = add <1024 x i16> %1, %0
  store <1024 x i16> %add, ptr %dst, align 2048
  ret void
}

declare hexagon_hlxcc ptr @bar0(ptr)

declare ptr @bar1(ptr)
