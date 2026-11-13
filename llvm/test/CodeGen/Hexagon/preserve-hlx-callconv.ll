; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=prolog-epilog < %s | FileCheck %s

; Check that HLX spills are not inserted when a HLX caller calls a scalar callee
; with preserve_hlx attribute.
; CHECK-LABEL: name: foo0
; CHECK-NOT: PS_storerll
; CHECK-NOT: PS_storerl
define inreg <1024 x i16> @foo0(ptr readonly %ptr) {
entry:
  %0 = load <1024 x i16>, ptr %ptr, align 2048
  %call = tail call hexagon_preserve_hlx ptr @bar0(ptr nonnull %ptr)
  %1 = load <1024 x i16>, ptr %call, align 2048
  %add = add <1024 x i16> %1, %0
  ret <1024 x i16> %add
}

; Check that HLX spills are inserted when scalar callee does not have preserve_hlx
; attribute.
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
define inreg <1024 x i16> @foo1(ptr readonly %ptr) {
entry:
  %0 = load <1024 x i16>, ptr %ptr, align 2048
  %call = tail call ptr @bar1(ptr nonnull %ptr)
  %1 = load <1024 x i16>, ptr %call, align 2048
  %add = add <1024 x i16> %1, %0
  ret <1024 x i16> %add
}

declare hexagon_preserve_hlx ptr @bar0(ptr)

declare ptr @bar1(ptr)
