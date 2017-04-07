; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,MSA32
; RUN: llc -march=mips64 -mattr=+msa,+fp64 -relocation-model=pic -target-abi n32 < %s \
; RUN:      | FileCheck %s -check-prefixes=CHECK,MSA64,MSA64N32
; RUN: llc -march=mips64 -mattr=+msa,+fp64 -relocation-model=pic -target-abi n64 < %s \
; RUN:      | FileCheck %s -check-prefixes=CHECK,MSA64,MSA64N64

; Test that the immediate intrinsics don't crash LLVM.

; Some of the intrinsics lower to equivalent forms.

define void @addvi_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: addvi_b:
; CHECK: addvi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.addvi.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @andi_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: andi_b:
; CHECK: andi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.andi.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bclri_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: bclri_b:
; CHECK: andi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bclri.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @binsli_b(<16 x i8> * %ptr, <16 x i8> * %ptr2) {
entry:
; CHECK-LABEL: binsli_b:
; CHECK: binsli.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %b = load <16 x i8>, <16 x i8> * %ptr2, align 16
  %r = call <16 x i8> @llvm.mips.binsli.b(<16 x i8> %a, <16 x i8> %b, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @binsri_b(<16 x i8> * %ptr, <16 x i8> * %ptr2) {
entry:
; CHECK-LABEL: binsri_b:
; CHECK: binsri.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %b = load <16 x i8>, <16 x i8> * %ptr2, align 16
  %r = call <16 x i8> @llvm.mips.binsri.b(<16 x i8> %a, <16 x i8> %b, i32 5)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bmnzi_b(<16 x i8> * %ptr, <16 x i8> * %ptr2) {
entry:
; CHECK-LABEL: bmnzi_b:
; CHECK: bmnzi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %b = load <16 x i8>, <16 x i8> * %ptr2, align 16
  %r = call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %a, <16 x i8> %b, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bmzi_b(<16 x i8> * %ptr, <16 x i8> * %ptr2) {
entry:
; CHECK-LABEL: bmzi_b:
; CHECK: bmnzi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %b = load <16 x i8>, <16 x i8> * %ptr2, align 16
  %r = call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %a, <16 x i8> %b, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bnegi_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: bnegi_b:
; CHECK: bnegi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bnegi.b(<16 x i8> %a, i32 6)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bseli_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: bseli_b:
; CHECK: bseli.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseli.b(<16 x i8> %a, <16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @bseti_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: bseti_b:
; CHECK: bseti.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseti.b(<16 x i8> %a, i32 5)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @clei_s_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: clei_s_b:
; CHECK: clei_s.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.s.b(<16 x i8> %a, i32 12)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @clei_u_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: clei_u_b:
; CHECK: clei_u.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.u.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @clti_s_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: clti_s_b:
; CHECK: clti_s.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.s.b(<16 x i8> %a, i32 15)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @clti_u_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: clti_u_b:
; CHECK: clti_u.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.u.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @ldi_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: ldi_b:
; CHECK: ldi.b
  %r = call <16 x i8> @llvm.mips.ldi.b(i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @maxi_s_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: maxi_s_b:
; CHECK: maxi_s.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.s.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @maxi_u_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: maxi_u_b:
; CHECK: maxi_u.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.u.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @mini_s_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: mini_s_b:
; CHECK: mini_s.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.s.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @mini_u_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: mini_u_b:
; CHECK: mini_u.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.u.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @nori_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: nori_b:
; CHECK: nori.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.nori.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @ori_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: ori_b:
; CHECK: ori.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.ori.b(<16 x i8> %a, i32 25)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @sldi_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: sldi_b:
; CHECK: sldi.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.sldi.b(<16 x i8> %a, <16 x i8> %a, i32 7)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @slli_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: slli_b:
; CHECK: slli.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.slli.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @splati_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: splati_b:
; CHECK: splati.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.splati.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @srai_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: srai_b:
; CHECK: srai.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srai.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @srari_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: srari_b:
; CHECK: srari.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srari.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @srli_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: srli_b:
; CHECK: srli.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srli.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @srlri_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: srlri_b:
; CHECK: srlri.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srlri.b(<16 x i8> %a, i32 3)
  store <16 x i8> %r, <16 x i8> * %ptr, align 16
  ret void
}

define void @ld_b(<16 x i8> * %ptr, i8 * %ldptr, i32 %offset) {
entry:
; CHECK-LABEL: ld_b
; MSA32: addu $[[R0:[0-9]]], $5, $6

; MSA64N32-DAG: sll $[[R2:[0-9]]], $6, 0
; MSA64N32-DAG: sll $[[R1:[0-9]]], $5, 0
; MSA64N32: addu $[[R0:[0-9]]], $[[R1]], $[[R2]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $5, $[[R1]]

; CHECK:    ld.b $w{{[0-9]+}}, 0($[[R0]])
  %a = call <16 x i8> @llvm.mips.ld.b(i8* %ldptr, i32 %offset)
  store <16 x i8> %a, <16 x i8> * %ptr, align 16
  ret void
}

define void @st_b(<16 x i8> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
entry:
; CHECK-LABEL: st_b
; MSA32: addu $[[R0:[0-9]]], $7, $6

; MSA64N32: sll $[[R1:[0-9]]], $6, 0
; MSA64N32: sll $[[R2:[0-9]]], $7, 0
; MSA64N32: addu $[[R0:[0-9]]], $[[R2]], $[[R1]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $7, $[[R1]]
; CHECK: st.b $w{{[0-9]+}}, 0($[[R0]])
  %a = call <16 x i8> @llvm.mips.ld.b(i8* %ldptr, i32 0)
  call void @llvm.mips.st.b(<16 x i8> %a, i8* %stptr, i32 %offset)
  ret void
}

define void @addvi_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: addvi_w:
; CHECK: addvi.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.addvi.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @bclri_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: bclri_w:
; CHECK: bclri.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bclri.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @binsli_w(<4 x i32> * %ptr, <4 x i32> * %ptr2) {
entry:
; CHECK-LABEL: binsli_w:
; CHECK: binsli.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %b = load <4 x i32>, <4 x i32> * %ptr2, align 16
  %r = call <4 x i32> @llvm.mips.binsli.w(<4 x i32> %a, <4 x i32> %b, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @binsri_w(<4 x i32> * %ptr, <4 x i32> * %ptr2) {
entry:
; CHECK-LABEL: binsri_w:
; CHECK: binsri.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %b = load <4 x i32>, <4 x i32> * %ptr2, align 16
  %r = call <4 x i32> @llvm.mips.binsri.w(<4 x i32> %a, <4 x i32> %b, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @bnegi_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: bnegi_w:
; CHECK: bnegi.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bnegi.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @bseti_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: bseti_w:
; CHECK: bseti.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bseti.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @clei_s_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: clei_s_w:
; CHECK: clei_s.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.s.w(<4 x i32> %a, i32 14)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @clei_u_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: clei_u_w:
; CHECK: clei_u.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.u.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @clti_s_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: clti_s_w:
; CHECK: clti_s.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.s.w(<4 x i32> %a, i32 15)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @clti_u_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: clti_u_w:
; CHECK: clti_u.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.u.w(<4 x i32> %a, i32 25)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @maxi_s_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: maxi_s_w:
; CHECK: maxi_s.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.s.w(<4 x i32> %a, i32 2)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @maxi_u_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: maxi_u_w:
; CHECK: maxi_u.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.u.w(<4 x i32> %a, i32 2)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @mini_s_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: mini_s_w:
; CHECK: mini_s.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.s.w(<4 x i32> %a, i32 2)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @mini_u_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: mini_u_w:
; CHECK: mini_u.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.u.w(<4 x i32> %a, i32 2)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @ldi_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: ldi_w:
; CHECK: ldi.w
  %r = call <4 x i32> @llvm.mips.ldi.w(i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @sldi_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: sldi_w:
; CHECK: sldi.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.sldi.w(<4 x i32> %a, <4 x i32> %a, i32 2)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @slli_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: slli_w:
; CHECK: slli.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.slli.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @splati_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: splati_w:
; CHECK: splati.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.splati.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @srai_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: srai_w:
; CHECK: srai.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srai.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @srari_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: srari_w:
; CHECK: srari.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srari.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @srli_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: srli_w:
; CHECK: srli.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srli.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @srlri_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: srlri_w:
; CHECK: srlri.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srlri.w(<4 x i32> %a, i32 3)
  store <4 x i32> %r, <4 x i32> * %ptr, align 16
  ret void
}

define void @ld_w(<4 x i32> * %ptr, i8 * %ldptr, i32 %offset) {
entry:
; CHECK-LABEL: ld_w
; MSA32: addu $[[R0:[0-9]]], $5, $6
; MSA64N32: sll $[[R2:[0-9]]], $6, 0
; MSA64N32: sll $[[R1:[0-9]]], $5, 0
; MSA64N32: addu $[[R0:[0-9]]], $[[R1]], $[[R2]]
; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $5, $[[R1]]
; CHECK: ld.w $w{{[0-9]+}}, 0($[[R0]])
  %a = call <4 x i32> @llvm.mips.ld.w(i8* %ldptr, i32 %offset)
  store <4 x i32> %a, <4 x i32> * %ptr, align 16
  ret void
}

define void @st_w(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
entry:
; CHECK-LABEL: st_w
; MSA32: addu $[[R0:[0-9]]], $7, $6

; MSA64N32: sll $[[R1:[0-9]+]], $6, 0
; MSA64N32: sll $[[R2:[0-9]+]], $7, 0
; MSA64N32: addu $[[R0:[0-9]+]], $[[R2]], $[[R1]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $7, $[[R1]]
; CHECK: st.w $w{{[0-9]+}}, 0($[[R0]])
  %a = call <4 x i32> @llvm.mips.ld.w(i8* %ldptr, i32 0)
  call void @llvm.mips.st.w(<4 x i32> %a, i8* %stptr, i32 %offset)
  ret void
}

define void @addvi_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: addvi_h:
; CHECK: addvi.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.addvi.h(<8 x i16> %a, i32 25)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @bclri_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: bclri_h:
; CHECK: bclri.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bclri.h(<8 x i16> %a, i32 8)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @binsli_h(<8 x i16> * %ptr, <8 x i16> * %ptr2) {
entry:
; CHECK-LABEL: binsli_h:
; CHECK: binsli.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %b = load <8 x i16>, <8 x i16> * %ptr2, align 16
  %r = call <8 x i16> @llvm.mips.binsli.h(<8 x i16> %a, <8 x i16> %b, i32 8)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @binsri_h(<8 x i16> * %ptr, <8 x i16> * %ptr2) {
entry:
; CHECK-LABEL: binsri_h:
; CHECK: binsri.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %b = load <8 x i16>, <8 x i16> * %ptr2, align 16
  %r = call <8 x i16> @llvm.mips.binsri.h(<8 x i16> %a, <8 x i16> %b, i32 14)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @bnegi_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: bnegi_h:
; CHECK: bnegi.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bnegi.h(<8 x i16> %a, i32 14)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @bseti_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: bseti_h:
; CHECK: bseti.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bseti.h(<8 x i16> %a, i32 15)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @clei_s_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: clei_s_h:
; CHECK: clei_s.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.s.h(<8 x i16> %a, i32 13)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @clei_u_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: clei_u_h:
; CHECK: clei_u.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.u.h(<8 x i16> %a, i32 25)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @clti_s_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: clti_s_h:
; CHECK: clti_s.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.s.h(<8 x i16> %a, i32 15)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @clti_u_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: clti_u_h:
; CHECK: clti_u.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.u.h(<8 x i16> %a, i32 25)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @maxi_s_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: maxi_s_h:
; CHECK: maxi_s.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.s.h(<8 x i16> %a, i32 2)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @maxi_u_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: maxi_u_h:
; CHECK: maxi_u.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.u.h(<8 x i16> %a, i32 2)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @mini_s_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: mini_s_h:
; CHECK: mini_s.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.s.h(<8 x i16> %a, i32 2)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @mini_u_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: mini_u_h:
; CHECK: mini_u.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.u.h(<8 x i16> %a, i32 2)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @ldi_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: ldi_h:
; CHECK: ldi.h
  %r = call <8 x i16> @llvm.mips.ldi.h(i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @sldi_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: sldi_h:
; CHECK: sldi.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.sldi.h(<8 x i16> %a, <8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @slli_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: slli_h:
; CHECK: slli.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.slli.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @splati_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: splati_h:
; CHECK: splati.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.splati.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @srai_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: srai_h:
; CHECK: srai.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srai.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @srari_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: srari_h:
; CHECK: srari.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srari.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @srli_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: srli_h:
; CHECK: srli.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srli.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @srlri_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: srlri_h:
; CHECK: srlri.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srlri.h(<8 x i16> %a, i32 3)
  store <8 x i16> %r, <8 x i16> * %ptr, align 16
  ret void
}

define void @ld_h(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset) {
entry:
; CHECK-LABEL: ld_h
; MSA32: addu $[[R0:[0-9]]], $5, $6

; MSA64N32-DAG: sll $[[R2:[0-9]]], $6, 0
; MSA64N32-DAG: sll $[[R1:[0-9]]], $5, 0
; MSA64N32: addu $[[R0:[0-9]]], $[[R1]], $[[R2]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $5, $[[R1]]

; CHECK:    ld.h $w{{[0-9]+}}, 0($[[R0]])
  %a = call <8 x i16> @llvm.mips.ld.h(i8* %ldptr, i32 %offset)
  store <8 x i16> %a, <8 x i16> * %ptr, align 16
  ret void
}

define void @st_h(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
entry:
; CHECK-LABEL: st_h
; MSA32: addu $[[R0:[0-9]]], $7, $6

; MSA64N32-DAG: sll $[[R1:[0-9]+]], $6, 0
; MSA64N32-DAG: sll $[[R2:[0-9]+]], $7, 0
; MSA64N32: addu $[[R0:[0-9]+]], $[[R2]], $[[R1]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $7, $[[R1]]
; CHECK: st.h $w{{[0-9]+}}, 0($[[R0]])
  %a = call <8 x i16> @llvm.mips.ld.h(i8* %ldptr, i32 0)
  call void @llvm.mips.st.h(<8 x i16> %a, i8* %stptr, i32 %offset)
  ret void
}

define i32 @copy_s_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: copy_s_b:
; CHECK: copy_s.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.b(<16 x i8> %a, i32 1)
  ret i32 %r
}
define i32 @copy_s_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: copy_s_h:
; CHECK: copy_s.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.h(<8 x i16> %a, i32 1)
  ret i32 %r
}
define i32 @copy_s_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: copy_s_w:
; CHECK: copy_s.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.w(<4 x i32> %a, i32 1)
  ret i32 %r
}
define i32 @copy_u_b(<16 x i8> * %ptr) {
entry:
; CHECK-LABEL: copy_u_b:
; CHECK: copy_u.b
  %a = load <16 x i8>, <16 x i8> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.b(<16 x i8> %a, i32 1)
  ret i32 %r
}
define i32 @copy_u_h(<8 x i16> * %ptr) {
entry:
; CHECK-LABEL: copy_u_h:
; CHECK: copy_u.h
  %a = load <8 x i16>, <8 x i16> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.h(<8 x i16> %a, i32 1)
  ret i32 %r
}
define i32 @copy_u_w(<4 x i32> * %ptr) {
entry:
; CHECK-LABEL: copy_u_w:
; MSA32: copy_s.w
; MSA64: copy_u.w
  %a = load <4 x i32>, <4 x i32> * %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.w(<4 x i32> %a, i32 1)
  ret i32 %r
}

define i64 @copy_s_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: copy_s_d:
; MSA32: copy_s.w
; MSA32: copy_s.w
; MSA64: copy_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call i64 @llvm.mips.copy.s.d(<2 x i64> %a, i32 1)
  ret i64 %r
}

define i64 @copy_u_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: copy_u_d:
; MSA32: copy_s.w
; MSA32: copy_s.w
; MSA64: copy_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call i64 @llvm.mips.copy.u.d(<2 x i64> %a, i32 1)
  ret i64 %r
}

define void @addvi_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: addvi_d:
; CHECK: addvi.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.addvi.d(<2 x i64> %a, i32 25)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @bclri_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: bclri_d:
; CHECK: bclri.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bclri.d(<2 x i64> %a, i32 16)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @binsli_d(<2 x i64> * %ptr, <2 x i64> * %ptr2) {
entry:
; CHECK-LABEL: binsli_d:
; CHECK: binsli.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %b = load <2 x i64>, <2 x i64> * %ptr2, align 16
  %r = call <2 x i64> @llvm.mips.binsli.d(<2 x i64> %a, <2 x i64> %b, i32 4)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @binsri_d(<2 x i64> * %ptr, <2 x i64> * %ptr2) {
entry:
; CHECK-LABEL: binsri_d:
; CHECK: binsri.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %b = load <2 x i64>, <2 x i64> * %ptr2, align 16
  %r = call <2 x i64> @llvm.mips.binsri.d(<2 x i64> %a, <2 x i64> %b, i32 5)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @bnegi_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: bnegi_d:
; CHECK: bnegi.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bnegi.d(<2 x i64> %a, i32 9)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @bseti_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: bseti_d:
; CHECK: bseti.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bseti.d(<2 x i64> %a, i32 25)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @clei_s_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: clei_s_d:
; CHECK: clei_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.s.d(<2 x i64> %a, i32 15)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @clei_u_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: clei_u_d:
; CHECK: clei_u.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.u.d(<2 x i64> %a, i32 25)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @clti_s_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: clti_s_d:
; CHECK: clti_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.s.d(<2 x i64> %a, i32 15)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @clti_u_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: clti_u_d:
; CHECK: clti_u.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.u.d(<2 x i64> %a, i32 25)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @ldi_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: ldi_d:
; CHECK: ldi.d
  %r = call <2 x i64> @llvm.mips.ldi.d(i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @maxi_s_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: maxi_s_d:
; CHECK: maxi_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.s.d(<2 x i64> %a, i32 2)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @maxi_u_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: maxi_u_d:
; CHECK: maxi_u.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.u.d(<2 x i64> %a, i32 2)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @mini_s_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: mini_s_d:
; CHECK: mini_s.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.s.d(<2 x i64> %a, i32 2)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @mini_u_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: mini_u_d:
; CHECK: mini_u.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.u.d(<2 x i64> %a, i32 2)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @sldi_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: sldi_d:
; CHECK: sldi.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.sldi.d(<2 x i64> %a, <2 x i64> %a, i32 1)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @slli_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: slli_d:
; CHECK: slli.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.slli.d(<2 x i64> %a, i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @srai_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: srai_d:
; CHECK: srai.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srai.d(<2 x i64> %a, i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @srari_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: srari_d:
; CHECK: srari.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srari.d(<2 x i64> %a, i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @srli_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: srli_d:
; CHECK: srli.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srli.d(<2 x i64> %a, i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @srlri_d(<2 x i64> * %ptr) {
entry:
; CHECK-LABEL: srlri_d:
; CHECK: srlri.d
  %a = load <2 x i64>, <2 x i64> * %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srlri.d(<2 x i64> %a, i32 3)
  store <2 x i64> %r, <2 x i64> * %ptr, align 16
  ret void
}

define void @ld_d(<2 x i64> * %ptr, i8 * %ldptr, i32 %offset) {
entry:
; CHECK-LABEL: ld_d
; MSA32: addu $[[R0:[0-9]]], $5, $6
; MSA64N32: sll $[[R2:[0-9]]], $6, 0
; MSA64N32: sll $[[R1:[0-9]]], $5, 0
; MSA64N32: addu $[[R0:[0-9]]], $[[R1]], $[[R2]]
; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $5, $[[R1]]
; CHECK: ld.d $w{{[0-9]+}}, 0($[[R0]])
  %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 %offset)
  store <2 x i64> %a, <2 x i64> * %ptr, align 16
  ret void
}

define void @ld_d2(<2 x i64> * %ptr, i8 * %ldptr) {
entry:
; CHECK-LABEL: ld_d2
; MSA32: addiu $[[R0:[0-9]]], $5, 4096
; MSA64N32: sll $[[R1:[0-9]]], $5, 0
; MSA64N32: addiu $[[R0:[0-9]]], $[[R1]], 4096
; MSA64N64: daddiu $[[R0:[0-9]]], $5, 4096
; CHECK: ld.d $w{{[0-9]+}}, 0($[[R0]])
  %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 4096)
  store <2 x i64> %a, <2 x i64> * %ptr, align 16
  ret void
}

define void @st_d(<2 x i64> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
entry:
; CHECK-LABEL: st_d
; MSA32: addu $[[R0:[0-9]]], $7, $6

; MSA64N32-DAG: sll $[[R1:[0-9]]], $6, 0
; MSA64N32-DAG: sll $[[R2:[0-9]+]], $7, 0
; MSA64N32: addu $[[R0:[0-9]+]], $[[R2]], $[[R1]]

; MSA64N64: sll $[[R1:[0-9]]], $6, 0
; MSA64N64: daddu $[[R0:[0-9]]], $7, $[[R1]]
; CHECK: st.d $w{{[0-9]+}}, 0($[[R0]])
  %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 0)
  call void @llvm.mips.st.d(<2 x i64> %a, i8* %stptr, i32 %offset)
  ret void
}


declare <8 x i16> @llvm.mips.ldi.h(i32)
declare <8 x i16> @llvm.mips.addvi.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.bclri.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.binsli.h(<8 x i16>, <8 x i16>, i32)
declare <8 x i16> @llvm.mips.binsri.h(<8 x i16>, <8 x i16>, i32)
declare <8 x i16> @llvm.mips.bnegi.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.bseti.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.clei.s.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.clei.u.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.clti.s.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.clti.u.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.maxi.s.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.maxi.u.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.mini.s.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.mini.u.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.sldi.h(<8 x i16>, <8 x i16>, i32)
declare <8 x i16> @llvm.mips.slli.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.splati.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.srai.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.srari.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.srli.h(<8 x i16>, i32)
declare <8 x i16> @llvm.mips.srlri.h(<8 x i16>, i32)
declare <4 x i32> @llvm.mips.addvi.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.bclri.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.binsli.w(<4 x i32>, <4 x i32>, i32)
declare <4 x i32> @llvm.mips.binsri.w(<4 x i32>, <4 x i32>, i32)
declare <4 x i32> @llvm.mips.bnegi.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.bseti.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.ldi.w(i32)
declare <4 x i32> @llvm.mips.clei.s.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.clei.u.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.clti.s.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.clti.u.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.maxi.s.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.maxi.u.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.mini.s.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.mini.u.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.sldi.w(<4 x i32>, <4 x i32>, i32)
declare <4 x i32> @llvm.mips.slli.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.splati.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.srai.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.srari.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.srli.w(<4 x i32>, i32)
declare <4 x i32> @llvm.mips.srlri.w(<4 x i32>, i32)
declare <2 x i64> @llvm.mips.ldi.d(i32)
declare <2 x i64> @llvm.mips.addvi.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.bclri.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.binsli.d(<2 x i64>, <2 x i64>, i32)
declare <2 x i64> @llvm.mips.binsri.d(<2 x i64>, <2 x i64>, i32)
declare <2 x i64> @llvm.mips.bnegi.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.bseti.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.clei.s.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.clei.u.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.clti.s.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.clti.u.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.maxi.s.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.maxi.u.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.mini.s.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.mini.u.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.sldi.d(<2 x i64>, <2 x i64>, i32)
declare <2 x i64> @llvm.mips.slli.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.splati.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.srai.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.srari.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.srli.d(<2 x i64>, i32)
declare <2 x i64> @llvm.mips.srlri.d(<2 x i64>, i32)
declare <16 x i8> @llvm.mips.ldi.b(i32)
declare <16 x i8> @llvm.mips.addvi.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.andi.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.bclri.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.binsli.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.binsri.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.bmnzi.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.bnegi.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.bseli.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.bseti.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.clei.s.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.clei.u.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.clti.s.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.clti.u.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.maxi.s.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.maxi.u.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.mini.s.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.mini.u.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.nori.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.ori.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.sldi.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.slli.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.splati.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.srai.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.srari.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.srli.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.srlri.b(<16 x i8>, i32)
declare i32 @llvm.mips.copy.s.h(<8 x i16>, i32)
declare i32 @llvm.mips.copy.u.h(<8 x i16>, i32)
declare i32 @llvm.mips.copy.s.w(<4 x i32>, i32)
declare i32 @llvm.mips.copy.u.w(<4 x i32>, i32)
declare i64 @llvm.mips.copy.s.d(<2 x i64>, i32)
declare i64 @llvm.mips.copy.u.d(<2 x i64>, i32)
declare i32 @llvm.mips.copy.s.b(<16 x i8>, i32)
declare i32 @llvm.mips.copy.u.b(<16 x i8>, i32)
declare <16 x i8> @llvm.mips.bmzi.b(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.mips.ld.b(i8*, i32)
declare <8 x i16> @llvm.mips.ld.h(i8*, i32)
declare <4 x i32> @llvm.mips.ld.w(i8*, i32)
declare <2 x i64> @llvm.mips.ld.d(i8*, i32)
declare void @llvm.mips.st.b(<16 x i8>, i8*, i32)
declare void @llvm.mips.st.h(<8 x i16>, i8*, i32)
declare void @llvm.mips.st.w(<4 x i32>, i8*, i32)
declare void @llvm.mips.st.d(<2 x i64>, i8*, i32)
