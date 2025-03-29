; RUN: not --crash llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s 2> %t1
; RUN: FileCheck %s < %t1

; Test that the immediate intrinsics with out of range values trigger an error.


define void @binsli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.binsli.b(<16 x i8> %a, <16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}
; CHECK: LLVM ERROR: Immediate out of range

define void @binsri_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.binsri.b(<16 x i8> %a, <16 x i8> %a, i32 5)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @bmnzi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %a, <16 x i8> %a, i32 63)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @bmzi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %a, <16 x i8> %a, i32 63)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @bnegi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bnegi.b(<16 x i8> %a, i32 6)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @bseli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseli.b(<16 x i8> %a, <16 x i8> %a, i32 63)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @bseti_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseti.b(<16 x i8> %a, i32 9)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @clei_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.s.b(<16 x i8> %a, i32 152)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @clei_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.u.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @clti_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.s.b(<16 x i8> %a, i32 129)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @clti_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.u.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @ldi_b(ptr %ptr) {
entry:
  %r = call <16 x i8> @llvm.mips.ldi.b(i32 1025)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.s.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.u.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @mini_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.s.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @mini_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.u.b(<16 x i8> %a, i32 163)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @nori_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.nori.b(<16 x i8> %a, i32 63)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @ori_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.ori.b(<16 x i8> %a, i32 63)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @sldi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.sldi.b(<16 x i8> %a, <16 x i8> %a, i32 7)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @slli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.slli.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @splati_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.splati.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @srai_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srai.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @srari_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srari.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @srli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srli.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @srlri_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srlri.b(<16 x i8> %a, i32 65)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @addvi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.addvi.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @bclri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bclri.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @binsli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.binsli.w(<4 x i32> %a, <4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @binsri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.binsri.w(<4 x i32> %a, <4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @bnegi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bnegi.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @bseti_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bseti.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @clei_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.s.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @clei_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.u.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @clti_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.s.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @clti_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.u.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.s.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.u.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @mini_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.s.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @mini_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.u.w(<4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @ldi_w(ptr %ptr) {
entry:
  %r = call <4 x i32> @llvm.mips.ldi.w(i32 1024)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @sldi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.sldi.w(<4 x i32> %a, <4 x i32> %a, i32 63)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @slli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.slli.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @splati_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.splati.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @srai_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srai.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @srari_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srari.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @srli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srli.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @srlri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srlri.w(<4 x i32> %a, i32 65)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @addvi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.addvi.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @bclri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bclri.h(<8 x i16> %a, i32 16)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @binsli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.binsli.h(<8 x i16> %a, <8 x i16> %a, i32 17)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @binsri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.binsri.h(<8 x i16> %a, <8 x i16> %a, i32 19)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @bnegi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bnegi.h(<8 x i16> %a, i32 19)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @bseti_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bseti.h(<8 x i16> %a, i32 19)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @clei_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.s.h(<8 x i16> %a, i32 63)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @clei_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.u.h(<8 x i16> %a, i32 130)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @clti_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.s.h(<8 x i16> %a, i32 63)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @clti_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.u.h(<8 x i16> %a, i32 63)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.s.h(<8 x i16> %a, i32 63)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.u.h(<8 x i16> %a, i32 130)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @mini_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.s.h(<8 x i16> %a, i32 63)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @mini_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.u.h(<8 x i16> %a, i32 130)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @ldi_h(ptr %ptr) {
entry:
  %r = call <8 x i16> @llvm.mips.ldi.h(i32 1024)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @sldi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.sldi.h(<8 x i16> %a, <8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @slli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.slli.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @splati_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.splati.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @srai_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srai.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @srari_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srari.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @srli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srli.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @srlri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srlri.h(<8 x i16> %a, i32 65)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define i32 @copy_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.b(<16 x i8> %a, i32 17)
  ret i32 %r
}


define i32 @copy_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.h(<8 x i16> %a, i32 9)
  ret i32 %r
}


define i32 @copy_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.w(<4 x i32> %a, i32 5)
  ret i32 %r
}


define i32 @copy_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.b(<16 x i8> %a, i32 16)
  ret i32 %r
}


define i32 @copy_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.h(<8 x i16> %a, i32 9)
  ret i32 %r
}


define i32 @copy_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.w(<4 x i32> %a, i32 5)
  ret i32 %r
}

define i64 @copy_s_d(ptr %ptr) {
entry:  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call i64 @llvm.mips.copy.s.d(<2 x i64> %a, i32 3)
  ret i64 %r
}

define i64 @copy_u_d(ptr %ptr) {
entry:  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call i64 @llvm.mips.copy.u.d(<2 x i64> %a, i32 3)
  ret i64 %r
}

define void @addvi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.addvi.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @bclri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bclri.d(<2 x i64> %a, i32 64)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @binsli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.binsli.d(<2 x i64> %a, <2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @binsri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.binsri.d(<2 x i64> %a, <2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @bnegi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bnegi.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @bseti_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bseti.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @clei_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.s.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @clei_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.u.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @clti_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.s.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @clti_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.u.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @ldi_d(ptr %ptr) {
entry:
  %r = call <2 x i64> @llvm.mips.ldi.d(i32 1024)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.s.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @maxi_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.u.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @mini_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.s.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @mini_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.u.d(<2 x i64> %a, i32 63)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @sldi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.sldi.d(<2 x i64> %a, <2 x i64> %a, i32 1)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @slli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.slli.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @srai_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srai.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @srari_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srari.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @srli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srli.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @srlri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srlri.d(<2 x i64> %a, i32 65)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}; Negative numbers


define void @neg_addvi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.addvi.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_andi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.andi.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bclri_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bclri.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.binsli.b(<16 x i8> %a, <16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsri_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.binsri.b(<16 x i8> %a, <16 x i8> %a, i32 5)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bmnzi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %a, <16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bmzi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %a, <16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bnegi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bnegi.b(<16 x i8> %a, i32 6)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bseli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseli.b(<16 x i8> %a, <16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bseti_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.bseti.b(<16 x i8> %a, i32 -5)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.s.b(<16 x i8> %a, i32 -120)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clei.u.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.s.b(<16 x i8> %a, i32 -35)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.clti.u.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_ldi_b(ptr %ptr) {
entry:
  %r = call <16 x i8> @llvm.mips.ldi.b(i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.s.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.maxi.u.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.s.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.mini.u.b(<16 x i8> %a, i32 2)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_nori_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.nori.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_ori_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.ori.b(<16 x i8> %a, i32 -25)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_sldi_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.sldi.b(<16 x i8> %a, <16 x i8> %a, i32 -7)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_slli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.slli.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_splati_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.splati.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srai_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srai.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srari_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srari.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srli_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srli.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srlri_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call <16 x i8> @llvm.mips.srlri.b(<16 x i8> %a, i32 -3)
  store <16 x i8> %r, ptr %ptr, align 16
  ret void
}

define void @neg_addvi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.addvi.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bclri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bclri.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.binsli.w(<4 x i32> %a, <4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.binsri.w(<4 x i32> %a, <4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bnegi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bnegi.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bseti_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.bseti.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.s.w(<4 x i32> %a, i32 -140)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clei.u.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.s.w(<4 x i32> %a, i32 -150)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.clti.u.w(<4 x i32> %a, i32 -25)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.s.w(<4 x i32> %a, i32 -200)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.maxi.u.w(<4 x i32> %a, i32 -200)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.s.w(<4 x i32> %a, i32 -200)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.mini.u.w(<4 x i32> %a, i32 -200)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_ldi_w(ptr %ptr) {
entry:
  %r = call <4 x i32> @llvm.mips.ldi.w(i32 -300)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_sldi_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.sldi.w(<4 x i32> %a, <4 x i32> %a, i32 -20)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_slli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.slli.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_splati_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.splati.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srai_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srai.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srari_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srari.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srli_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srli.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srlri_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call <4 x i32> @llvm.mips.srlri.w(<4 x i32> %a, i32 -3)
  store <4 x i32> %r, ptr %ptr, align 16
  ret void
}

define void @neg_addvi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.addvi.h(<8 x i16> %a, i32 -25)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bclri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bclri.h(<8 x i16> %a, i32 -8)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.binsli.h(<8 x i16> %a, <8 x i16> %a, i32 -8)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.binsri.h(<8 x i16> %a, <8 x i16> %a, i32 -15)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bnegi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bnegi.h(<8 x i16> %a, i32 -14)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bseti_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.bseti.h(<8 x i16> %a, i32 -15)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.s.h(<8 x i16> %a, i32 -25)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clei.u.h(<8 x i16> %a, i32 -25)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.s.h(<8 x i16> %a, i32 -150)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.clti.u.h(<8 x i16> %a, i32 -25)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.s.h(<8 x i16> %a, i32 -200)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.maxi.u.h(<8 x i16> %a, i32 -200)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.s.h(<8 x i16> %a, i32 -200)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.mini.u.h(<8 x i16> %a, i32 -2)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_ldi_h(ptr %ptr) {
entry:
  %r = call <8 x i16> @llvm.mips.ldi.h(i32 -300)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_sldi_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.sldi.h(<8 x i16> %a, <8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_slli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.slli.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_splati_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.splati.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srai_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srai.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srari_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srari.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srli_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srli.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srlri_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call <8 x i16> @llvm.mips.srlri.h(<8 x i16> %a, i32 -3)
  store <8 x i16> %r, ptr %ptr, align 16
  ret void
}

define i32 @neg_copy_s_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.b(<16 x i8> %a, i32 -1)
  ret i32 %r
}

define i32 @neg_copy_s_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.h(<8 x i16> %a, i32 -1)
  ret i32 %r
}

define i32 @neg_copy_s_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.s.w(<4 x i32> %a, i32 -1)
  ret i32 %r
}

define i32 @neg_copy_u_b(ptr %ptr) {
entry:
  %a = load <16 x i8>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.b(<16 x i8> %a, i32 -1)
  ret i32 %r
}


define i32 @neg_copy_u_h(ptr %ptr) {
entry:
  %a = load <8 x i16>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.h(<8 x i16> %a, i32 -1)
  ret i32 %r
}


define i32 @neg_copy_u_w(ptr %ptr) {
entry:
  %a = load <4 x i32>, ptr %ptr, align 16
  %r = call i32 @llvm.mips.copy.u.w(<4 x i32> %a, i32 -1)
  ret i32 %r
}

define i64 @neg_copy_s_d(ptr %ptr) {
entry:  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call i64 @llvm.mips.copy.s.d(<2 x i64> %a, i32 -1)
  ret i64 %r
}

define i64 @neg_copy_u_d(ptr %ptr) {
entry:  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call i64 @llvm.mips.copy.u.d(<2 x i64> %a, i32 -1)
  ret i64 %r
}

define void @neg_addvi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.addvi.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bclri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bclri.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.binsli.d(<2 x i64> %a, <2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_binsri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.binsri.d(<2 x i64> %a, <2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bnegi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bnegi.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_bseti_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.bseti.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.s.d(<2 x i64> %a, i32 -45)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clei_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clei.u.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.s.d(<2 x i64> %a, i32 -32)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_clti_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.clti.u.d(<2 x i64> %a, i32 -25)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_ldi_d(ptr %ptr) {
entry:
  %r = call <2 x i64> @llvm.mips.ldi.d(i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.s.d(<2 x i64> %a, i32 -202)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_maxi_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.maxi.u.d(<2 x i64> %a, i32 -2)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_s_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.s.d(<2 x i64> %a, i32 -202)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_mini_u_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.mini.u.d(<2 x i64> %a, i32 -2)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_sldi_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.sldi.d(<2 x i64> %a, <2 x i64> %a, i32 -1)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_slli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.slli.d(<2 x i64> %a, i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srai_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srai.d(<2 x i64> %a, i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srari_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srari.d(<2 x i64> %a, i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srli_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srli.d(<2 x i64> %a, i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
  ret void
}

define void @neg_srlri_d(ptr %ptr) {
entry:
  %a = load <2 x i64>, ptr %ptr, align 16
  %r = call <2 x i64> @llvm.mips.srlri.d(<2 x i64> %a, i32 -3)
  store <2 x i64> %r, ptr %ptr, align 16
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
