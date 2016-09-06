; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <2 x i32> @fcvtas_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtas_2s:
;CHECK-NOT: ld1
;CHECK: fcvtas.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtas.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtas_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtas_4s:
;CHECK-NOT: ld1
;CHECK: fcvtas.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtas.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtas_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtas_2d:
;CHECK-NOT: ld1
;CHECK: fcvtas.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtas.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtas.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtas.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtas.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtau_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtau_2s:
;CHECK-NOT: ld1
;CHECK: fcvtau.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtau.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtau_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtau_4s:
;CHECK-NOT: ld1
;CHECK: fcvtau.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtau.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtau_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtau_2d:
;CHECK-NOT: ld1
;CHECK: fcvtau.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtau.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtau.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtau.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtau.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtms_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtms_2s:
;CHECK-NOT: ld1
;CHECK: fcvtms.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtms.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtms_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtms_4s:
;CHECK-NOT: ld1
;CHECK: fcvtms.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtms.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtms_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtms_2d:
;CHECK-NOT: ld1
;CHECK: fcvtms.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtms.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtms.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtms.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtms.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtmu_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtmu_2s:
;CHECK-NOT: ld1
;CHECK: fcvtmu.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtmu.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtmu_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtmu_4s:
;CHECK-NOT: ld1
;CHECK: fcvtmu.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtmu.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtmu_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtmu_2d:
;CHECK-NOT: ld1
;CHECK: fcvtmu.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtmu.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtmu.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtmu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtmu.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtps_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtps_2s:
;CHECK-NOT: ld1
;CHECK: fcvtps.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtps.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtps_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtps_4s:
;CHECK-NOT: ld1
;CHECK: fcvtps.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtps.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtps_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtps_2d:
;CHECK-NOT: ld1
;CHECK: fcvtps.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtps.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtps.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtps.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtps.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtpu_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtpu_2s:
;CHECK-NOT: ld1
;CHECK: fcvtpu.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtpu.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtpu_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtpu_4s:
;CHECK-NOT: ld1
;CHECK: fcvtpu.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtpu.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtpu_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtpu_2d:
;CHECK-NOT: ld1
;CHECK: fcvtpu.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtpu.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtpu.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtpu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtpu.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtns_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtns_2s:
;CHECK-NOT: ld1
;CHECK: fcvtns.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtns.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtns_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtns_4s:
;CHECK-NOT: ld1
;CHECK: fcvtns.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtns.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtns_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtns_2d:
;CHECK-NOT: ld1
;CHECK: fcvtns.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtns.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtns.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtns.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtns.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtnu_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtnu_2s:
;CHECK-NOT: ld1
;CHECK: fcvtnu.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.fcvtnu.v2i32.v2f32(<2 x float> %A)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtnu_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtnu_4s:
;CHECK-NOT: ld1
;CHECK: fcvtnu.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.fcvtnu.v4i32.v4f32(<4 x float> %A)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtnu_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtnu_2d:
;CHECK-NOT: ld1
;CHECK: fcvtnu.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.fcvtnu.v2i64.v2f64(<2 x double> %A)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.fcvtnu.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.fcvtnu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.fcvtnu.v2i64.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtzs_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtzs_2s:
;CHECK-NOT: ld1
;CHECK: fcvtzs.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptosi <2 x float> %A to <2 x i32>
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtzs_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtzs_4s:
;CHECK-NOT: ld1
;CHECK: fcvtzs.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptosi <4 x float> %A to <4 x i32>
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtzs_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtzs_2d:
;CHECK-NOT: ld1
;CHECK: fcvtzs.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptosi <2 x double> %A to <2 x i64>
	ret <2 x i64> %tmp3
}


define <2 x i32> @fcvtzu_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtzu_2s:
;CHECK-NOT: ld1
;CHECK: fcvtzu.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptoui <2 x float> %A to <2 x i32>
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtzu_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtzu_4s:
;CHECK-NOT: ld1
;CHECK: fcvtzu.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptoui <4 x float> %A to <4 x i32>
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtzu_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtzu_2d:
;CHECK-NOT: ld1
;CHECK: fcvtzu.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = fptoui <2 x double> %A to <2 x i64>
	ret <2 x i64> %tmp3
}

define <2 x float> @frinta_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frinta_2s:
;CHECK-NOT: ld1
;CHECK: frinta.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.round.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frinta_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frinta_4s:
;CHECK-NOT: ld1
;CHECK: frinta.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.round.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frinta_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frinta_2d:
;CHECK-NOT: ld1
;CHECK: frinta.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.round.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.round.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.round.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.round.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frinti_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frinti_2s:
;CHECK-NOT: ld1
;CHECK: frinti.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frinti_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frinti_4s:
;CHECK-NOT: ld1
;CHECK: frinti.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frinti_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frinti_2d:
;CHECK-NOT: ld1
;CHECK: frinti.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.nearbyint.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frintm_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintm_2s:
;CHECK-NOT: ld1
;CHECK: frintm.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.floor.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintm_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintm_4s:
;CHECK-NOT: ld1
;CHECK: frintm.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.floor.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintm_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintm_2d:
;CHECK-NOT: ld1
;CHECK: frintm.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.floor.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.floor.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.floor.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frintn_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintn_2s:
;CHECK-NOT: ld1
;CHECK: frintn.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintn_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintn_4s:
;CHECK-NOT: ld1
;CHECK: frintn.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintn_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintn_2d:
;CHECK-NOT: ld1
;CHECK: frintn.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frintp_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintp_2s:
;CHECK-NOT: ld1
;CHECK: frintp.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.ceil.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintp_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintp_4s:
;CHECK-NOT: ld1
;CHECK: frintp.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.ceil.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintp_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintp_2d:
;CHECK-NOT: ld1
;CHECK: frintp.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.ceil.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.ceil.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.ceil.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.ceil.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frintx_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintx_2s:
;CHECK-NOT: ld1
;CHECK: frintx.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.rint.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintx_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintx_4s:
;CHECK-NOT: ld1
;CHECK: frintx.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.rint.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintx_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintx_2d:
;CHECK-NOT: ld1
;CHECK: frintx.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.rint.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.rint.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.rint.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.rint.v2f64(<2 x double>) nounwind readnone

define <2 x float> @frintz_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintz_2s:
;CHECK-NOT: ld1
;CHECK: frintz.2s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.trunc.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintz_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintz_4s:
;CHECK-NOT: ld1
;CHECK: frintz.4s v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.trunc.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintz_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintz_2d:
;CHECK-NOT: ld1
;CHECK: frintz.2d v0, v0
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.trunc.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.trunc.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.trunc.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.trunc.v2f64(<2 x double>) nounwind readnone

define <2 x float> @fcvtxn_2s(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtxn_2s:
;CHECK-NOT: ld1
;CHECK: fcvtxn v0.2s, v0.2d
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @fcvtxn_4s(<2 x float> %ret, <2 x double> %A) nounwind {
;CHECK-LABEL: fcvtxn_4s:
;CHECK-NOT: ld1
;CHECK: fcvtxn2 v0.4s, v1.2d
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %A)
        %res = shufflevector <2 x float> %ret, <2 x float> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	ret <4 x float> %res
}

declare <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double>) nounwind readnone

define <2 x i32> @fcvtzsc_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtzsc_2s:
;CHECK-NOT: ld1
;CHECK: fcvtzs.2s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> %A, i32 1)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtzsc_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtzsc_4s:
;CHECK-NOT: ld1
;CHECK: fcvtzs.4s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> %A, i32 1)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtzsc_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtzsc_2d:
;CHECK-NOT: ld1
;CHECK: fcvtzs.2d v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double> %A, i32 1)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double>, i32) nounwind readnone

define <2 x i32> @fcvtzuc_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: fcvtzuc_2s:
;CHECK-NOT: ld1
;CHECK: fcvtzu.2s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> %A, i32 1)
	ret <2 x i32> %tmp3
}

define <4 x i32> @fcvtzuc_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: fcvtzuc_4s:
;CHECK-NOT: ld1
;CHECK: fcvtzu.4s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> %A, i32 1)
	ret <4 x i32> %tmp3
}

define <2 x i64> @fcvtzuc_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: fcvtzuc_2d:
;CHECK-NOT: ld1
;CHECK: fcvtzu.2d v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double> %A, i32 1)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double>, i32) nounwind readnone

define <2 x float> @scvtf_2sc(<2 x i32> %A) nounwind {
;CHECK-LABEL: scvtf_2sc:
;CHECK-NOT: ld1
;CHECK: scvtf.2s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> %A, i32 1)
	ret <2 x float> %tmp3
}

define <4 x float> @scvtf_4sc(<4 x i32> %A) nounwind {
;CHECK-LABEL: scvtf_4sc:
;CHECK-NOT: ld1
;CHECK: scvtf.4s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> %A, i32 1)
	ret <4 x float> %tmp3
}

define <2 x double> @scvtf_2dc(<2 x i64> %A) nounwind {
;CHECK-LABEL: scvtf_2dc:
;CHECK-NOT: ld1
;CHECK: scvtf.2d v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64> %A, i32 1)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64>, i32) nounwind readnone

define <2 x float> @ucvtf_2sc(<2 x i32> %A) nounwind {
;CHECK-LABEL: ucvtf_2sc:
;CHECK-NOT: ld1
;CHECK: ucvtf.2s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> %A, i32 1)
	ret <2 x float> %tmp3
}

define <4 x float> @ucvtf_4sc(<4 x i32> %A) nounwind {
;CHECK-LABEL: ucvtf_4sc:
;CHECK-NOT: ld1
;CHECK: ucvtf.4s v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %A, i32 1)
	ret <4 x float> %tmp3
}

define <2 x double> @ucvtf_2dc(<2 x i64> %A) nounwind {
;CHECK-LABEL: ucvtf_2dc:
;CHECK-NOT: ld1
;CHECK: ucvtf.2d v0, v0, #1
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> %A, i32 1)
	ret <2 x double> %tmp3
}


;CHECK-LABEL: autogen_SD28458:
;CHECK: fcvt
;CHECK: ret
define void @autogen_SD28458(<8 x double> %val.f64, <8 x float>* %addr.f32) {
  %Tr53 = fptrunc <8 x double> %val.f64 to <8 x float>
  store <8 x float> %Tr53, <8 x float>* %addr.f32
  ret void
}

;CHECK-LABEL: autogen_SD19225:
;CHECK: fcvt
;CHECK: ret
define void @autogen_SD19225(<8 x double>* %addr.f64, <8 x float>* %addr.f32) {
  %A = load <8 x float>, <8 x float>* %addr.f32
  %Tr53 = fpext <8 x float> %A to <8 x double>
  store <8 x double> %Tr53, <8 x double>* %addr.f64
  ret void
}

declare <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64>, i32) nounwind readnone
