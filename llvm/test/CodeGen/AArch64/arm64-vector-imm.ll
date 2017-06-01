; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @v_orrimm(<8 x i8>* %A) nounwind {
; CHECK-LABEL: v_orrimm:
; CHECK-NOT: mov
; CHECK-NOT: mvn
; CHECK: orr
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp3 = or <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1>
	ret <8 x i8> %tmp3
}

define <16 x i8> @v_orrimmQ(<16 x i8>* %A) nounwind {
; CHECK: v_orrimmQ
; CHECK-NOT: mov
; CHECK-NOT: mvn
; CHECK: orr
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp3 = or <16 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1>
	ret <16 x i8> %tmp3
}

define <8 x i8> @v_bicimm(<8 x i8>* %A) nounwind {
; CHECK-LABEL: v_bicimm:
; CHECK-NOT: mov
; CHECK-NOT: mvn
; CHECK: bic
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp3 = and <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0 >
	ret <8 x i8> %tmp3
}

define <16 x i8> @v_bicimmQ(<16 x i8>* %A) nounwind {
; CHECK-LABEL: v_bicimmQ:
; CHECK-NOT: mov
; CHECK-NOT: mvn
; CHECK: bic
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp3 = and <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0 >
	ret <16 x i8> %tmp3
}

define <2 x double> @foo(<2 x double> %bar) nounwind {
; CHECK: foo
; CHECK: fmov.2d	v1, #1.0000000
  %add = fadd <2 x double> %bar, <double 1.0, double 1.0>
  ret <2 x double> %add
}

define <4 x i32> @movi_4s_imm_t1() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t1:
; CHECK: movi.4s v0, #75
  ret <4 x i32> <i32 75, i32 75, i32 75, i32 75>
}

define <4 x i32> @movi_4s_imm_t2() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t2:
; CHECK: movi.4s v0, #75, lsl #8
  ret <4 x i32> <i32 19200, i32 19200, i32 19200, i32 19200>
}

define <4 x i32> @movi_4s_imm_t3() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t3:
; CHECK: movi.4s v0, #75, lsl #16
  ret <4 x i32> <i32 4915200, i32 4915200, i32 4915200, i32 4915200>
}

define <4 x i32> @movi_4s_imm_t4() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t4:
; CHECK: movi.4s v0, #75, lsl #24
  ret <4 x i32> <i32 1258291200, i32 1258291200, i32 1258291200, i32 1258291200>
}

define <8 x i16> @movi_8h_imm_t5() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_8h_imm_t5:
; CHECK: movi.8h v0, #75
  ret <8 x i16> <i16 75, i16 75, i16 75, i16 75, i16 75, i16 75, i16 75, i16 75>
}

; rdar://11989841
define <8 x i16> @movi_8h_imm_t6() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_8h_imm_t6:
; CHECK: movi.8h v0, #75, lsl #8
  ret <8 x i16> <i16 19200, i16 19200, i16 19200, i16 19200, i16 19200, i16 19200, i16 19200, i16 19200>
}

define <4 x i32> @movi_4s_imm_t7() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t7:
; CHECK: movi.4s v0, #75, msl #8
ret <4 x i32> <i32 19455, i32 19455, i32 19455, i32 19455>
}

define <4 x i32> @movi_4s_imm_t8() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t8:
; CHECK: movi.4s v0, #75, msl #16
ret <4 x i32> <i32 4980735, i32 4980735, i32 4980735, i32 4980735>
}

define <16 x i8> @movi_16b_imm_t9() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_16b_imm_t9:
; CHECK: movi.16b v0, #75
ret <16 x i8> <i8 75, i8 75, i8 75, i8 75, i8 75, i8 75, i8 75, i8 75,
               i8 75, i8 75, i8 75, i8 75, i8 75, i8 75, i8 75, i8 75>
}

define <2 x i64> @movi_2d_imm_t10() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_2d_imm_t10:
; CHECK: movi.2d v0, #0xff00ff00ff00ff
ret <2 x i64> <i64 71777214294589695, i64 71777214294589695>
}

define <4 x i32> @movi_4s_imm_t11() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_4s_imm_t11:
; CHECK: fmov.4s v0, #-0.32812500
ret <4 x i32> <i32 3198681088, i32 3198681088, i32 3198681088, i32 3198681088>
}

define <2 x i64> @movi_2d_imm_t12() nounwind readnone ssp {
entry:
; CHECK-LABEL: movi_2d_imm_t12:
; CHECK: fmov.2d v0, #-0.17187500
ret <2 x i64> <i64 13818732506632945664, i64 13818732506632945664>
}
