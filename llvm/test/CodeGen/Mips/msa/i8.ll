; Test the MSA intrinsics that are encoded with the I8 instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_andi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_andi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_andi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_andi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.andi.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_andi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.andi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_andi_b_test:
; CHECK: ld.b
; CHECK: andi.b
; CHECK: st.b
; CHECK: .size llvm_mips_andi_b_test

@llvm_mips_bmnzi_b_ARG1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_bmnzi_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmnzi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmnzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bmnzi_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bmnzi_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %0, <16 x i8> %1, i32 25)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bmnzi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bmnzi.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_bmnzi_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG2)(
; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: bmnzi.b [[R3]], [[R4]], 25
; CHECK-DAG: st.b [[R3]], 0(
; CHECK: .size llvm_mips_bmnzi_b_test

@llvm_mips_bmzi_b_ARG1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_bmzi_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmzi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bmzi_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bmzi_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %0, <16 x i8> %1, i32 25)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bmzi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bmzi.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_bmzi_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmzi_b_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmzi_b_ARG2)(
; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; bmnzi.b is the same as bmzi.b with ws and wd_in swapped
; CHECK-DAG: bmnzi.b [[R4]], [[R3]], 25
; CHECK-DAG: st.b [[R4]], 0(
; CHECK: .size llvm_mips_bmzi_b_test

@llvm_mips_bseli_b_ARG1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_bseli_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bseli_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bseli_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bseli_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bseli_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bseli.b(<16 x i8> %0, <16 x i8> %1, i32 25)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bseli_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bseli.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_bseli_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bseli_b_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bseli_b_ARG2)(
; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: bseli.b [[R3]], [[R4]], 25
; CHECK-DAG: st.b [[R3]], 0(
; CHECK: .size llvm_mips_bseli_b_test

@llvm_mips_nori_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_nori_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_nori_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_nori_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.nori.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_nori_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.nori.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_nori_b_test:
; CHECK: ld.b
; CHECK: nori.b
; CHECK: st.b
; CHECK: .size llvm_mips_nori_b_test
;
@llvm_mips_ori_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ori_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ori_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ori_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.ori.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ori_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ori.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_ori_b_test:
; CHECK: ld.b
; CHECK: ori.b
; CHECK: st.b
; CHECK: .size llvm_mips_ori_b_test
;
@llvm_mips_shf_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_shf_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_shf_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_shf_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.shf.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_shf_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.shf.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_shf_b_test:
; CHECK: ld.b
; CHECK: shf.b
; CHECK: st.b
; CHECK: .size llvm_mips_shf_b_test
;
@llvm_mips_shf_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_shf_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_shf_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_shf_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.shf.h(<8 x i16> %0, i32 25)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_shf_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.shf.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_shf_h_test:
; CHECK: ld.h
; CHECK: shf.h
; CHECK: st.h
; CHECK: .size llvm_mips_shf_h_test
;
@llvm_mips_shf_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_shf_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_shf_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_shf_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.shf.w(<4 x i32> %0, i32 25)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_shf_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.shf.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_shf_w_test:
; CHECK: ld.w
; CHECK: shf.w
; CHECK: st.w
; CHECK: .size llvm_mips_shf_w_test
;
@llvm_mips_xori_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_xori_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_xori_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_xori_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.xori.b(<16 x i8> %0, i32 25)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_xori_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.xori.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_xori_b_test:
; CHECK: ld.b
; CHECK: xori.b
; CHECK: st.b
; CHECK: .size llvm_mips_xori_b_test
;
