; Test the MSA element insertion intrinsics that are encoded with the ELM
; instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=MIPS-ANY,MIPS32
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=MIPS-ANY,MIPS32
; RUN: llc -march=mips64 -mcpu=mips64r2 -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=MIPS-ANY,MIPS64
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=MIPS-ANY,MIPS64

@llvm_mips_insert_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_insert_b_ARG3 = global i32 27, align 16
@llvm_mips_insert_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_insert_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_insert_b_ARG1
  %1 = load i32, i32* @llvm_mips_insert_b_ARG3
  %2 = tail call <16 x i8> @llvm.mips.insert.b(<16 x i8> %0, i32 1, i32 %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_insert_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.insert.b(<16 x i8>, i32, i32) nounwind

; MIPS-ANY: llvm_mips_insert_b_test:
; MIPS-ANY-DAG: lw [[R1:\$[0-9]+]], 0(
; MIPS-ANY-DAG: ld.b [[R2:\$w[0-9]+]], 0(
; MIPS-ANY-DAG: insert.b [[R2]][1], [[R1]]
; MIPS-ANY-DAG: st.b [[R2]], 0(
; MIPS-ANY: .size llvm_mips_insert_b_test
;
@llvm_mips_insert_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_insert_h_ARG3 = global i32 27, align 16
@llvm_mips_insert_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_insert_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_insert_h_ARG1
  %1 = load i32, i32* @llvm_mips_insert_h_ARG3
  %2 = tail call <8 x i16> @llvm.mips.insert.h(<8 x i16> %0, i32 1, i32 %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_insert_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.insert.h(<8 x i16>, i32, i32) nounwind

; MIPS-ANY: llvm_mips_insert_h_test:
; MIPS-ANY-DAG: lw [[R1:\$[0-9]+]], 0(
; MIPS-ANY-DAG: ld.h [[R2:\$w[0-9]+]], 0(
; MIPS-ANY-DAG: insert.h [[R2]][1], [[R1]]
; MIPS-ANY-DAG: st.h [[R2]], 0(
; MIPS-ANY: .size llvm_mips_insert_h_test
;
@llvm_mips_insert_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_insert_w_ARG3 = global i32 27, align 16
@llvm_mips_insert_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_insert_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_insert_w_ARG1
  %1 = load i32, i32* @llvm_mips_insert_w_ARG3
  %2 = tail call <4 x i32> @llvm.mips.insert.w(<4 x i32> %0, i32 1, i32 %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_insert_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.insert.w(<4 x i32>, i32, i32) nounwind

; MIPS-ANY: llvm_mips_insert_w_test:
; MIPS-ANY-DAG: lw [[R1:\$[0-9]+]], 0(
; MIPS-ANY-DAG: ld.w [[R2:\$w[0-9]+]], 0(
; MIPS-ANY-DAG: insert.w [[R2]][1], [[R1]]
; MIPS-ANY-DAG: st.w [[R2]], 0(
; MIPS-ANY: .size llvm_mips_insert_w_test
;
@llvm_mips_insert_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_insert_d_ARG3 = global i64 27, align 16
@llvm_mips_insert_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_insert_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_insert_d_ARG1
  %1 = load i64, i64* @llvm_mips_insert_d_ARG3
  %2 = tail call <2 x i64> @llvm.mips.insert.d(<2 x i64> %0, i32 1, i64 %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_insert_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.insert.d(<2 x i64>, i32, i64) nounwind

; MIPS-ANY: llvm_mips_insert_d_test:
; MIPS32-DAG: lw [[R1:\$[0-9]+]], 0(
; MIPS32-DAG: lw [[R2:\$[0-9]+]], 4(
; MIPS64-DAG: ld [[R1:\$[0-9]+]], 0(
; MIPS32-DAG: ld.w [[R3:\$w[0-9]+]],
; MIPS64-DAG: ld.d [[W1:\$w[0-9]+]],
; MIPS32-DAG: insert.w [[R3]][2], [[R1]]
; MIPS32-DAG: insert.w [[R3]][3], [[R2]]
; MIPS64-DAG: insert.d [[W1]][1], [[R1]]
; MIPS32-DAG: st.w [[R3]],
; MIPS64-DAG: st.d [[W1]],
; MIPS-ANY: .size llvm_mips_insert_d_test
;
@llvm_mips_insve_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_insve_b_ARG3 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_insve_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_insve_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_insve_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_insve_b_ARG3
  %2 = tail call <16 x i8> @llvm.mips.insve.b(<16 x i8> %0, i32 1, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_insve_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.insve.b(<16 x i8>, i32, <16 x i8>) nounwind

; MIPS-ANY: llvm_mips_insve_b_test:
; MIPS32-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_insve_b_ARG1)(
; MIPS32-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_insve_b_ARG3)(
; MIPS64-DAG: ld [[R1:\$[0-9]+]], %got_disp(llvm_mips_insve_b_ARG1)(
; MIPS64-DAG: ld [[R2:\$[0-9]+]], %got_disp(llvm_mips_insve_b_ARG3)(
; MIPS-ANY-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS-ANY-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; MIPS-ANY-DAG: insve.b [[R3]][1], [[R4]][0]
; MIPS-ANY-DAG: st.b [[R3]],
; MIPS-ANY: .size llvm_mips_insve_b_test
;
@llvm_mips_insve_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_insve_h_ARG3 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_insve_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_insve_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_insve_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_insve_h_ARG3
  %2 = tail call <8 x i16> @llvm.mips.insve.h(<8 x i16> %0, i32 1, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_insve_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.insve.h(<8 x i16>, i32, <8 x i16>) nounwind

; MIPS-ANY: llvm_mips_insve_h_test:
; MIPS32-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_insve_h_ARG1)(
; MIPS32-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_insve_h_ARG3)(
; MIPS64-DAG: ld [[R1:\$[0-9]+]], %got_disp(llvm_mips_insve_h_ARG1)(
; MIPS64-DAG: ld [[R2:\$[0-9]+]], %got_disp(llvm_mips_insve_h_ARG3)(
; MIPS-ANY-DAG: ld.h [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS-ANY-DAG: ld.h [[R4:\$w[0-9]+]], 0([[R2]])
; MIPS-ANY-DAG: insve.h [[R3]][1], [[R4]][0]
; MIPS-ANY-DAG: st.h [[R3]],
; MIPS-ANY: .size llvm_mips_insve_h_test
;
@llvm_mips_insve_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_insve_w_ARG3 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_insve_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_insve_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_insve_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_insve_w_ARG3
  %2 = tail call <4 x i32> @llvm.mips.insve.w(<4 x i32> %0, i32 1, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_insve_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.insve.w(<4 x i32>, i32, <4 x i32>) nounwind

; MIPS-ANY: llvm_mips_insve_w_test:
; MIPS32-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_insve_w_ARG1)(
; MIPS32-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_insve_w_ARG3)(
; MIPS64-DAG: ld [[R1:\$[0-9]+]], %got_disp(llvm_mips_insve_w_ARG1)(
; MIPS64-DAG: ld [[R2:\$[0-9]+]], %got_disp(llvm_mips_insve_w_ARG3)(
; MIPS-ANY-DAG: ld.w [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS-ANY-DAG: ld.w [[R4:\$w[0-9]+]], 0([[R2]])
; MIPS-ANY-DAG: insve.w [[R3]][1], [[R4]][0]
; MIPS-ANY-DAG: st.w [[R3]],
; MIPS-ANY: .size llvm_mips_insve_w_test
;
@llvm_mips_insve_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_insve_d_ARG3 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_insve_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_insve_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_insve_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_insve_d_ARG3
  %2 = tail call <2 x i64> @llvm.mips.insve.d(<2 x i64> %0, i32 1, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_insve_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.insve.d(<2 x i64>, i32, <2 x i64>) nounwind

; MIPS-ANY: llvm_mips_insve_d_test:
; MIPS32-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_insve_d_ARG1)(
; MIPS32-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_insve_d_ARG3)(
; MIPS64-DAG: ld [[R1:\$[0-9]+]], %got_disp(llvm_mips_insve_d_ARG1)(
; MIPS64-DAG: ld [[R2:\$[0-9]+]], %got_disp(llvm_mips_insve_d_ARG3)(
; MIPS-ANY-DAG: ld.d [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS-ANY-DAG: ld.d [[R4:\$w[0-9]+]], 0([[R2]])
; MIPS-ANY-DAG: insve.d [[R3]][1], [[R4]][0]
; MIPS-ANY-DAG: st.d [[R3]],
; MIPS-ANY: .size llvm_mips_insve_d_test
;
