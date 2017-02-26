; Test the MSA intrinsics that are encoded with the I5 instruction format and
; are loads or stores.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_ld_b_ARG = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ld_b_RES = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ld_b_test() nounwind {
entry:
  %0 = bitcast <16 x i8>* @llvm_mips_ld_b_ARG to i8*
  %1 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 16)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ld_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ld.b(i8*, i32) nounwind

; CHECK: llvm_mips_ld_b_test:
; CHECK: ld.b [[R1:\$w[0-9]+]], 16(
; CHECK: st.b
; CHECK: .size llvm_mips_ld_b_test
;

define void @llvm_mips_ld_b_unaligned_test() nounwind {
entry:
  %0 = bitcast <16 x i8>* @llvm_mips_ld_b_ARG to i8*
  %1 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 9)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ld_b_RES
  ret void
}

; CHECK: llvm_mips_ld_b_unaligned_test:
; CHECK: ld.b [[R1:\$w[0-9]+]], 9(
; CHECK: st.b
; CHECK: .size llvm_mips_ld_b_unaligned_test
;

define void @llvm_mips_ld_b_valid_range_tests() nounwind {
entry:
  %0 = bitcast <16 x i8>* @llvm_mips_ld_b_ARG to i8*
  %1 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 -512)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ld_b_RES
  %2 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 511)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ld_b_RES
  ret void
}

; CHECK: llvm_mips_ld_b_valid_range_tests:
; CHECK: ld.b [[R1:\$w[0-9]+]], -512(
; CHECK: st.b
; CHECK: ld.b [[R1:\$w[0-9]+]], 511(
; CHECK: st.b
; CHECK: .size llvm_mips_ld_b_valid_range_tests
;

define void @llvm_mips_ld_b_invalid_range_tests() nounwind {
entry:
  %0 = bitcast <16 x i8>* @llvm_mips_ld_b_ARG to i8*
  %1 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 -513)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ld_b_RES
  %2 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 512)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ld_b_RES
  ret void
}

; CHECK: llvm_mips_ld_b_invalid_range_tests:
; CHECK: addiu $3, $2, -513
; CHECK: ld.b [[R1:\$w[0-9]+]], 0(
; CHECK: st.b
; CHECK: addiu $2, $2, 512
; CHECK: ld.b [[R1:\$w[0-9]+]], 0(
; CHECK: st.b
; CHECK: .size llvm_mips_ld_b_invalid_range_tests
;

@llvm_mips_ld_h_ARG = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ld_h_RES = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ld_h_test() nounwind {
entry:
  %0 = bitcast <8 x i16>* @llvm_mips_ld_h_ARG to i8*
  %1 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 16)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ld_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ld.h(i8*, i32) nounwind

; CHECK: llvm_mips_ld_h_test:
; CHECK: ld.h [[R1:\$w[0-9]+]], 16(
; CHECK: st.h
; CHECK: .size llvm_mips_ld_h_test
;

define void @llvm_mips_ld_h_unaligned_test() nounwind {
entry:
  %0 = bitcast <8 x i16>* @llvm_mips_ld_h_ARG to i8*
  %1 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 9)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ld_h_RES
  ret void
}

; CHECK: llvm_mips_ld_h_unaligned_test:
; CHECK: addiu $2, $2, 9
; CHECK: ld.h [[R1:\$w[0-9]+]], 0($2)
; CHECK: st.h
; CHECK: .size llvm_mips_ld_h_unaligned_test
;

define void @llvm_mips_ld_h_valid_range_tests() nounwind {
entry:
  %0 = bitcast <8 x i16>* @llvm_mips_ld_h_ARG to i8*
  %1 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 -1024)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ld_h_RES
  %2 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 1022)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ld_h_RES
  ret void
}

; CHECK: llvm_mips_ld_h_valid_range_tests:
; CHECK: ld.h [[R1:\$w[0-9]+]], -1024(
; CHECK: st.h
; CHECK: ld.h [[R1:\$w[0-9]+]], 1022(
; CHECK: st.h
; CHECK: .size llvm_mips_ld_h_valid_range_tests
;

define void @llvm_mips_ld_h_invalid_range_tests() nounwind {
entry:
  %0 = bitcast <8 x i16>* @llvm_mips_ld_h_ARG to i8*
  %1 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 -1026)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ld_h_RES
  %2 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 1024)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ld_h_RES
  ret void
}

; CHECK: llvm_mips_ld_h_invalid_range_tests:
; CHECK: addiu $3, $2, -1026
; CHECK: ld.h [[R1:\$w[0-9]+]], 0(
; CHECK: st.h
; CHECK: addiu $2, $2, 1024
; CHECK: ld.h [[R1:\$w[0-9]+]], 0(
; CHECK: st.h
; CHECK: .size llvm_mips_ld_h_invalid_range_tests
;

@llvm_mips_ld_w_ARG = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ld_w_RES = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ld_w_test() nounwind {
entry:
  %0 = bitcast <4 x i32>* @llvm_mips_ld_w_ARG to i8*
  %1 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 16)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ld_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ld.w(i8*, i32) nounwind

; CHECK: llvm_mips_ld_w_test:
; CHECK: ld.w [[R1:\$w[0-9]+]], 16(
; CHECK: st.w
; CHECK: .size llvm_mips_ld_w_test
;
@llvm_mips_ld_d_ARG = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ld_d_RES = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ld_w_unaligned_test() nounwind {
entry:
  %0 = bitcast <4 x i32>* @llvm_mips_ld_w_ARG to i8*
  %1 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 9)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ld_w_RES
  ret void
}

; CHECK: llvm_mips_ld_w_unaligned_test:
; CHECK: addiu $2, $2, 9
; CHECK: ld.w [[R1:\$w[0-9]+]], 0($2)
; CHECK: st.w
; CHECK: .size llvm_mips_ld_w_unaligned_test
;

define void @llvm_mips_ld_w_valid_range_tests() nounwind {
entry:
  %0 = bitcast <4 x i32>* @llvm_mips_ld_w_ARG to i8*
  %1 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 -2048)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ld_w_RES
  %2 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 2044)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ld_w_RES
  ret void
}

; CHECK: llvm_mips_ld_w_valid_range_tests:
; CHECK: ld.w [[R1:\$w[0-9]+]], -2048(
; CHECK: st.w
; CHECK: ld.w [[R1:\$w[0-9]+]], 2044(
; CHECK: st.w
; CHECK: .size llvm_mips_ld_w_valid_range_tests
;

define void @llvm_mips_ld_w_invalid_range_tests() nounwind {
entry:
  %0 = bitcast <4 x i32>* @llvm_mips_ld_w_ARG to i8*
  %1 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 -2052)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ld_w_RES
  %2 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 2048)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ld_w_RES
  ret void
}

; CHECK: llvm_mips_ld_w_invalid_range_tests:
; CHECK: addiu $3, $2, -2052
; CHECK: ld.w [[R1:\$w[0-9]+]], 0(
; CHECK: st.w
; CHECK: addiu $2, $2, 2048
; CHECK: ld.w [[R1:\$w[0-9]+]], 0(
; CHECK: st.w
; CHECK: .size llvm_mips_ld_w_invalid_range_tests
;

define void @llvm_mips_ld_d_test() nounwind {
entry:
  %0 = bitcast <2 x i64>* @llvm_mips_ld_d_ARG to i8*
  %1 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 16)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ld_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ld.d(i8*, i32) nounwind

; CHECK: llvm_mips_ld_d_test:
; CHECK: ld.d [[R1:\$w[0-9]+]], 16(
; CHECK: st.d
; CHECK: .size llvm_mips_ld_d_test
;

define void @llvm_mips_ld_d_unaligned_test() nounwind {
entry:
  %0 = bitcast <2 x i64>* @llvm_mips_ld_d_ARG to i8*
  %1 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 9)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ld_d_RES
  ret void
}

; CHECK: llvm_mips_ld_d_unaligned_test:
; CHECK: addiu $2, $2, 9
; CHECK: ld.d [[R1:\$w[0-9]+]], 0($2)
; CHECK: st.d
; CHECK: .size llvm_mips_ld_d_unaligned_test
;

define void @llvm_mips_ld_d_valid_range_tests() nounwind {
entry:
  %0 = bitcast <2 x i64>* @llvm_mips_ld_d_ARG to i8*
  %1 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 -4096)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ld_d_RES
  %2 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 4088)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ld_d_RES
  ret void
}

; CHECK: llvm_mips_ld_d_valid_range_tests:
; CHECK: ld.d [[R1:\$w[0-9]+]], -4096(
; CHECK: st.d
; CHECK: ld.d [[R1:\$w[0-9]+]], 4088(
; CHECK: st.d
; CHECK: .size llvm_mips_ld_d_valid_range_tests
;

define void @llvm_mips_ld_d_invalid_range_tests() nounwind {
entry:
  %0 = bitcast <2 x i64>* @llvm_mips_ld_d_ARG to i8*
  %1 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 -4104)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ld_d_RES
  %2 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 4096)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ld_d_RES
  ret void
}

; CHECK: llvm_mips_ld_d_invalid_range_tests:
; CHECK: addiu $3, $2, -4104
; CHECK: ld.d [[R1:\$w[0-9]+]], 0(
; CHECK: st.d
; CHECK: addiu $2, $2, 4096
; CHECK: ld.d [[R1:\$w[0-9]+]], 0(
; CHECK: st.d
; CHECK: .size llvm_mips_ld_d_invalid_range_tests
;



@llvm_mips_st_b_ARG = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_st_b_RES = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_st_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_st_b_ARG
  %1 = bitcast <16 x i8>* @llvm_mips_st_b_RES to i8*
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.b(<16 x i8>, i8*, i32) nounwind

; CHECK: llvm_mips_st_b_test:
; CHECK: ld.b
; CHECK: st.b [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_b_test
;

define void @llvm_mips_st_b_unaligned_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_st_b_ARG
  %1 = bitcast <16 x i8>* @llvm_mips_st_b_RES to i8*
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 9)
  ret void
}

; CHECK: llvm_mips_st_b_unaligned_test:
; CHECK: ld.b
; CHECK: st.b [[R1:\$w[0-9]+]], 9(
; CHECK: .size llvm_mips_st_b_unaligned_test
;

define void @llvm_mips_st_b_valid_range_tests() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_st_b_ARG
  %1 = bitcast <16 x i8>* @llvm_mips_st_b_RES to i8*
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 -512)
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 511)
  ret void
}

; CHECK: llvm_mips_st_b_valid_range_tests:
; CHECK: ld.b
; CHECK: st.b [[R1:\$w[0-9]+]], -512(
; CHECK: st.b [[R1:\$w[0-9]+]], 511(
; CHECK: .size llvm_mips_st_b_valid_range_tests
;

define void @llvm_mips_st_b_invalid_range_tests() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_st_b_ARG
  %1 = bitcast <16 x i8>* @llvm_mips_st_b_RES to i8*
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 -513)
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 512)
  ret void
}

; CHECK: llvm_mips_st_b_invalid_range_tests:
; CHECK: addiu $2, $1, -513
; CHECK: ld.b
; CHECK: st.b [[R1:\$w[0-9]+]], 0(
; CHECK: addiu $1, $1, 512
; CHECK: st.b [[R1:\$w[0-9]+]], 0(
; CHECK: .size llvm_mips_st_b_invalid_range_tests
;

@llvm_mips_st_h_ARG = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_st_h_RES = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_st_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_st_h_ARG
  %1 = bitcast <8 x i16>* @llvm_mips_st_h_RES to i8*
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.h(<8 x i16>, i8*, i32) nounwind

; CHECK: llvm_mips_st_h_test:
; CHECK: ld.h
; CHECK: st.h [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_h_test
;

define void @llvm_mips_st_h_unaligned_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_st_h_ARG
  %1 = bitcast <8 x i16>* @llvm_mips_st_h_RES to i8*
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 9)
  ret void
}

; CHECK: llvm_mips_st_h_unaligned_test:
; CHECK: addiu $1, $1, 9
; CHECK: ld.h
; CHECK: st.h [[R1:\$w[0-9]+]], 0($1)
; CHECK: .size llvm_mips_st_h_unaligned_test
;

define void @llvm_mips_st_h_valid_range_tests() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_st_h_ARG
  %1 = bitcast <8 x i16>* @llvm_mips_st_h_RES to i8*
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 -1024)
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 1022)
  ret void
}

; CHECK: llvm_mips_st_h_valid_range_tests:
; CHECK: ld.h
; CHECK: st.h [[R1:\$w[0-9]+]], -1024(
; CHECK: st.h [[R1:\$w[0-9]+]], 1022(
; CHECK: .size llvm_mips_st_h_valid_range_tests
;

define void @llvm_mips_st_h_invalid_range_tests() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_st_h_ARG
  %1 = bitcast <8 x i16>* @llvm_mips_st_h_RES to i8*
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 -1026)
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 1024)
  ret void
}

; CHECK: llvm_mips_st_h_invalid_range_tests:
; CHECK: addiu $2, $1, -1026
; CHECK: ld.h
; CHECK: st.h [[R1:\$w[0-9]+]], 0(
; CHECK: addiu $1, $1, 1024
; CHECK: st.h [[R1:\$w[0-9]+]], 0(
; CHECK: .size llvm_mips_st_h_invalid_range_tests
;

@llvm_mips_st_w_ARG = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_st_w_RES = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_st_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_st_w_ARG
  %1 = bitcast <4 x i32>* @llvm_mips_st_w_RES to i8*
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.w(<4 x i32>, i8*, i32) nounwind

; CHECK: llvm_mips_st_w_test:
; CHECK: ld.w
; CHECK: st.w [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_w_test
;

define void @llvm_mips_st_w_unaligned_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_st_w_ARG
  %1 = bitcast <4 x i32>* @llvm_mips_st_w_RES to i8*
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 9)
  ret void
}

; CHECK: llvm_mips_st_w_unaligned_test:
; CHECK: addiu $1, $1, 9
; CHECK: ld.w
; CHECK: st.w [[R1:\$w[0-9]+]], 0($1)
; CHECK: .size llvm_mips_st_w_unaligned_test
;

define void @llvm_mips_st_w_valid_range_tests() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_st_w_ARG
  %1 = bitcast <4 x i32>* @llvm_mips_st_w_RES to i8*
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 -2048)
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 2044)
  ret void
}

; CHECK: llvm_mips_st_w_valid_range_tests:
; CHECK: ld.w
; CHECK: st.w [[R1:\$w[0-9]+]], -2048(
; CHECK: st.w [[R1:\$w[0-9]+]], 2044(
; CHECK: .size llvm_mips_st_w_valid_range_tests
;

define void @llvm_mips_st_w_invalid_range_tests() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_st_w_ARG
  %1 = bitcast <4 x i32>* @llvm_mips_st_w_RES to i8*
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 -2052)
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 2048)
  ret void
}

; CHECK: llvm_mips_st_w_invalid_range_tests:
; CHECK: addiu $2, $1, -2052
; CHECK: ld.w
; CHECK: st.w [[R1:\$w[0-9]+]], 0(
; CHECK: addiu $1, $1, 2048
; CHECK: st.w [[R1:\$w[0-9]+]], 0(
; CHECK: .size llvm_mips_st_w_invalid_range_tests
;

@llvm_mips_st_d_ARG = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_st_d_RES = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_st_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_st_d_ARG
  %1 = bitcast <2 x i64>* @llvm_mips_st_d_RES to i8*
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.d(<2 x i64>, i8*, i32) nounwind

; CHECK: llvm_mips_st_d_test:
; CHECK: ld.d
; CHECK: st.d [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_d_test
;

define void @llvm_mips_st_d_unaligned_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_st_d_ARG
  %1 = bitcast <2 x i64>* @llvm_mips_st_d_RES to i8*
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 9)
  ret void
}

; CHECK: llvm_mips_st_d_unaligned_test:
; CHECK: addiu $1, $1, 9
; CHECK: ld.d
; CHECK: st.d [[R1:\$w[0-9]+]], 0($1)
; CHECK: .size llvm_mips_st_d_unaligned_test
;

define void @llvm_mips_st_d_valid_range_tests() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_st_d_ARG
  %1 = bitcast <2 x i64>* @llvm_mips_st_d_RES to i8*
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 -4096)
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 4088)
  ret void
}

; CHECK: llvm_mips_st_d_valid_range_tests:
; CHECK: ld.d
; CHECK: st.d [[R1:\$w[0-9]+]], -4096(
; CHECK: st.d [[R1:\$w[0-9]+]], 4088(
; CHECK: .size llvm_mips_st_d_valid_range_tests
;

define void @llvm_mips_st_d_invalid_range_tests() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_st_d_ARG
  %1 = bitcast <2 x i64>* @llvm_mips_st_d_RES to i8*
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 -4104)
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 4096)
  ret void
}

; CHECK: llvm_mips_st_d_invalid_range_tests:
; CHECK: addiu $2, $1, -4104
; CHECK: ld.d
; CHECK: st.d [[R1:\$w[0-9]+]], 0(
; CHECK: addiu $1, $1, 4096
; CHECK: st.d [[R1:\$w[0-9]+]], 0(
; CHECK: .size llvm_mips_st_d_invalid_range_tests
;
