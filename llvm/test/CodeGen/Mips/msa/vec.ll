; Test the MSA intrinsics that are encoded with the VEC instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefix=ANYENDIAN %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefix=ANYENDIAN %s

@llvm_mips_and_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_and_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_and_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_and_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_and_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_and_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.and.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_and_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_and_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: and.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_and_v_b_test
;
@llvm_mips_and_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_and_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_and_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_and_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_and_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_and_v_h_ARG2
  %2 = bitcast <8 x i16> %0 to <16 x i8>
  %3 = bitcast <8 x i16> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.and.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <8 x i16>
  store <8 x i16> %5, <8 x i16>* @llvm_mips_and_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_and_v_h_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: and.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_and_v_h_test
;
@llvm_mips_and_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_and_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_and_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_and_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_and_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_and_v_w_ARG2
  %2 = bitcast <4 x i32> %0 to <16 x i8>
  %3 = bitcast <4 x i32> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.and.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <4 x i32>
  store <4 x i32> %5, <4 x i32>* @llvm_mips_and_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_and_v_w_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: and.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_and_v_w_test
;
@llvm_mips_and_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_and_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_and_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_and_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_and_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_and_v_d_ARG2
  %2 = bitcast <2 x i64> %0 to <16 x i8>
  %3 = bitcast <2 x i64> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.and.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <2 x i64>
  store <2 x i64> %5, <2 x i64>* @llvm_mips_and_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_and_v_d_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: and.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_and_v_d_test
;
define void @and_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_and_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_and_v_b_ARG2
  %2 = and <16 x i8> %0, %1
  store <16 x i8> %2, <16 x i8>* @llvm_mips_and_v_b_RES
  ret void
}

; ANYENDIAN: and_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: and.v
; ANYENDIAN: st.b
; ANYENDIAN: .size and_v_b_test
;
define void @and_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_and_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_and_v_h_ARG2
  %2 = and <8 x i16> %0, %1
  store <8 x i16> %2, <8 x i16>* @llvm_mips_and_v_h_RES
  ret void
}

; ANYENDIAN: and_v_h_test:
; ANYENDIAN: ld.h
; ANYENDIAN: ld.h
; ANYENDIAN: and.v
; ANYENDIAN: st.h
; ANYENDIAN: .size and_v_h_test
;

define void @and_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_and_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_and_v_w_ARG2
  %2 = and <4 x i32> %0, %1
  store <4 x i32> %2, <4 x i32>* @llvm_mips_and_v_w_RES
  ret void
}

; ANYENDIAN: and_v_w_test:
; ANYENDIAN: ld.w
; ANYENDIAN: ld.w
; ANYENDIAN: and.v
; ANYENDIAN: st.w
; ANYENDIAN: .size and_v_w_test
;

define void @and_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_and_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_and_v_d_ARG2
  %2 = and <2 x i64> %0, %1
  store <2 x i64> %2, <2 x i64>* @llvm_mips_and_v_d_RES
  ret void
}

; ANYENDIAN: and_v_d_test:
; ANYENDIAN: ld.d
; ANYENDIAN: ld.d
; ANYENDIAN: and.v
; ANYENDIAN: st.d
; ANYENDIAN: .size and_v_d_test
;
@llvm_mips_bmnz_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmnz_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bmnz_v_b_ARG3 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmnz_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmnz_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bmnz_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bmnz_v_b_ARG2
  %2 = load <16 x i8>, <16 x i8>* @llvm_mips_bmnz_v_b_ARG3
  %3 = bitcast <16 x i8> %0 to <16 x i8>
  %4 = bitcast <16 x i8> %1 to <16 x i8>
  %5 = bitcast <16 x i8> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmnz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <16 x i8>
  store <16 x i8> %7, <16 x i8>* @llvm_mips_bmnz_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmnz_v_b_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnz_v_b_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnz_v_b_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmnz_v_b_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; ANYENDIAN-DAG: bmnz.v [[R4]], [[R5]], [[R6]]
; ANYENDIAN-DAG: st.b [[R4]], 0(
; ANYENDIAN: .size llvm_mips_bmnz_v_b_test

@llvm_mips_bmnz_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bmnz_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bmnz_v_h_ARG3 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bmnz_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bmnz_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bmnz_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bmnz_v_h_ARG2
  %2 = load <8 x i16>, <8 x i16>* @llvm_mips_bmnz_v_h_ARG3
  %3 = bitcast <8 x i16> %0 to <16 x i8>
  %4 = bitcast <8 x i16> %1 to <16 x i8>
  %5 = bitcast <8 x i16> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmnz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <8 x i16>
  store <8 x i16> %7, <8 x i16>* @llvm_mips_bmnz_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmnz_v_h_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnz_v_h_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnz_v_h_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmnz_v_h_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; ANYENDIAN-DAG: bmnz.v [[R4]], [[R5]], [[R6]]
; ANYENDIAN-DAG: st.b [[R4]], 0(
; ANYENDIAN: .size llvm_mips_bmnz_v_h_test

@llvm_mips_bmnz_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bmnz_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bmnz_v_w_ARG3 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bmnz_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bmnz_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bmnz_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bmnz_v_w_ARG2
  %2 = load <4 x i32>, <4 x i32>* @llvm_mips_bmnz_v_w_ARG3
  %3 = bitcast <4 x i32> %0 to <16 x i8>
  %4 = bitcast <4 x i32> %1 to <16 x i8>
  %5 = bitcast <4 x i32> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmnz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <4 x i32>
  store <4 x i32> %7, <4 x i32>* @llvm_mips_bmnz_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmnz_v_w_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnz_v_w_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnz_v_w_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmnz_v_w_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; ANYENDIAN-DAG: bmnz.v [[R4]], [[R5]], [[R6]]
; ANYENDIAN-DAG: st.b [[R4]], 0(
; ANYENDIAN: .size llvm_mips_bmnz_v_w_test

@llvm_mips_bmnz_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bmnz_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bmnz_v_d_ARG3 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bmnz_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bmnz_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bmnz_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bmnz_v_d_ARG2
  %2 = load <2 x i64>, <2 x i64>* @llvm_mips_bmnz_v_d_ARG3
  %3 = bitcast <2 x i64> %0 to <16 x i8>
  %4 = bitcast <2 x i64> %1 to <16 x i8>
  %5 = bitcast <2 x i64> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmnz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <2 x i64>
  store <2 x i64> %7, <2 x i64>* @llvm_mips_bmnz_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmnz_v_d_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnz_v_d_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnz_v_d_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmnz_v_d_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; ANYENDIAN-DAG: bmnz.v [[R4]], [[R5]], [[R6]]
; ANYENDIAN-DAG: st.b [[R4]], 0(
; ANYENDIAN: .size llvm_mips_bmnz_v_d_test

@llvm_mips_bmz_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmz_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bmz_v_b_ARG3 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bmz_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bmz_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bmz_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bmz_v_b_ARG2
  %2 = load <16 x i8>, <16 x i8>* @llvm_mips_bmz_v_b_ARG3
  %3 = bitcast <16 x i8> %0 to <16 x i8>
  %4 = bitcast <16 x i8> %1 to <16 x i8>
  %5 = bitcast <16 x i8> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <16 x i8>
  store <16 x i8> %7, <16 x i8>* @llvm_mips_bmz_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmz_v_b_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmz_v_b_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmz_v_b_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmz_v_b_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bmz.v with ws and wd_in swapped
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R4]], [[R6]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bmz_v_b_test

@llvm_mips_bmz_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bmz_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bmz_v_h_ARG3 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bmz_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bmz_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bmz_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bmz_v_h_ARG2
  %2 = load <8 x i16>, <8 x i16>* @llvm_mips_bmz_v_h_ARG3
  %3 = bitcast <8 x i16> %0 to <16 x i8>
  %4 = bitcast <8 x i16> %1 to <16 x i8>
  %5 = bitcast <8 x i16> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <8 x i16>
  store <8 x i16> %7, <8 x i16>* @llvm_mips_bmz_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmz_v_h_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmz_v_h_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmz_v_h_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmz_v_h_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bmz.v with ws and wd_in swapped
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R4]], [[R6]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bmz_v_h_test

@llvm_mips_bmz_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bmz_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bmz_v_w_ARG3 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bmz_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bmz_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bmz_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bmz_v_w_ARG2
  %2 = load <4 x i32>, <4 x i32>* @llvm_mips_bmz_v_w_ARG3
  %3 = bitcast <4 x i32> %0 to <16 x i8>
  %4 = bitcast <4 x i32> %1 to <16 x i8>
  %5 = bitcast <4 x i32> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <4 x i32>
  store <4 x i32> %7, <4 x i32>* @llvm_mips_bmz_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmz_v_w_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmz_v_w_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmz_v_w_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmz_v_w_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bmz.v with ws and wd_in swapped
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R4]], [[R6]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bmz_v_w_test

@llvm_mips_bmz_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bmz_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bmz_v_d_ARG3 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bmz_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bmz_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bmz_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bmz_v_d_ARG2
  %2 = load <2 x i64>, <2 x i64>* @llvm_mips_bmz_v_d_ARG3
  %3 = bitcast <2 x i64> %0 to <16 x i8>
  %4 = bitcast <2 x i64> %1 to <16 x i8>
  %5 = bitcast <2 x i64> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bmz.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <2 x i64>
  store <2 x i64> %7, <2 x i64>* @llvm_mips_bmz_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_bmz_v_d_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmz_v_d_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmz_v_d_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bmz_v_d_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bmz.v with ws and wd_in swapped
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R4]], [[R6]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bmz_v_d_test

@llvm_mips_bsel_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bsel_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bsel_v_b_ARG3 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bsel_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bsel_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bsel_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bsel_v_b_ARG2
  %2 = load <16 x i8>, <16 x i8>* @llvm_mips_bsel_v_b_ARG3
  %3 = bitcast <16 x i8> %0 to <16 x i8>
  %4 = bitcast <16 x i8> %1 to <16 x i8>
  %5 = bitcast <16 x i8> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bsel.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <16 x i8>
  store <16 x i8> %7, <16 x i8>* @llvm_mips_bsel_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_bsel_v_b_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bsel_v_b_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bsel_v_b_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bsel_v_b_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bsel.v with (wd_in, wt, ws) -> (wt, ws, wd_in)
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R6]], [[R4]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bsel_v_b_test

@llvm_mips_bsel_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bsel_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bsel_v_h_ARG3 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bsel_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bsel_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bsel_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bsel_v_h_ARG2
  %2 = load <8 x i16>, <8 x i16>* @llvm_mips_bsel_v_h_ARG3
  %3 = bitcast <8 x i16> %0 to <16 x i8>
  %4 = bitcast <8 x i16> %1 to <16 x i8>
  %5 = bitcast <8 x i16> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bsel.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <8 x i16>
  store <8 x i16> %7, <8 x i16>* @llvm_mips_bsel_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_bsel_v_h_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bsel_v_h_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bsel_v_h_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bsel_v_h_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bsel.v with (wd_in, wt, ws) -> (wt, ws, wd_in)
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R6]], [[R4]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bsel_v_h_test

@llvm_mips_bsel_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bsel_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bsel_v_w_ARG3 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bsel_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bsel_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bsel_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bsel_v_w_ARG2
  %2 = load <4 x i32>, <4 x i32>* @llvm_mips_bsel_v_w_ARG3
  %3 = bitcast <4 x i32> %0 to <16 x i8>
  %4 = bitcast <4 x i32> %1 to <16 x i8>
  %5 = bitcast <4 x i32> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bsel.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <4 x i32>
  store <4 x i32> %7, <4 x i32>* @llvm_mips_bsel_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_bsel_v_w_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bsel_v_w_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bsel_v_w_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bsel_v_w_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bsel.v with (wd_in, wt, ws) -> (wt, ws, wd_in)
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R6]], [[R4]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bsel_v_w_test

@llvm_mips_bsel_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bsel_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bsel_v_d_ARG3 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bsel_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bsel_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bsel_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bsel_v_d_ARG2
  %2 = load <2 x i64>, <2 x i64>* @llvm_mips_bsel_v_d_ARG3
  %3 = bitcast <2 x i64> %0 to <16 x i8>
  %4 = bitcast <2 x i64> %1 to <16 x i8>
  %5 = bitcast <2 x i64> %2 to <16 x i8>
  %6 = tail call <16 x i8> @llvm.mips.bsel.v(<16 x i8> %3, <16 x i8> %4, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <2 x i64>
  store <2 x i64> %7, <2 x i64>* @llvm_mips_bsel_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_bsel_v_d_test:
; ANYENDIAN-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_bsel_v_d_ARG1)(
; ANYENDIAN-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_bsel_v_d_ARG2)(
; ANYENDIAN-DAG: lw [[R3:\$[0-9]+]], %got(llvm_mips_bsel_v_d_ARG3)(
; ANYENDIAN-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; ANYENDIAN-DAG: ld.b [[R5:\$w[0-9]+]], 0([[R2]])
; ANYENDIAN-DAG: ld.b [[R6:\$w[0-9]+]], 0([[R3]])
; bmnz.v is the same as bsel.v with (wd_in, wt, ws) -> (wt, ws, wd_in)
; ANYENDIAN-DAG: bmnz.v [[R5]], [[R6]], [[R4]]
; ANYENDIAN-DAG: st.b [[R5]], 0(
; ANYENDIAN: .size llvm_mips_bsel_v_d_test

@llvm_mips_nor_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_nor_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_nor_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_nor_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_nor_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_nor_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.nor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_nor_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_nor_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: nor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_nor_v_b_test
;
@llvm_mips_nor_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_nor_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_nor_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_nor_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_nor_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_nor_v_h_ARG2
  %2 = bitcast <8 x i16> %0 to <16 x i8>
  %3 = bitcast <8 x i16> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.nor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <8 x i16>
  store <8 x i16> %5, <8 x i16>* @llvm_mips_nor_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_nor_v_h_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: nor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_nor_v_h_test
;
@llvm_mips_nor_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_nor_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_nor_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_nor_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_nor_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_nor_v_w_ARG2
  %2 = bitcast <4 x i32> %0 to <16 x i8>
  %3 = bitcast <4 x i32> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.nor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <4 x i32>
  store <4 x i32> %5, <4 x i32>* @llvm_mips_nor_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_nor_v_w_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: nor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_nor_v_w_test
;
@llvm_mips_nor_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_nor_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_nor_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_nor_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_nor_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_nor_v_d_ARG2
  %2 = bitcast <2 x i64> %0 to <16 x i8>
  %3 = bitcast <2 x i64> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.nor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <2 x i64>
  store <2 x i64> %5, <2 x i64>* @llvm_mips_nor_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_nor_v_d_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: nor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_nor_v_d_test
;
@llvm_mips_or_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_or_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_or_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_or_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_or_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_or_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.or.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_or_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_or_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: or.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_or_v_b_test
;
@llvm_mips_or_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_or_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_or_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_or_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_or_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_or_v_h_ARG2
  %2 = bitcast <8 x i16> %0 to <16 x i8>
  %3 = bitcast <8 x i16> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.or.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <8 x i16>
  store <8 x i16> %5, <8 x i16>* @llvm_mips_or_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_or_v_h_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: or.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_or_v_h_test
;
@llvm_mips_or_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_or_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_or_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_or_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_or_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_or_v_w_ARG2
  %2 = bitcast <4 x i32> %0 to <16 x i8>
  %3 = bitcast <4 x i32> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.or.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <4 x i32>
  store <4 x i32> %5, <4 x i32>* @llvm_mips_or_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_or_v_w_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: or.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_or_v_w_test
;
@llvm_mips_or_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_or_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_or_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_or_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_or_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_or_v_d_ARG2
  %2 = bitcast <2 x i64> %0 to <16 x i8>
  %3 = bitcast <2 x i64> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.or.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <2 x i64>
  store <2 x i64> %5, <2 x i64>* @llvm_mips_or_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_or_v_d_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: or.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_or_v_d_test
;
define void @or_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_or_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_or_v_b_ARG2
  %2 = or <16 x i8> %0, %1
  store <16 x i8> %2, <16 x i8>* @llvm_mips_or_v_b_RES
  ret void
}

; ANYENDIAN: or_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: or.v
; ANYENDIAN: st.b
; ANYENDIAN: .size or_v_b_test
;
define void @or_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_or_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_or_v_h_ARG2
  %2 = or <8 x i16> %0, %1
  store <8 x i16> %2, <8 x i16>* @llvm_mips_or_v_h_RES
  ret void
}

; ANYENDIAN: or_v_h_test:
; ANYENDIAN: ld.h
; ANYENDIAN: ld.h
; ANYENDIAN: or.v
; ANYENDIAN: st.h
; ANYENDIAN: .size or_v_h_test
;

define void @or_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_or_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_or_v_w_ARG2
  %2 = or <4 x i32> %0, %1
  store <4 x i32> %2, <4 x i32>* @llvm_mips_or_v_w_RES
  ret void
}

; ANYENDIAN: or_v_w_test:
; ANYENDIAN: ld.w
; ANYENDIAN: ld.w
; ANYENDIAN: or.v
; ANYENDIAN: st.w
; ANYENDIAN: .size or_v_w_test
;

define void @or_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_or_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_or_v_d_ARG2
  %2 = or <2 x i64> %0, %1
  store <2 x i64> %2, <2 x i64>* @llvm_mips_or_v_d_RES
  ret void
}

; ANYENDIAN: or_v_d_test:
; ANYENDIAN: ld.d
; ANYENDIAN: ld.d
; ANYENDIAN: or.v
; ANYENDIAN: st.d
; ANYENDIAN: .size or_v_d_test
;
@llvm_mips_xor_v_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_xor_v_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_xor_v_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_xor_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_xor_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_xor_v_b_ARG2
  %2 = bitcast <16 x i8> %0 to <16 x i8>
  %3 = bitcast <16 x i8> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.xor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* @llvm_mips_xor_v_b_RES
  ret void
}

; ANYENDIAN: llvm_mips_xor_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: xor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_xor_v_b_test
;
@llvm_mips_xor_v_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_xor_v_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_xor_v_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_xor_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_xor_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_xor_v_h_ARG2
  %2 = bitcast <8 x i16> %0 to <16 x i8>
  %3 = bitcast <8 x i16> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.xor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <8 x i16>
  store <8 x i16> %5, <8 x i16>* @llvm_mips_xor_v_h_RES
  ret void
}

; ANYENDIAN: llvm_mips_xor_v_h_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: xor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_xor_v_h_test
;
@llvm_mips_xor_v_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_xor_v_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_xor_v_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_xor_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_xor_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_xor_v_w_ARG2
  %2 = bitcast <4 x i32> %0 to <16 x i8>
  %3 = bitcast <4 x i32> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.xor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <4 x i32>
  store <4 x i32> %5, <4 x i32>* @llvm_mips_xor_v_w_RES
  ret void
}

; ANYENDIAN: llvm_mips_xor_v_w_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: xor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_xor_v_w_test
;
@llvm_mips_xor_v_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_xor_v_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_xor_v_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_xor_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_xor_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_xor_v_d_ARG2
  %2 = bitcast <2 x i64> %0 to <16 x i8>
  %3 = bitcast <2 x i64> %1 to <16 x i8>
  %4 = tail call <16 x i8> @llvm.mips.xor.v(<16 x i8> %2, <16 x i8> %3)
  %5 = bitcast <16 x i8> %4 to <2 x i64>
  store <2 x i64> %5, <2 x i64>* @llvm_mips_xor_v_d_RES
  ret void
}

; ANYENDIAN: llvm_mips_xor_v_d_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: xor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size llvm_mips_xor_v_d_test
;
define void @xor_v_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_xor_v_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_xor_v_b_ARG2
  %2 = xor <16 x i8> %0, %1
  store <16 x i8> %2, <16 x i8>* @llvm_mips_xor_v_b_RES
  ret void
}

; ANYENDIAN: xor_v_b_test:
; ANYENDIAN: ld.b
; ANYENDIAN: ld.b
; ANYENDIAN: xor.v
; ANYENDIAN: st.b
; ANYENDIAN: .size xor_v_b_test
;
define void @xor_v_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_xor_v_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_xor_v_h_ARG2
  %2 = xor <8 x i16> %0, %1
  store <8 x i16> %2, <8 x i16>* @llvm_mips_xor_v_h_RES
  ret void
}

; ANYENDIAN: xor_v_h_test:
; ANYENDIAN: ld.h
; ANYENDIAN: ld.h
; ANYENDIAN: xor.v
; ANYENDIAN: st.h
; ANYENDIAN: .size xor_v_h_test
;

define void @xor_v_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_xor_v_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_xor_v_w_ARG2
  %2 = xor <4 x i32> %0, %1
  store <4 x i32> %2, <4 x i32>* @llvm_mips_xor_v_w_RES
  ret void
}

; ANYENDIAN: xor_v_w_test:
; ANYENDIAN: ld.w
; ANYENDIAN: ld.w
; ANYENDIAN: xor.v
; ANYENDIAN: st.w
; ANYENDIAN: .size xor_v_w_test
;

define void @xor_v_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_xor_v_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_xor_v_d_ARG2
  %2 = xor <2 x i64> %0, %1
  store <2 x i64> %2, <2 x i64>* @llvm_mips_xor_v_d_RES
  ret void
}

; ANYENDIAN: xor_v_d_test:
; ANYENDIAN: ld.d
; ANYENDIAN: ld.d
; ANYENDIAN: xor.v
; ANYENDIAN: st.d
; ANYENDIAN: .size xor_v_d_test
;
declare <16 x i8> @llvm.mips.and.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bmnz.v(<16 x i8>, <16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bmz.v(<16 x i8>, <16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.bsel.v(<16 x i8>, <16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.nor.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.or.v(<16 x i8>, <16 x i8>) nounwind
declare <16 x i8> @llvm.mips.xor.v(<16 x i8>, <16 x i8>) nounwind
