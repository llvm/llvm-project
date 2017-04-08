; Test the MSA intrinsics that are encoded with the I5 instruction format.
; There are lots of these so this covers those beginning with 'b'

; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_bclri_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bclri_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bclri_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bclri_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bclri.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bclri_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bclri.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bclri_b_test:
; CHECK: ld.b
; andi.b is equivalent to bclri.b
; CHECK: andi.b {{\$w[0-9]}}, {{\$w[0-9]}}, 127
; CHECK: st.b
; CHECK: .size llvm_mips_bclri_b_test
;
@llvm_mips_bclri_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bclri_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bclri_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bclri_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bclri.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_bclri_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bclri.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_bclri_h_test:
; CHECK: ld.h
; CHECK: bclri.h
; CHECK: st.h
; CHECK: .size llvm_mips_bclri_h_test
;
@llvm_mips_bclri_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bclri_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bclri_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bclri_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bclri.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_bclri_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bclri.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_bclri_w_test:
; CHECK: ld.w
; CHECK: bclri.w
; CHECK: st.w
; CHECK: .size llvm_mips_bclri_w_test
;
@llvm_mips_bclri_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bclri_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bclri_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bclri_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bclri.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_bclri_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bclri.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_bclri_d_test:
; CHECK: ld.d
; CHECK: bclri.d
; CHECK: st.d
; CHECK: .size llvm_mips_bclri_d_test
;
@llvm_mips_binsli_b_ARG1 = global <16 x i8> zeroinitializer, align 16
@llvm_mips_binsli_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_binsli_b_RES  = global <16 x i8> zeroinitializer, align 16

define void @llvm_mips_binsli_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_binsli_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_binsli_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.binsli.b(<16 x i8> %0, <16 x i8> %1, i32 6)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_binsli_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.binsli.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_binsli_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsli_b_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsli_b_ARG2)(
; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsli.b [[R3]], [[R4]], 6
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsli_b_RES)(
; CHECK-DAG: st.b [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsli_b_test

@llvm_mips_binsli_h_ARG1 = global <8 x i16> zeroinitializer, align 16
@llvm_mips_binsli_h_ARG2 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_binsli_h_RES  = global <8 x i16> zeroinitializer, align 16

define void @llvm_mips_binsli_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_binsli_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_binsli_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.binsli.h(<8 x i16> %0, <8 x i16> %1, i32 7)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_binsli_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.binsli.h(<8 x i16>, <8 x i16>, i32) nounwind

; CHECK: llvm_mips_binsli_h_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsli_h_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsli_h_ARG2)(
; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.h [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsli.h [[R3]], [[R4]], 7
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsli_h_RES)(
; CHECK-DAG: st.h [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsli_h_test

@llvm_mips_binsli_w_ARG1 = global <4 x i32> zeroinitializer, align 16
@llvm_mips_binsli_w_ARG2 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_binsli_w_RES  = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_binsli_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_binsli_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_binsli_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.binsli.w(<4 x i32> %0, <4 x i32> %1, i32 7)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_binsli_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.binsli.w(<4 x i32>, <4 x i32>, i32) nounwind

; CHECK: llvm_mips_binsli_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsli_w_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsli_w_ARG2)(
; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.w [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsli.w [[R3]], [[R4]], 7
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsli_w_RES)(
; CHECK-DAG: st.w [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsli_w_test

@llvm_mips_binsli_d_ARG1 = global <2 x i64> zeroinitializer, align 16
@llvm_mips_binsli_d_ARG2 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_binsli_d_RES  = global <2 x i64> zeroinitializer, align 16

define void @llvm_mips_binsli_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_binsli_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_binsli_d_ARG2
  ; TODO: We use a particularly wide mask here to work around a legalization
  ;       issue. If the mask doesn't fit within a 10-bit immediate, it gets
  ;       legalized into a constant pool. We should add a test to cover the
  ;       other cases once they correctly select binsli.d.
  %2 = tail call <2 x i64> @llvm.mips.binsli.d(<2 x i64> %0, <2 x i64> %1, i32 61)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_binsli_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.binsli.d(<2 x i64>, <2 x i64>, i32) nounwind

; CHECK: llvm_mips_binsli_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsli_d_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsli_d_ARG2)(
; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.d [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsli.d [[R3]], [[R4]], 61
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsli_d_RES)(
; CHECK-DAG: st.d [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsli_d_test

@llvm_mips_binsri_b_ARG1 = global <16 x i8> zeroinitializer, align 16
@llvm_mips_binsri_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_binsri_b_RES  = global <16 x i8> zeroinitializer, align 16

define void @llvm_mips_binsri_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_binsri_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_binsri_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.binsri.b(<16 x i8> %0, <16 x i8> %1, i32 6)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_binsri_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.binsri.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_binsri_b_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsri_b_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsri_b_ARG2)(
; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.b [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsri.b [[R3]], [[R4]], 6
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsri_b_RES)(
; CHECK-DAG: st.b [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsri_b_test

@llvm_mips_binsri_h_ARG1 = global <8 x i16> zeroinitializer, align 16
@llvm_mips_binsri_h_ARG2 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_binsri_h_RES  = global <8 x i16> zeroinitializer, align 16

define void @llvm_mips_binsri_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_binsri_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_binsri_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.binsri.h(<8 x i16> %0, <8 x i16> %1, i32 7)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_binsri_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.binsri.h(<8 x i16>, <8 x i16>, i32) nounwind

; CHECK: llvm_mips_binsri_h_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsri_h_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsri_h_ARG2)(
; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.h [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsri.h [[R3]], [[R4]], 7
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsri_h_RES)(
; CHECK-DAG: st.h [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsri_h_test

@llvm_mips_binsri_w_ARG1 = global <4 x i32> zeroinitializer, align 16
@llvm_mips_binsri_w_ARG2 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_binsri_w_RES  = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_binsri_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_binsri_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_binsri_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.binsri.w(<4 x i32> %0, <4 x i32> %1, i32 7)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_binsri_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.binsri.w(<4 x i32>, <4 x i32>, i32) nounwind

; CHECK: llvm_mips_binsri_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsri_w_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsri_w_ARG2)(
; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.w [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsri.w [[R3]], [[R4]], 7
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsri_w_RES)(
; CHECK-DAG: st.w [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsri_w_test

@llvm_mips_binsri_d_ARG1 = global <2 x i64> zeroinitializer, align 16
@llvm_mips_binsri_d_ARG2 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_binsri_d_RES  = global <2 x i64> zeroinitializer, align 16

define void @llvm_mips_binsri_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_binsri_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_binsri_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.binsri.d(<2 x i64> %0, <2 x i64> %1, i32 7)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_binsri_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.binsri.d(<2 x i64>, <2 x i64>, i32) nounwind

; CHECK: llvm_mips_binsri_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_binsri_d_ARG1)(
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_binsri_d_ARG2)(
; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ld.d [[R4:\$w[0-9]+]], 0([[R2]])
; CHECK-DAG: binsri.d [[R3]], [[R4]], 7
; CHECK-DAG: lw [[R5:\$[0-9]+]], %got(llvm_mips_binsri_d_RES)(
; CHECK-DAG: st.d [[R3]], 0([[R5]])
; CHECK: .size llvm_mips_binsri_d_test

@llvm_mips_bnegi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bnegi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bnegi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bnegi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bnegi.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bnegi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bnegi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bnegi_b_test:
; CHECK: ld.b
; CHECK: bnegi.b
; CHECK: st.b
; CHECK: .size llvm_mips_bnegi_b_test
;
@llvm_mips_bnegi_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bnegi_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bnegi_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bnegi_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bnegi.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_bnegi_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bnegi.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_bnegi_h_test:
; CHECK: ld.h
; CHECK: bnegi.h
; CHECK: st.h
; CHECK: .size llvm_mips_bnegi_h_test
;
@llvm_mips_bnegi_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bnegi_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bnegi_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bnegi_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bnegi.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_bnegi_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bnegi.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_bnegi_w_test:
; CHECK: ld.w
; CHECK: bnegi.w
; CHECK: st.w
; CHECK: .size llvm_mips_bnegi_w_test
;
@llvm_mips_bnegi_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bnegi_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bnegi_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bnegi_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bnegi.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_bnegi_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bnegi.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_bnegi_d_test:
; CHECK: ld.d
; CHECK: bnegi.d
; CHECK: st.d
; CHECK: .size llvm_mips_bnegi_d_test
;
@llvm_mips_bseti_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bseti_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bseti_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bseti_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bseti.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_bseti_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bseti.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_bseti_b_test:
; CHECK: ld.b
; CHECK: bseti.b
; CHECK: st.b
; CHECK: .size llvm_mips_bseti_b_test
;
@llvm_mips_bseti_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bseti_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bseti_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bseti_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bseti.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_bseti_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bseti.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_bseti_h_test:
; CHECK: ld.h
; CHECK: bseti.h
; CHECK: st.h
; CHECK: .size llvm_mips_bseti_h_test
;
@llvm_mips_bseti_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bseti_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bseti_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bseti_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bseti.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_bseti_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bseti.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_bseti_w_test:
; CHECK: ld.w
; CHECK: bseti.w
; CHECK: st.w
; CHECK: .size llvm_mips_bseti_w_test
;
@llvm_mips_bseti_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bseti_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bseti_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bseti_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bseti.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_bseti_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bseti.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_bseti_d_test:
; CHECK: ld.d
; CHECK: bseti.d
; CHECK: st.d
; CHECK: .size llvm_mips_bseti_d_test
;
