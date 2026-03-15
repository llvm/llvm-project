; Test the MSA splat intrinsics that are encoded with the 3R instruction
; format.

; RUN: llc -mtriple=mips-elf -mcpu=mips32r5 -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefix=MIPS32 %s
; RUN: llc -mtriple=mipsel-elf -mcpu=mips32r5 -mattr=+msa,+fp64 -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefix=MIPS32 %s

@llvm_mips_splat_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_splat_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_splat_b_test(i32 %a) nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_splat_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.splat.b(<16 x i8> %0, i32 %a)
  store <16 x i8> %1, ptr @llvm_mips_splat_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.splat.b(<16 x i8>, i32) nounwind

; MIPS32: llvm_mips_splat_b_test:
; MIPS32-DAG: lw   [[R1:\$[0-9]+]], %got(llvm_mips_splat_b_ARG1)(
; MIPS32-DAG: lw   [[R2:\$[0-9]+]], %got(llvm_mips_splat_b_RES)(
; MIPS32-DAG: ld.b [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splat.b [[R4:\$w[0-9]+]], [[R3]][$4]
; MIPS32-DAG: st.b [[R4]], 0([[R2]])
; MIPS32: .size llvm_mips_splat_b_test

@llvm_mips_splat_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_splat_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_splat_h_test(i32 %a) nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_splat_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.splat.h(<8 x i16> %0, i32 %a)
  store <8 x i16> %1, ptr @llvm_mips_splat_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.splat.h(<8 x i16>, i32) nounwind

; MIPS32: llvm_mips_splat_h_test:
; MIPS32-DAG: lw   [[R1:\$[0-9]+]], %got(llvm_mips_splat_h_ARG1)(
; MIPS32-DAG: lw   [[R2:\$[0-9]+]], %got(llvm_mips_splat_h_RES)(
; MIPS32-DAG: ld.h [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splat.h [[R4:\$w[0-9]+]], [[R3]][$4]
; MIPS32-DAG: st.h [[R4]], 0([[R2]])
; MIPS32: .size llvm_mips_splat_h_test

@llvm_mips_splat_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_splat_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_splat_w_test(i32 %a) nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_splat_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.splat.w(<4 x i32> %0, i32 %a)
  store <4 x i32> %1, ptr @llvm_mips_splat_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.splat.w(<4 x i32>, i32) nounwind

; MIPS32: llvm_mips_splat_w_test:
; MIPS32-DAG: lw   [[R1:\$[0-9]+]], %got(llvm_mips_splat_w_ARG1)(
; MIPS32-DAG: lw   [[R2:\$[0-9]+]], %got(llvm_mips_splat_w_RES)(
; MIPS32-DAG: ld.w [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splat.w [[R4:\$w[0-9]+]], [[R3]][$4]
; MIPS32-DAG: st.w [[R4]], 0([[R2]])
; MIPS32: .size llvm_mips_splat_w_test

@llvm_mips_splat_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_splat_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_splat_d_test(i32 %a) nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_splat_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.splat.d(<2 x i64> %0, i32 %a)
  store <2 x i64> %1, ptr @llvm_mips_splat_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.splat.d(<2 x i64>, i32) nounwind

; MIPS32: llvm_mips_splat_d_test:
; MIPS32-DAG: lw   [[R1:\$[0-9]+]], %got(llvm_mips_splat_d_ARG1)(
; MIPS32-DAG: lw   [[R2:\$[0-9]+]], %got(llvm_mips_splat_d_RES)(
; MIPS32-DAG: ld.d [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splat.d [[R4:\$w[0-9]+]], [[R3]][$4]
; MIPS32-DAG: st.d [[R4]], 0([[R2]])
; MIPS32: .size llvm_mips_splat_d_test

define void @llvm_mips_splat_d_arg_test(i32 %arg) {
entry:
  %0 = tail call <2 x i64> @llvm.mips.splat.d(<2 x i64> <i64 12720328, i64 10580959>, i32 %arg)
  store volatile <2 x i64> %0, ptr @llvm_mips_splat_d_RES
  ret void
}
; MIPS32-LABEL: llvm_mips_splat_d_arg_test
; MIPS32-DAG: lw      [[R0:\$[0-9]+]], %got(
; MIPS32-DAG: addiu   [[R1:\$[0-9]+]], [[R0]], %lo(
; MIPS32-DAG: lw      [[R2:\$[0-9]+]], %got(llvm_mips_splat_d_RES)(
; MIPS32-DAG: ld.d    [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splat.d [[R4:\$w[0-9]+]], [[R3]][$4]
; MIPS32-DAG: st.d    [[R4]], 0([[R2]])
; MIPS32-NOT: vshf.d

define void @llvm_mips_splat_d_imm_test() {
entry:
  %0 = tail call <2 x i64> @llvm.mips.splat.d(<2 x i64> <i64 12720328, i64 10580959>, i32 76)
  store volatile<2 x i64> %0, ptr @llvm_mips_splat_d_RES
  ret void
}
; MIPS32-LABEL: llvm_mips_splat_d_imm_test
; MIPS32-DAG: lw       [[R0:\$[0-9]+]], %got(
; MIPS32-DAG: addiu    [[R1:\$[0-9]+]], [[R0]], %lo(
; MIPS32-DAG: lw       [[R2:\$[0-9]+]], %got(llvm_mips_splat_d_RES)(
; MIPS32-DAG: ld.d     [[R3:\$w[0-9]+]], 0([[R1]])
; MIPS32-DAG: splati.d [[R4:\$w[0-9]+]], [[R3]][0]
; MIPS32-DAG: st.d     [[R4]], 0([[R2]])
; MIPS32-NOT: vshf.d
