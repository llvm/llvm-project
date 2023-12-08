; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_bmnzi_b_ARG1 = global <16 x i8> <i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15>, align 16
@llvm_mips_bmnzi_b_ARG2 = global <16 x i8> zeroinitializer, align 16
@llvm_mips_bmnzi_b_RES = global <16 x i8> zeroinitializer, align 16

define void @llvm_mips_bmnzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bmnzi_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_bmnzi_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %0, <16 x i8> %1, i32 240)
  store volatile <16 x i8> %2, ptr @llvm_mips_bmnzi_b_RES
  %3 = tail call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %0, <16 x i8> %1, i32 15)
  store volatile <16 x i8> %3, ptr @llvm_mips_bmnzi_b_RES
  %4 = tail call <16 x i8> @llvm.mips.bmnzi.b(<16 x i8> %0, <16 x i8> %1, i32 170)
  store <16 x i8> %4, ptr @llvm_mips_bmnzi_b_RES
  ret void
}
; CHECK-LABEL: llvm_mips_bmnzi_b_test:
; CHECK: lw [[R0:\$[0-9]+]], %got(llvm_mips_bmnzi_b_RES)(
; CHECK: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG1)(
; CHECK: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG2)(
; CHECK: ld.b [[R3:\$w[0-9]+]], 0([[R2]])
; CHECK: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; CHECK: move.v [[R5:\$w[0-9]+]], [[R4]]
; CHECK: binsli.b [[R5]], [[R3]], 3
; CHECK: binsri.b [[R5]], [[R3]], 3
; CHECK: bmnzi.b [[R4]], [[R3]], 170

define void @llvm_mips_bmzi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bmnzi_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_bmnzi_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %0, <16 x i8> %1, i32 240)
  store volatile <16 x i8> %2, ptr @llvm_mips_bmnzi_b_RES
  %3 = tail call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %0, <16 x i8> %1, i32 15)
  store volatile <16 x i8> %3, ptr @llvm_mips_bmnzi_b_RES
  %4 = tail call <16 x i8> @llvm.mips.bmzi.b(<16 x i8> %0, <16 x i8> %1, i32 170)
  store <16 x i8> %4, ptr @llvm_mips_bmnzi_b_RES
  ret void
}
; CHECK-LABEL: llvm_mips_bmzi_b_test:
; CHECK: lw [[R0:\$[0-9]+]], %got(llvm_mips_bmnzi_b_RES)(
; CHECK: lw [[R1:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG2)(
; CHECK: lw [[R2:\$[0-9]+]], %got(llvm_mips_bmnzi_b_ARG1)(
; CHECK: ld.b [[R3:\$w[0-9]+]], 0([[R2]])
; CHECK: ld.b [[R4:\$w[0-9]+]], 0([[R1]])
; CHECK: move.v [[R5:\$w[0-9]+]], [[R4]]
; CHECK: binsli.b [[R5]], [[R3]], 3
; CHECK: binsri.b [[R5]], [[R3]], 3
; bmnzi.b is the same as bmzi.b with ws and wd_in swapped
; CHECK: bmnzi.b [[R4]], [[R3]], 170

declare <16 x i8> @llvm.mips.bmnzi.b(<16 x i8>, <16 x i8>, i32) nounwind
declare <16 x i8> @llvm.mips.bmzi.b(<16 x i8>, <16 x i8>, i32) nounwind
