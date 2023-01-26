; Test the MSA move intrinsics (which are encoded with the ELM instruction
; format).

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_move_vb_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_move_vb_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_move_vb_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_move_vb_ARG1
  %1 = tail call <16 x i8> @llvm.mips.move.v(<16 x i8> %0)
  store <16 x i8> %1, ptr @llvm_mips_move_vb_RES
  ret void
}

declare <16 x i8> @llvm.mips.move.v(<16 x i8>) nounwind

; CHECK: llvm_mips_move_vb_test:
; CHECK: ld.b
; CHECK: move.v
; CHECK: st.b
; CHECK: .size llvm_mips_move_vb_test
;
