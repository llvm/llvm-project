; Test the MSA intrinsics that are encoded with the VECS10 instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_bnz_v_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16

define i32 @llvm_mips_bnz_v_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bnz_v_ARG1
  %1 = tail call i32 @llvm.mips.bnz.v(<16 x i8> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bnz.v(<16 x i8>) nounwind

; CHECK: llvm_mips_bnz_v_test:
; CHECK-DAG: ld.b [[R0:\$w[0-9]+]]
; CHECK-DAG: bnz.v [[R0]]
; CHECK: .size llvm_mips_bnz_v_test

@llvm_mips_bz_v_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16

define i32 @llvm_mips_bz_v_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bz_v_ARG1
  %1 = tail call i32 @llvm.mips.bz.v(<16 x i8> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bz.v(<16 x i8>) nounwind

; CHECK: llvm_mips_bz_v_test:
; CHECK-DAG: ld.b [[R0:\$w[0-9]+]]
; CHECK-DAG: bz.v [[R0]]
; CHECK: .size llvm_mips_bz_v_test
;
