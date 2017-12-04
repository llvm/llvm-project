; RUN: llc -mtriple=i686-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s --check-prefix=X32
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s --check-prefix=X64

; Test for case where insertps folds the load of an insertion element from a constant pool.

define <4 x float> @fold_from_constantpool(<4 x float> %a) {
; X32-LABEL: fold_from_constantpool:
; X32:       # %bb.0:
; X32-NEXT:    insertps {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; X32-NEXT:    retl
;
; X64-LABEL: fold_from_constantpool:
; X64:       # %bb.0:
; X64-NEXT:    insertps {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; X64-NEXT:    retq
  %1 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> <float 0.0, float 1.0, float 0.0, float 0.0>, i8 64)
  ret <4 x float> %1
}

declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i8) nounwind readnone
