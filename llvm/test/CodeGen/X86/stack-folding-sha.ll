; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+sha < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <4 x i32> @stack_fold_sha1msg1(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha1msg1
  ;CHECK:       sha1msg1 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha1msg1(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha1msg1(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_sha1msg2(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha1msg2
  ;CHECK:       sha1msg2 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha1msg2(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha1msg2(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_sha1nexte(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha1nexte
  ;CHECK:       sha1nexte {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha1nexte(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha1nexte(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_sha1rnds4(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha1rnds4
  ;CHECK:       sha1rnds4 $3, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha1rnds4(<4 x i32> %a0, <4 x i32> %a1, i8 3)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha1rnds4(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <4 x i32> @stack_fold_sha256msg1(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha256msg1
  ;CHECK:       sha256msg1 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha256msg1(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha256msg1(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_sha256msg2(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_sha256msg2
  ;CHECK:       sha256msg2 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha256msg2(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha256msg2(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_sha256rnds2(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_sha256rnds2
  ;CHECK:       sha256rnds2 {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = tail call <4 x i32> @llvm.x86.sha256rnds2(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sha256rnds2(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
