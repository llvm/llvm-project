; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512vl,+avx512bw,+avx512dq,+avx512vbmi,+avx512cd < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <8 x i32> @stack_fold_valignd_ymm(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: stack_fold_valignd_ymm
  ;CHECK:   valignd $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32><i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  ret <8 x i32> %2
}

define <8 x i32> @stack_fold_valignd_ymm_mask(<8 x i32> %a, <8 x i32> %b, <8 x i32>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_valignd_ymm_mask
  ;CHECK:   valignd $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32><i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = load <8 x i32>, <8 x i32>* %passthru
  %5 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> %4
  ret <8 x i32> %5
}

define <8 x i32> @stack_fold_valignd_ymm_maskz(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: stack_fold_valignd_ymm_maskz
  ;CHECK:   valignd $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32><i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> zeroinitializer
  ret <8 x i32> %4
}

define <4 x i64> @stack_fold_valignq_ymm(<4 x i64> %a, <4 x i64> %b) {
  ;CHECK-LABEL: stack_fold_valignq_ymm
  ;CHECK:   valignq $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i64> %2
}

define <16 x i8> @stack_fold_pavgb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pavgb
  ;CHECK:       vpavgb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <16 x i8> %a0 to <16 x i16>
  %3 = zext <16 x i8> %a1 to <16 x i16>
  %4 = add <16 x i16> %2, %3
  %5 = add <16 x i16> %4, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %6 = lshr <16 x i16> %5, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %7 = trunc <16 x i16> %6 to <16 x i8>
  ret <16 x i8> %7
}

define <32 x i8> @stack_fold_pavgb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pavgb_ymm
  ;CHECK:       vpavgb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <32 x i8> %a0 to <32 x i16>
  %3 = zext <32 x i8> %a1 to <32 x i16>
  %4 = add <32 x i16> %2, %3
  %5 = add <32 x i16> %4, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1,    i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %6 = lshr <32 x i16> %5, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1,   i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %7 = trunc <32 x i16> %6 to <32 x i8>
  ret <32 x i8> %7
}

define <8 x i16> @stack_fold_pavgw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pavgw
  ;CHECK:       vpavgw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <8 x i16> %a0 to <8 x i32>
  %3 = zext <8 x i16> %a1 to <8 x i32>
  %4 = add <8 x i32> %2, %3
  %5 = add <8 x i32> %4, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = lshr <8 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = trunc <8 x i32> %6 to <8 x i16>
  ret <8 x i16> %7
}

define <16 x i16> @stack_fold_pavgw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pavgw_ymm
  ;CHECK:       vpavgw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
 %2 = zext <16 x i16> %a0 to <16 x i32>
 %3 = zext <16 x i16> %a1 to <16 x i32>
 %4 = add <16 x i32> %2, %3
 %5 = add <16 x i32> %4, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
 %6 = lshr <16 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
 %7 = trunc <16 x i32> %6 to <16 x i16>
 ret <16 x i16> %7
}

define <4 x i32> @stack_fold_vpconflictd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpconflictd
  ;CHECK:       vpconflictd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32> %a0, <4 x i32> undef, i8 -1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx512.mask.conflict.d.128(<4 x i32>, <4 x i32>, i8) nounwind readonly

define <8 x i32> @stack_fold_vpconflictd_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpconflictd_ymm
  ;CHECK:       vpconflictd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx512.mask.conflict.d.256(<8 x i32> %a0, <8 x i32> undef, i8 -1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx512.mask.conflict.d.256(<8 x i32>, <8 x i32>, i8) nounwind readonly

define <2 x i64> @stack_fold_vpconflictq(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpconflictq
  ;CHECK:       vpconflictq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.conflict.q.128(<2 x i64> %a0, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.conflict.q.128(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_vpconflictq_ymm(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpconflictq_ymm
  ;CHECK:       vpconflictq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.conflict.q.256(<4 x i64> %a0, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.conflict.q.256(<4 x i64>, <4 x i64>, i8) nounwind readnone

define <4 x i32> @stack_fold_extracti32x4(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x4
  ;CHECK:       vextracti128 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <8 x i32> %a0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %2 = shufflevector <8 x i32> %1, <8 x i32> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_extracti64x2(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_extracti64x2
  ;CHECK:       vextracti128 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <4 x i64> %a0, <i64 1, i64 1, i64 1, i64 1>
  %2 = shufflevector <4 x i64> %1, <4 x i64> %a1, <2 x i32> <i32 2, i32 3>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <2 x i64> %2
}

define <8 x i32> @stack_fold_inserti32x4(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_inserti32x4
  ;CHECK:       vinserti128 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; add forces execution domain
  %3 = add <8 x i32> %2, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %3
}

define <4 x i64> @stack_fold_inserti64x2(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_inserti64x2
  ;CHECK:       vinserti128 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <2 x i64> %a0, <2 x i64> %a1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; add forces execution domain
  %3 = add <4 x i64> %2, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %3
}

define <16 x i8> @stack_fold_pabsb(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pabsb
  ;CHECK:       vpabsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8> %a0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_pabsb_ymm(<32 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pabsb_ymm
  ;CHECK:       vpabsb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.pabs.b(<32 x i8> %a0)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.pabs.b(<32 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pabsd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pabsd
  ;CHECK:       vpabsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pabsd_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pabsd_ymm
  ;CHECK:       vpabsd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pabs.d(<8 x i32> %a0)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pabs.d(<8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pabsq(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_pabsq
  ;CHECK:       vpabsq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pabs.q.128(<2 x i64> %a0, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.pabs.q.128(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_pabsq_ymm(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_pabsq_ymm
  ;CHECK:       vpabsq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pabs.q.256(<4 x i64> %a0, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.pabs.q.256(<4 x i64>, <4 x i64>, i8) nounwind readnone

define <8 x i16> @stack_fold_pabsw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pabsw
  ;CHECK:       vpabsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_pabsw_ymm(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pabsw_ymm
  ;CHECK:       vpabsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pabs.w(<16 x i16> %a0)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pabs.w(<16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_packssdw(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packssdw
  ;CHECK:       vpackssdw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %a0, <4 x i32> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>) nounwind readnone

define <16 x i16> @stack_fold_packssdw_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packssdw_ymm
  ;CHECK:       vpackssdw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32> %a0, <8 x i32> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32>, <8 x i32>) nounwind readnone

define <16 x i8> @stack_fold_packsswb(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packsswb
  ;CHECK:       vpacksswb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16>, <8 x i16>) nounwind readnone

define <32 x i8> @stack_fold_packsswb_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packsswb_ymm
  ;CHECK:       vpacksswb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16> %a0, <16 x i16> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_packusdw(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packusdw
  ;CHECK:       vpackusdw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32> %a0, <4 x i32> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32>, <4 x i32>) nounwind readnone

define <16 x i16> @stack_fold_packusdw_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packusdw_ymm
  ;CHECK:       vpackusdw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32> %a0, <8 x i32> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32>, <8 x i32>) nounwind readnone

define <16 x i8> @stack_fold_packuswb(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packuswb
  ;CHECK:       vpackuswb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>) nounwind readnone

define <32 x i8> @stack_fold_packuswb_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packuswb_ymm
  ;CHECK:       vpackuswb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %a0, <16 x i16> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_paddb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddb
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <16 x i8> %a0, %a1
  ret <16 x i8> %2
}

define <16 x i8> @stack_fold_paddb_mask(<16 x i8> %a0, <16 x i8> %a1, <16 x i8>* %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_paddb_mask
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <16 x i8> %a0, %a1
  %3 = bitcast i16 %mask to <16 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <16 x i8>, <16 x i8>* %a2
  %5 = select <16 x i1> %3, <16 x i8> %2, <16 x i8> %4
  ret <16 x i8> %5
}

define <16 x i8> @stack_fold_paddb_maskz(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_paddb_maskz
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <16 x i8> %a0, %a1
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i8> %2, <16 x i8> zeroinitializer
  ret <16 x i8> %4
}

define <32 x i8> @stack_fold_paddb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddb_ymm
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <32 x i8> %a0, %a1
  ret <32 x i8> %2
}

define <32 x i8> @stack_fold_paddb_mask_ymm(<32 x i8> %a0, <32 x i8> %a1, <32 x i8>* %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_paddb_mask_ymm
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <32 x i8> %a0, %a1
  %3 = bitcast i32 %mask to <32 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <32 x i8>, <32 x i8>* %a2
  %5 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> %4
  ret <32 x i8> %5
}

define <32 x i8> @stack_fold_paddb_maskz_ymm(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_paddb_maskz_ymm
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <32 x i8> %a0, %a1
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> zeroinitializer
  ret <32 x i8> %4
}

define <4 x i32> @stack_fold_paddd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_paddd
  ;CHECK:       vpaddd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <4 x i32> %a0, %a1
  ret <4 x i32> %2
}

define <8 x i32> @stack_fold_paddd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_paddd_ymm
  ;CHECK:       vpaddd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <8 x i32> %a0, %a1
  ret <8 x i32> %2
}

define <2 x i64> @stack_fold_paddq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_paddq
  ;CHECK:       vpaddq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <2 x i64> %a0, %a1
  ret <2 x i64> %2
}

define <4 x i64> @stack_fold_paddq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_paddq_ymm
  ;CHECK:       vpaddq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <4 x i64> %a0, %a1
  ret <4 x i64> %2
}

define <16 x i8> @stack_fold_paddsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddsb
  ;CHECK:       vpaddsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_paddsb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddsb_ymm
  ;CHECK:       vpaddsb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.padds.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.padds.b(<32 x i8>, <32 x i8>) nounwind readnone

define <8 x i16> @stack_fold_paddsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddsw
  ;CHECK:       vpaddsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_paddsw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddsw_ymm
  ;CHECK:       vpaddsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.padds.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.padds.w(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_paddusb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddusb
  ;CHECK:       vpaddusb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_paddusb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddusb_ymm
  ;CHECK:       vpaddusb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8>, <32 x i8>) nounwind readnone

define <8 x i16> @stack_fold_paddusw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddusw
  ;CHECK:       vpaddusw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_paddusw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddusw_ymm
  ;CHECK:       vpaddusw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_paddw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddw
  ;CHECK:       vpaddw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <8 x i16> %a0, %a1
  ret <8 x i16> %2
}

define <16 x i16> @stack_fold_paddw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddw_ymm
  ;CHECK:       vpaddw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <16 x i16> %a0, %a1
  ret <16 x i16> %2
}

define <32 x i8> @stack_fold_palignr(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_palignr
  ;CHECK:       vpalignr $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a1, <32 x i8> %a0, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  ret <32 x i8> %2
}

define <32 x i8> @stack_fold_palignr_mask(<32 x i8> %a0, <32 x i8> %a1, <32 x i8>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_palignr_mask
  ;CHECK:       vpalignr $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a1, <32 x i8> %a0, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x i8>, <32 x i8>* %passthru
  %5 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> %4
  ret <32 x i8> %5
}

define <32 x i8> @stack_fold_palignr_maskz(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_palignr_maskz
  ;CHECK:       vpalignr $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a1, <32 x i8> %a0, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> zeroinitializer
  ret <32 x i8> %4
}

define i16 @stack_fold_pcmpeqb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqb
  ;CHECK:       vpcmpeqb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <16 x i8> %a0, %a1
  %3 = bitcast <16 x i1> %2 to i16
  ret i16 %3
}

define i8 @stack_fold_pcmpeqd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqd
  ;CHECK:       vpcmpeqd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <4 x i32> %a0, %a1
  %3 = shufflevector <4 x i1> %2, <4 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %4 = bitcast <8 x i1> %3 to i8
  ret i8 %4
}

define i8 @stack_fold_pcmpeqq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqq
  ;CHECK:       vpcmpeqq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <2 x i64> %a0, %a1
  %3 = shufflevector <2 x i1> %2, <2 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  %4 = bitcast <8 x i1> %3 to i8
  ret i8 %4
}

define i8 @stack_fold_pcmpeqw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqw
  ;CHECK:       vpcmpeqw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <8 x i16> %a0, %a1
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}

define <32 x i8> @stack_fold_permbvar(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_permbvar
  ;CHECK:   vpermb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx512.mask.permvar.qi.256(<32 x i8> %a1, <32 x i8> %a0, <32 x i8> undef, i32 -1)
  ; add forces execution domain
  %3 = add <32 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <32 x i8> %3
}
declare <32 x i8> @llvm.x86.avx512.mask.permvar.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32) nounwind readonly

define <8 x i32> @stack_fold_permd(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permd
  ;CHECK:   vpermd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %a1, <8 x i32> %a0)
  ; add forces execution domain
  %3 = add <8 x i32> %2, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %3
}
declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>) nounwind readonly

define <16 x i8> @stack_fold_vpermi2b(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2b
  ;CHECK:       vpermi2b {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @stack_fold_vpermi2b_ymm(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2b_ymm
  ;CHECK:       vpermi2b {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <32 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <4 x i32> @stack_fold_vpermi2d(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2d
  ;CHECK:       vpermi2d {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @stack_fold_vpermi2d_ymm(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2d_ymm
  ;CHECK:       vpermi2d {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <2 x i64> @stack_fold_vpermi2q(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2q
  ;CHECK:       vpermi2q {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x i64> @llvm.x86.avx512.mask.vpermt2var.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.vpermt2var.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <4 x i64> @stack_fold_vpermi2q_ymm(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2q_ymm
  ;CHECK:       vpermi2q {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i64> @llvm.x86.avx512.mask.vpermt2var.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.vpermt2var.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <8 x i16> @stack_fold_vpermi2w(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2w
  ;CHECK:       vpermi2w {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @stack_fold_vpermi2w_ymm(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2w_ymm
  ;CHECK:       vpermi2w {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <4 x i64> @stack_fold_permq(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_permq
  ;CHECK:   vpermq $235, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i64> %a0, <4 x i64> undef, <4 x i32> <i32 3, i32 2, i32 2, i32 3>
  ; add forces execution domain
  %3 = add <4 x i64> %2, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %3
}

define <4 x i64> @stack_fold_permqvar(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permqvar
  ;CHECK:   vpermq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64> %a1, <4 x i64> %a0, <4 x i64> undef, i8 -1)
  ; add forces execution domain
  %3 = add <4 x i64> %2, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %3
}
declare <4 x i64> @llvm.x86.avx512.mask.permvar.di.256(<4 x i64>, <4 x i64>, <4 x i64>, i8) nounwind readonly

define <16 x i8> @stack_fold_vpermt2b(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2b
  ;CHECK:       vpermt2b {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @stack_fold_vpermt2b_ymm(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2b_ymm
  ;CHECK:       vpermt2b {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <32 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <4 x i32> @stack_fold_vpermt2d(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2d
  ;CHECK:       vpermt2d {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpermi2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.vpermi2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @stack_fold_vpermt2d_ymm(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2d_ymm
  ;CHECK:       vpermt2d {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpermi2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.vpermi2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <2 x i64> @stack_fold_vpermt2q(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2q
  ;CHECK:       vpermt2q {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x i64> @llvm.x86.avx512.mask.vpermi2var.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.vpermi2var.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <4 x i64> @stack_fold_vpermt2q_ymm(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2q_ymm
  ;CHECK:       vpermt2q {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i64> @llvm.x86.avx512.mask.vpermi2var.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.vpermi2var.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <8 x i16> @stack_fold_vpermt2w(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2w
  ;CHECK:       vpermt2w {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @stack_fold_vpermt2w_ymm(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2w_ymm
  ;CHECK:       vpermt2w {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16> @stack_fold_permwvar(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_permwvar
  ;CHECK:   vpermw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16> %a1, <16 x i16> %a0, <16 x i16> undef, i16 -1)
  ; add forces execution domain
  %3 = add <16 x i16> %2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %3
}
declare <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16) nounwind readonly

define <4 x i32> @stack_fold_vplzcntd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vplzcntd
  ;CHECK:       vplzcntd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>) nounwind readonly

define <8 x i32> @stack_fold_vplzcntd_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vplzcntd_ymm
  ;CHECK:       vplzcntd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a0)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.ctlz.v8i32(<8 x i32>) nounwind readonly

define <2 x i64> @stack_fold_vplzcntq(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vplzcntq
  ;CHECK:       vplzcntq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_vplzcntq_ymm(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vplzcntq_ymm
  ;CHECK:       vplzcntq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a0)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.ctlz.v4i64(<4 x i64>) nounwind readnone

define <8 x i16> @stack_fold_pmaddubsw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddubsw
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a0, <16 x i8> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_pmaddubsw_mask(<8 x i16>* %passthru, <16 x i8> %a0, <16 x i8> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddubsw_mask
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a0, <16 x i8> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <8 x i16>, <8 x i16>* %passthru
  %5 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> %4
  ret <8 x i16> %5
}

define <8 x i16> @stack_fold_pmaddubsw_maskz(<16 x i8> %a0, <16 x i8> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddubsw_maskz
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a0, <16 x i8> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> zeroinitializer
  ret <8 x i16> %4
}

define <16 x i16> @stack_fold_pmaddubsw_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddubsw_ymm
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %a0, <32 x i8> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>) nounwind readnone

define <16 x i16> @stack_fold_pmaddubsw_ymm_mask(<16 x i16>* %passthru, <32 x i8> %a0, <32 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddubsw_ymm_mask
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %a0, <32 x i8> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <16 x i16>, <16 x i16>* %passthru
  %5 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> %4
  ret <16 x i16> %5
}

define <16 x i16> @stack_fold_pmaddubsw_ymm_maskz(<32 x i8> %a0, <32 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddubsw_ymm_maskz
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %a0, <32 x i8> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> zeroinitializer
  ret <16 x i16> %4
}

define <4 x i32> @stack_fold_pmaddwd(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddwd
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a0, <8 x i16> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_pmaddwd_mask(<4 x i32>* %passthru, <8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddwd_mask
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a0, <8 x i16> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; load needed to keep the operation from being scheduled about the asm block
  %5 = load <4 x i32>, <4 x i32>* %passthru
  %6 = select <4 x i1> %4, <4 x i32> %2, <4 x i32> %5
  ret <4 x i32> %6
}

define <4 x i32> @stack_fold_pmaddwd_maskz(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddwd_maskz
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a0, <8 x i16> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x i32> %2, <4 x i32> zeroinitializer
  ret <4 x i32> %5
}

define <8 x i32> @stack_fold_pmaddwd_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddwd_ymm
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %a0, <16 x i16> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i32> @stack_fold_pmaddwd_ymm_mask(<8 x i32>* %passthru, <16 x i16> %a0, <16 x i16> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddwd_ymm_mask
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %a0, <16 x i16> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <8 x i32>, <8 x i32>* %passthru
  %5 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> %4
  ret <8 x i32> %5
}

define <8 x i32> @stack_fold_pmaddwd_ymm_maskz(<16 x i16> %a0, <16 x i16> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaddwd_ymm_maskz
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %a0, <16 x i16> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> zeroinitializer
  ret <8 x i32> %4
}

define <16 x i8> @stack_fold_pmaxsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsb
  ;CHECK:       vpmaxsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_pmaxsb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsb_ymm
  ;CHECK:       vpmaxsb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.pmaxs.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.pmaxs.b(<32 x i8>, <32 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pmaxsd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsd
  ;CHECK:       vpmaxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pmaxsd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsd_ymm
  ;CHECK:       vpmaxsd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmaxs.d(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pmaxs.d(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pmaxsq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsq
  ;CHECK:       vpmaxsq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pmaxs.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.pmaxs.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_pmaxsq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsq_ymm
  ;CHECK:       vpmaxsq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pmaxs.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.pmaxs.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8) nounwind readnone

define <8 x i16> @stack_fold_pmaxsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsw
  ;CHECK:       vpmaxsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_pmaxsw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsw_ymm
  ;CHECK:       vpmaxsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmaxs.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pmaxs.w(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pmaxub(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxub
  ;CHECK:       vpmaxub {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_pmaxub_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxub_ymm
  ;CHECK:       vpmaxub {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.pmaxu.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.pmaxu.b(<32 x i8>, <32 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pmaxud(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxud
  ;CHECK:       vpmaxud {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pmaxud_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxud_ymm
  ;CHECK:       vpmaxud {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmaxu.d(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pmaxu.d(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pmaxuq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxuq
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8) nounwind readnone

define <2 x i64> @stack_fold_pmaxuq_mask(<2 x i64>* %passthru, <2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaxuq_mask
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <2 x i64>, <2 x i64>* %passthru
  %3 = call <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %2, i8 %mask)
  ret <2 x i64> %3
}

define <2 x i64> @stack_fold_pmaxuq_maskz(<2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaxuq_maskz
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pmaxu.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %2
}

define <4 x i64> @stack_fold_pmaxuq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxuq_ymm
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_pmaxuq_ymm_mask(<4 x i64>* %passthru, <4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaxuq_ymm_mask
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <4 x i64>, <4 x i64>* %passthru
  %3 = call <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %2, i8 %mask)
  ret <4 x i64> %3
}

define <4 x i64> @stack_fold_pmaxuq_ymm_maskz(<4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmaxuq_ymm_maskz
  ;CHECK:       vpmaxuq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pmaxu.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %2
}

define <8 x i16> @stack_fold_pmaxuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxuw
  ;CHECK:       vpmaxuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_pmaxuw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxuw_ymm
  ;CHECK:       vpmaxuw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmaxu.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pmaxu.w(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pminsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminsb
  ;CHECK:       vpminsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_pminsb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminsb_ymm
  ;CHECK:       vpminsb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.pmins.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.pmins.b(<32 x i8>, <32 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pminsd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminsd
  ;CHECK:       vpminsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pminsd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminsd_ymm
  ;CHECK:       vpminsd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pmins.d(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pmins.d(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pminsq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pminsq
  ;CHECK:       vpminsq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pmins.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.pmins.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_pminsq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pminsq_ymm
  ;CHECK:       vpminsq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pmins.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.pmins.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8) nounwind readnone

define <8 x i16> @stack_fold_pminsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminsw
  ;CHECK:       vpminsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_pminsw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminsw_ymm
  ;CHECK:       vpminsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pmins.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pmins.w(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pminub(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminub
  ;CHECK:       vpminub {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_pminub_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminub_ymm
  ;CHECK:       vpminub {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.pminu.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.pminu.b(<32 x i8>, <32 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pminud(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminud
  ;CHECK:       vpminud {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pminud_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminud_ymm
  ;CHECK:       vpminud {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.pminu.d(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.pminu.d(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pminuq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pminuq
  ;CHECK:       vpminuq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.mask.pminu.q.128(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> undef, i8 -1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.mask.pminu.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i64> @stack_fold_pminuq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pminuq_ymm
  ;CHECK:       vpminuq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.mask.pminu.q.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> undef, i8 -1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.mask.pminu.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8) nounwind readnone

define <8 x i16> @stack_fold_pminuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminuw
  ;CHECK:       vpminuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_pminuw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminuw_ymm
  ;CHECK:       vpminuw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.pminu.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.pminu.w(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_vpmovdw(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovdw
  ;CHECK:       vpmovdw %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmov.dw.256(<8 x i32>, <8 x i16>, i8)

define <4 x i32> @stack_fold_vpmovqd(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovqd
  ;CHECK:       vpmovqd %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64> %a0, <4 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %1
}
declare <4 x i32> @llvm.x86.avx512.mask.pmov.qd.256(<4 x i64>, <4 x i32>, i8)

define <16 x i8> @stack_fold_vpmovwb(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovwb
  ;CHECK:       vpmovwb %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmov.wb.256(<16 x i16>, <16 x i8>, i16)

define <8 x i16> @stack_fold_vpmovsdw(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsdw
  ;CHECK:       vpmovsdw %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmovs.dw.256(<8 x i32>, <8 x i16>, i8)

define <4 x i32> @stack_fold_vpmovsqd(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsqd
  ;CHECK:       vpmovsqd %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64> %a0, <4 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %1
}
declare <4 x i32> @llvm.x86.avx512.mask.pmovs.qd.256(<4 x i64>, <4 x i32>, i8)

define <16 x i8> @stack_fold_vpmovswb(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovswb
  ;CHECK:       vpmovswb %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmovs.wb.256(<16 x i16>, <16 x i8>, i16)

define <4 x i32> @stack_fold_pmovsxbd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbd
  ;CHECK:       vpmovsxbd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i32> @stack_fold_pmovsxbd_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbd_ymm
  ;CHECK:       vpmovsxbd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <2 x i64> @stack_fold_pmovsxbq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbq
  ;CHECK:       vpmovsxbq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i8> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovsxbq_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbq_ymm
  ;CHECK:       pmovsxbq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i16> @stack_fold_pmovsxbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbw
  ;CHECK:       vpmovsxbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i16>
  ret <8 x i16> %3
}

define <16 x i16> @stack_fold_pmovsxbw_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbw_ymm
  ;CHECK:       vpmovsxbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <16 x i8> %a0 to <16 x i16>
  ret <16 x i16> %2
}

define <2 x i64> @stack_fold_pmovsxdq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxdq
  ;CHECK:       vpmovsxdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovsxdq_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxdq_ymm
  ;CHECK:       vpmovsxdq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <4 x i32> %a0 to <4 x i64>
  ret <4 x i64> %2
}

define <4 x i32> @stack_fold_pmovsxwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwd
  ;CHECK:       vpmovsxwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i32> @stack_fold_pmovsxwd_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwd_ymm
  ;CHECK:       vpmovsxwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <8 x i16> %a0 to <8 x i32>
  ret <8 x i32> %2
}

define <2 x i64> @stack_fold_pmovsxwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwq
  ;CHECK:       vpmovsxwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i16> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovsxwq_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwq_ymm
  ;CHECK:       vpmovsxwq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i16> @stack_fold_vpmovusdw(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusdw
  ;CHECK:       vpmovusdw %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmovus.dw.256(<8 x i32>, <8 x i16>, i8)

define <4 x i32> @stack_fold_vpmovusqd(<4 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusqd
  ;CHECK:       vpmovusqd %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64> %a0, <4 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %1
}
declare <4 x i32> @llvm.x86.avx512.mask.pmovus.qd.256(<4 x i64>, <4 x i32>, i8)

define <16 x i8> @stack_fold_vpmovuswb(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovuswb
  ;CHECK:       vpmovuswb %ymm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmovus.wb.256(<16 x i16>, <16 x i8>, i16)

define <4 x i32> @stack_fold_pmovzxbd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbd
  ;CHECK:       vpmovzxbd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 17, i32 18, i32 1, i32 19, i32 20, i32 21, i32 2, i32 22, i32 23, i32 24, i32 3, i32 25, i32 26, i32 27>
  %3 = bitcast <16 x i8> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i32> @stack_fold_pmovzxbd_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbd_ymm
  ;CHECK:       vpmovzxbd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = zext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <2 x i64> @stack_fold_pmovzxbq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbq
  ;CHECK:       vpmovzxbq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 1, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
  %3 = bitcast <16 x i8> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovzxbq_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbq_ymm
  ;CHECK:       vpmovzxbq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i16> @stack_fold_pmovzxbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbw
  ;CHECK:       vpmovzxbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %3 = bitcast <16 x i8> %2 to <8 x i16>
  ret <8 x i16> %3
}

define <16 x i16> @stack_fold_pmovzxbw_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbw_ymm
  ;CHECK:       vpmovzxbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <16 x i8> %a0 to <16 x i16>
  ret <16 x i16> %2
}

define <2 x i64> @stack_fold_pmovzxdq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxdq
  ;CHECK:       vpmovzxdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %3 = bitcast <4 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovzxdq_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxdq_ymm
  ;CHECK:       vpmovzxdq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <4 x i32> %a0 to <4 x i64>
  ret <4 x i64> %2
}

define <4 x i32> @stack_fold_pmovzxwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwd
  ;CHECK:       vpmovzxwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %3 = bitcast <8 x i16> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i32> @stack_fold_pmovzxwd_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwd_ymm
  ;CHECK:       vpmovzxwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <8 x i16> %a0 to <8 x i32>
  ret <8 x i32> %2
}

define <2 x i64> @stack_fold_pmovzxwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwq
  ;CHECK:       vpmovzxwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 9, i32 10, i32 1, i32 11, i32 12, i32 13>
  %3 = bitcast <8 x i16> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i64> @stack_fold_pmovzxwq_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwq_ymm
  ;CHECK:       vpmovzxwq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i16> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @stack_fold_pmovzxwq_maskz_ymm(<8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmovzxwq_maskz_ymm
  ;CHECK:       vpmovzxwq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i16> %2 to <4 x i64>
  %4 = bitcast i8 %mask to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x i64> %3, <4 x i64> zeroinitializer
  ret <4 x i64> %6
}

define <4 x i64> @stack_fold_pmovzxwq_mask_ymm(<4 x i64> %passthru, <8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmovzxwq_mask_ymm
  ;CHECK:       vpmovzxwq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i16> %2 to <4 x i64>
  %4 = bitcast i8 %mask to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x i64> %3, <4 x i64> %passthru
  ret <4 x i64> %6
}

define <2 x i64> @stack_fold_pmuldq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuldq
  ;CHECK:       vpmuldq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32> %a0, <4 x i32> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i64> @stack_fold_pmuldq_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuldq_ymm
  ;CHECK:       vpmuldq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.pmul.dq(<8 x i32> %a0, <8 x i32> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.pmul.dq(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_pmuludq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuludq
  ;CHECK:       vpmuludq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32> %a0, <4 x i32> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i64> @stack_fold_pmuludq_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuludq_ymm
  ;CHECK:       vpmuludq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32> %a0, <8 x i32> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32>, <8 x i32>) nounwind readnone

define <4 x i64> @stack_fold_pmuludq_ymm_mask(<4 x i64>* %passthru, <8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmuludq_ymm_mask
  ;CHECK:       vpmuludq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32> %a0, <8 x i32> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = load <4 x i64>, <4 x i64>* %passthru
  %6 = select <4 x i1> %4, <4 x i64> %2, <4 x i64> %5
  ret <4 x i64> %6
}

define <4 x i64> @stack_fold_pmuludq_ymm_maskz(<8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pmuludq_ymm_maskz
  ;CHECK:       vpmuludq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.pmulu.dq(<8 x i32> %a0, <8 x i32> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x i64> %2, <4 x i64> zeroinitializer
  ret <4 x i64> %5
}

define <2 x i64> @stack_fold_psadbw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psadbw
  ;CHECK:       vpsadbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %a0, <16 x i8> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i64> @stack_fold_psadbw_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psadbw_ymm
  ;CHECK:       vpsadbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %a0, <32 x i8> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>) nounwind readnone

define <16 x i8> @stack_fold_pshufb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pshufb
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> undef, i16 -1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16) nounwind readnone

define <16 x i8> @stack_fold_pshufb_mask(<16 x i8>* %passthru, <16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshufb_mask
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x i8>, <16 x i8>* %passthru
  %3 = call <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %2, i16 %mask)
  ret <16 x i8> %3
}

define <16 x i8> @stack_fold_pshufb_maskz(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshufb_maskz
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.avx512.mask.pshuf.b.128(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %2
}

define <32 x i8> @stack_fold_pshufb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pshufb_ymm
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8> %a0, <32 x i8> %a1, <32 x i8> undef, i32 -1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <32 x i8> @stack_fold_pshufb_ymm_mask(<32 x i8>* %passthru, <32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_pshufb_ymm_mask
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x i8>, <32 x i8>* %passthru
  %3 = call <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8> %a0, <32 x i8> %a1, <32 x i8> %2, i32 %mask)
  ret <32 x i8> %3
}

define <32 x i8> @stack_fold_pshufb_ymm_maskz(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_pshufb_ymm_maskz
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx512.mask.pshuf.b.256(<32 x i8> %a0, <32 x i8> %a1, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %2
}

define <4 x i32> @stack_fold_pshufd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pshufd
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %3 = add <4 x i32> %2, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %3
}

define <4 x i32> @stack_fold_pshufd_mask(<4 x i32> %passthru, <4 x i32> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufd_mask
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x i32> %2, <4 x i32> %passthru
  ret <4 x i32> %5
}

define <4 x i32> @stack_fold_pshufd_maskz(<4 x i32> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufd_maskz
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x i32> %2, <4 x i32> zeroinitializer
  ret <4 x i32> %5
}

define <8 x i32> @stack_fold_pshufd_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pshufd_ymm
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a0, <8 x i32> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  %3 = add <8 x i32> %2, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %3
}

define <8 x i32> @stack_fold_pshufd_ymm_mask(<8 x i32> %passthru, <8 x i32> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufd_ymm_mask
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a0, <8 x i32> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> %passthru
  ret <8 x i32> %4
}

define <8 x i32> @stack_fold_pshufd_ymm_maskz(<8 x i32> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufd_ymm_maskz
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i32> %a0, <8 x i32> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> zeroinitializer
  ret <8 x i32> %4
}

define <8 x i16> @stack_fold_pshufhw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshufhw
  ;CHECK:       vpshufhw $11, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 4, i32 4>
  ret <8 x i16> %2
}

define <8 x i16> @stack_fold_pshufhw_mask(<8 x i16> %passthru, <8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufhw_mask
  ;CHECK:       vpshufhw $11, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 4, i32 4>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> %passthru
  ret <8 x i16> %4
}

define <8 x i16> @stack_fold_pshufhw_maskz(<8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshufhw_maskz
  ;CHECK:       vpshufhw $11, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 4, i32 4>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> zeroinitializer
  ret <8 x i16> %4
}

define <16 x i16> @stack_fold_pshufhw_ymm(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshufhw_ymm
  ;CHECK:       vpshufhw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 5, i32 4, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i16> %2
}

define <16 x i16> @stack_fold_pshufhw_ymm_mask(<16 x i16> %passthru, <16 x i16> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshufhw_ymm_mask
  ;CHECK:       vpshufhw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 5, i32 4, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 13, i32 12>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> %passthru
  ret <16 x i16> %4
}

define <16 x i16> @stack_fold_pshufhw_ymm_maskz(<16 x i16> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshufhw_ymm_maskz
  ;CHECK:       vpshufhw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 5, i32 4, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 13, i32 12>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> zeroinitializer
  ret <16 x i16> %4
}

define <8 x i16> @stack_fold_pshuflw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshuflw
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %2
}

define <8 x i16> @stack_fold_pshuflw_mask(<8 x i16> %passthru, <8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshuflw_mask
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> %passthru
  ret <8 x i16> %4
}

define <8 x i16> @stack_fold_pshuflw_maskz(<8 x i16> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_pshuflw_maskz
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> zeroinitializer
  ret <8 x i16> %4
}

define <16 x i16> @stack_fold_pshuflw_ymm(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshuflw_ymm
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7, i32 11, i32 10, i32 9, i32 8, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %2
}

define <16 x i16> @stack_fold_pshuflw_ymm_mask(<16 x i16> %passthru, <16 x i16> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshuflw_ymm_mask
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7, i32 11, i32 10, i32 9, i32 8, i32 12, i32 13, i32 14, i32 15>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> %passthru
  ret <16 x i16> %4
}

define <16 x i16> @stack_fold_pshuflw_ymm_maskz(<16 x i16> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_pshuflw_ymm_maskz
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i16> %a0, <16 x i16> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7, i32 11, i32 10, i32 9, i32 8, i32 12, i32 13, i32 14, i32 15>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> zeroinitializer
  ret <16 x i16> %4
}

define <4 x i32> @stack_fold_pslld(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pslld
  ;CHECK:       vpslld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_pslld_ymm(<8 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pslld_ymm
  ;CHECK:       vpslld {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psll.d(<8 x i32> %a0, <4 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psll.d(<8 x i32>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_pslldq(<16 x i8> %a) {
  ;CHECK-LABEL: stack_fold_pslldq
  ;CHECK:       vpslldq $12, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> zeroinitializer, <16 x i8> %a, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 17, i32 18, i32 19>
  ret <16 x i8> %2
}

define <32 x i8> @stack_fold_pslldq_ymm(<32 x i8> %a) {
  ;CHECK-LABEL: stack_fold_pslldq_ymm
  ;CHECK:       vpslldq $15, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> zeroinitializer, <32 x i8> %a, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 32, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 48>
  ret <32 x i8> %2
}

define <2 x i64> @stack_fold_psllq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psllq
  ;CHECK:       vpsllq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psllq_ymm(<4 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psllq_ymm
  ;CHECK:       vpsllq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.psll.q(<4 x i64> %a0, <2 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.psll.q(<4 x i64>, <2 x i64>) nounwind readnone

define <4 x i32> @stack_fold_psllvd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psllvd
  ;CHECK:       vpsllvd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_psllvd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psllvd_ymm
  ;CHECK:       vpsllvd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psllvq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psllvq
  ;CHECK:       vpsllvq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx2.psllv.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx2.psllv.q(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psllvq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psllvq_ymm
  ;CHECK:       vpsllvq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.psllv.q.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.psllv.q.256(<4 x i64>, <4 x i64>) nounwind readnone

define <8 x i16> @stack_fold_psllvw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psllvw
  ;CHECK:       vpsllvw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512.psllv.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512.psllv.w.128(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psllvw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psllvw_ymm
  ;CHECK:       vpsllvw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512.psllv.w.256(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx512.psllv.w.256(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_psllw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psllw
  ;CHECK:       vpsllw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psllw_ymm(<16 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psllw_ymm
  ;CHECK:       vpsllw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.psll.w(<16 x i16> %a0, <8 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.psll.w(<16 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_psrad(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrad
  ;CHECK:       vpsrad {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_psrad_ymm(<8 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrad_ymm
  ;CHECK:       vpsrad {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32> %a0, <4 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psraq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psraq
  ;CHECK:       vpsraq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.psra.q.128(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.psra.q.128(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psraq_ymm(<4 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psraq_ymm
  ;CHECK:       vpsraq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.psra.q.256(<4 x i64> %a0, <2 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.psra.q.256(<4 x i64>, <2 x i64>) nounwind readnone

define <4 x i32> @stack_fold_psravd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psravd
  ;CHECK:       vpsravd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx2.psrav.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx2.psrav.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_psravd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psravd_ymm
  ;CHECK:       vpsravd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psrav.d.256(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psrav.d.256(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psravq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psravq
  ;CHECK:       vpsravq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx512.psrav.q.128(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx512.psrav.q.128(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psravq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psravq_ymm
  ;CHECK:       vpsravq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx512.psrav.q.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx512.psrav.q.256(<4 x i64>, <4 x i64>) nounwind readnone

define <8 x i16> @stack_fold_psravw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psravw
  ;CHECK:       vpsravw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512.psrav.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512.psrav.w.128(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psravw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psravw_ymm
  ;CHECK:       vpsravw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512.psrav.w.256(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx512.psrav.w.256(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_psraw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psraw
  ;CHECK:       vpsraw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psraw_ymm(<16 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psraw_ymm
  ;CHECK:       vpsraw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16> %a0, <8 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_psrld(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrld
  ;CHECK:       vpsrld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_psrld_ymm(<8 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrld_ymm
  ;CHECK:       vpsrld {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32> %a0, <4 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_psrldq(<16 x i8> %a) {
  ;CHECK-LABEL: stack_fold_psrldq
  ;CHECK:       vpsrldq $12, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> zeroinitializer, <16 x i8> %a, <16 x i32> <i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 09, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i8> %2
}

define <32 x i8> @stack_fold_psrldq_ymm(<32 x i8> %a) {
  ;CHECK-LABEL: stack_fold_psrldq_ymm
  ;CHECK:       vpsrldq $15, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> zeroinitializer, <32 x i8> %a, <32 x i32> <i32 47, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 63, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %2
}

define <2 x i64> @stack_fold_psrlq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psrlq
  ;CHECK:       vpsrlq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psrlq_ymm(<4 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psrlq_ymm
  ;CHECK:       vpsrlq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64> %a0, <2 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64>, <2 x i64>) nounwind readnone

define <4 x i32> @stack_fold_psrlvd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvd
  ;CHECK:       vpsrlvd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx2.psrlv.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx2.psrlv.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i32> @stack_fold_psrlvd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvd_ymm
  ;CHECK:       vpsrlvd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx2.psrlv.d.256(<8 x i32> %a0, <8 x i32> %a1)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx2.psrlv.d.256(<8 x i32>, <8 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psrlvq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvq
  ;CHECK:       vpsrlvq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.avx2.psrlv.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.avx2.psrlv.q(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_psrlvq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvq_ymm
  ;CHECK:       vpsrlvq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.avx2.psrlv.q.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.avx2.psrlv.q.256(<4 x i64>, <4 x i64>) nounwind readnone

define <8 x i16> @stack_fold_psrlvw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvw
  ;CHECK:       vpsrlvw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512.psrlv.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512.psrlv.w.128(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psrlvw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psrlvw_ymm
  ;CHECK:       vpsrlvw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512.psrlv.w.256(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx512.psrlv.w.256(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_psrlw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psrlw
  ;CHECK:       vpsrlw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psrlw_ymm(<16 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psrlw_ymm
  ;CHECK:       vpsrlw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16> %a0, <8 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_psubb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubb
  ;CHECK:       vpsubb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <16 x i8> %a0, %a1
  ret <16 x i8> %2
}

define <32 x i8> @stack_fold_psubb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubb_ymm
  ;CHECK:       vpsubb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <32 x i8> %a0, %a1
  ret <32 x i8> %2
}

define <4 x i32> @stack_fold_psubd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psubd
  ;CHECK:       vpsubd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <4 x i32> %a0, %a1
  ret <4 x i32> %2
}

define <8 x i32> @stack_fold_psubd_ymm(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psubd_ymm
  ;CHECK:       vpsubd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <8 x i32> %a0, %a1
  ret <8 x i32> %2
}

define <2 x i64> @stack_fold_psubq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psubq
  ;CHECK:       vpsubq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <2 x i64> %a0, %a1
  ret <2 x i64> %2
}

define <4 x i64> @stack_fold_psubq_ymm(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psubq_ymm
  ;CHECK:       vpsubq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <4 x i64> %a0, %a1
  ret <4 x i64> %2
}

define <16 x i8> @stack_fold_psubsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubsb
  ;CHECK:       vpsubsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_psubsb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubsb_ymm
  ;CHECK:       vpsubsb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.psubs.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.psubs.b(<32 x i8>, <32 x i8>) nounwind readnone

define <8 x i16> @stack_fold_psubsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubsw
  ;CHECK:       vpsubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psubsw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubsw_ymm
  ;CHECK:       vpsubsw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.psubs.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.psubs.w(<16 x i16>, <16 x i16>) nounwind readnone

define <16 x i8> @stack_fold_psubusb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubusb
  ;CHECK:       vpsubusb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8>, <16 x i8>) nounwind readnone

define <32 x i8> @stack_fold_psubusb_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubusb_ymm
  ;CHECK:       vpsubusb {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %2
}
declare <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8>, <32 x i8>) nounwind readnone

define <8 x i16> @stack_fold_psubusw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubusw
  ;CHECK:       vpsubusw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @stack_fold_psubusw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubusw_ymm
  ;CHECK:       vpsubusw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16> %a0, <16 x i16> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16>, <16 x i16>) nounwind readnone

define <8 x i16> @stack_fold_psubw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubw
  ;CHECK:       vpsubw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <8 x i16> %a0, %a1
  ret <8 x i16> %2
}

define <16 x i16> @stack_fold_psubw_ymm(<16 x i16> %a0, <16 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubw_ymm
  ;CHECK:       vpsubw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <16 x i16> %a0, %a1
  ret <16 x i16> %2
}

define <16 x i8> @stack_fold_punpckhbw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhbw
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i8> %2
}

define <16 x i8> @stack_fold_punpckhbw_mask(<16 x i8>* %passthru, <16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_punpckhbw_mask
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %3 = bitcast i16 %mask to <16 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <16 x i8>, <16 x i8>* %passthru
  %5 = select <16 x i1> %3, <16 x i8> %2, <16 x i8> %4
  ret <16 x i8> %5
}

define <16 x i8> @stack_fold_punpckhbw_maskz(<16 x i8> %passthru, <16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_punpckhbw_maskz
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i8> %2, <16 x i8> zeroinitializer
  ret <16 x i8> %4
}

define <32 x i8> @stack_fold_punpckhbw_ymm(<32 x i8> %a0, <32 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhbw_ymm
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a0, <32 x i8> %a1, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  ret <32 x i8> %2
}

define <32 x i8> @stack_fold_punpckhbw_mask_ymm(<32 x i8>* %passthru, <32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_punpckhbw_mask_ymm
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a0, <32 x i8> %a1, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  %3 = bitcast i32 %mask to <32 x i1>
  ; load needed to keep the operation from being scheduled about the asm block
  %4 = load <32 x i8>, <32 x i8>* %passthru
  %5 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> %4
  ret <32 x i8> %5
}

define <32 x i8> @stack_fold_punpckhbw_maskz_ymm(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_punpckhbw_maskz_ymm
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <32 x i8> %a0, <32 x i8> %a1, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x i8> %2, <32 x i8> zeroinitializer
  ret <32 x i8> %4
}
