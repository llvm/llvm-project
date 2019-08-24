; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512dq,+avx512vbmi,+avx512cd,+avx512vpopcntdq,+avx512vnni < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <16 x i32> @stack_fold_vpdpwssd(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpdpwssd:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2)
  ret <16 x i32> %2
}

define <16 x i32> @stack_fold_vpdpwssd_commuted(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpdpwssd_commuted:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %a0, <16 x i32> %a2, <16 x i32> %a1)
  ret <16 x i32> %2
}

define <16 x i32> @stack_fold_vpdpwssd_mask(<16 x i32>* %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssd_mask:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x i32>, <16 x i32>* %a0
  %3 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %2, <16 x i32> %a1, <16 x i32> %a2)
  %4 = bitcast i16 %mask to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %3, <16 x i32> %2
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssd_mask_commuted(<16 x i32>* %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssd_mask_commuted:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x i32>, <16 x i32>* %a0
  %3 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %2, <16 x i32> %a2, <16 x i32> %a1)
  %4 = bitcast i16 %mask to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %3, <16 x i32> %2
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssd_maskz(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssd_maskz:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %2, <16 x i32> zeroinitializer
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssd_maskz_commuted(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssd_maskz_commuted:
  ;CHECK:       vpdpwssd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %a0, <16 x i32> %a2, <16 x i32> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %2, <16 x i32> zeroinitializer
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssds(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpdpwssds:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2)
  ret <16 x i32> %2
}

define <16 x i32> @stack_fold_vpdpwssds_commuted(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpdpwssds_commuted:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %a0, <16 x i32> %a2, <16 x i32> %a1)
  ret <16 x i32> %2
}

define <16 x i32> @stack_fold_vpdpwssds_mask(<16 x i32>* %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssds_mask:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x i32>, <16 x i32>* %a0
  %3 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %2, <16 x i32> %a1, <16 x i32> %a2)
  %4 = bitcast i16 %mask to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %3, <16 x i32> %2
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssds_mask_commuted(<16 x i32>* %a0, <16 x i32> %a1, <16 x i32> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssds_mask_commuted:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x i32>, <16 x i32>* %a0
  %3 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %2, <16 x i32> %a2, <16 x i32> %a1)
  %4 = bitcast i16 %mask to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %3, <16 x i32> %2
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssds_maskz(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssds_maskz:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %2, <16 x i32> zeroinitializer
  ret <16 x i32> %5
}

define <16 x i32> @stack_fold_vpdpwssds_maskz_commuted(<16 x i32> %a0, <16 x i32> %a1, <16 x i32> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_vpdpwssds_maskz_commuted:
  ;CHECK:       vpdpwssds {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %a0, <16 x i32> %a2, <16 x i32> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x i32> %2, <16 x i32> zeroinitializer
  ret <16 x i32> %5
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>)
