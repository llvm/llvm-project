; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512fp16 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <32 x half> @stack_fold_addph_zmm(<32 x half> %a0, <32 x half> %a1) {
  ;CHECK-LABEL: stack_fold_addph_zmm
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <32 x half> %a0, %a1
  ret <32 x half> %2
}

define <32 x half> @stack_fold_addph_zmm_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_addph_zmm_k:
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <32 x half> %a0, %a1
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_addph_zmm_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_addph_zmm_k_commuted:
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <32 x half> %a1, %a0
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_addph_zmm_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_addph_zmm_kz
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <32 x half> %a1, %a0
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define half @stack_fold_addsh(half %a0, half %a1) {
  ;CHECK-LABEL: stack_fold_addsh
  ;CHECK:       vaddsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd half %a0, %a1
  ret half %2
}

define <8 x half> @stack_fold_addsh_int(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_addsh_int
  ;CHECK:       vaddsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <8 x half> %a0, i32 0
  %3 = extractelement <8 x half> %a1, i32 0
  %4 = fadd half %2, %3
  %5 = insertelement <8 x half> %a0, half %4, i32 0
  ret <8 x half> %5
}

define i32 @stack_fold_cmpph(<32 x half> %a0, <32 x half> %a1) {
  ;CHECK-LABEL: stack_fold_cmpph
  ;CHECK:       vcmpeqph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %a0, <32 x half> %a1, i32 0, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 4)
  %2 = bitcast <32 x i1> %res to i32
  ret i32 %2
}
declare <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half>, <32 x half>, i32, <32 x i1>, i32)

define <32 x half> @stack_fold_cmpph_mask(<32 x half> %a0, <32 x half> %a1, <32 x half>* %a2, i32 %mask, <32 x half> %b0, <32 x half> %b1) {
  ;CHECK-LABEL: stack_fold_cmpph_mask:
  ;CHECK:       vcmpeqph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load and fadd are here to keep the operations below the side effecting block and to avoid folding the wrong load
  %2 = load <32 x half>, <32 x half>* %a2
  %3 = fadd <32 x half> %a1, %2
  %4 = bitcast i32 %mask to <32 x i1>
  %5 = call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %3, <32 x half> %a0, i32 0, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 4)
  %6 = and <32 x i1> %4, %5
  %7 = select <32 x i1> %6, <32 x half> %b0, <32 x half> %b1
  ret <32 x half> %7
}

define <32 x half> @stack_fold_cmpph_mask_commuted(<32 x half> %a0, <32 x half> %a1, <32 x half>* %a2, i32 %mask, <32 x half> %b0, <32 x half> %b1) {
  ;CHECK-LABEL: stack_fold_cmpph_mask_commuted:
  ;CHECK:       vcmpeqph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load and fadd are here to keep the operations below the side effecting block and to avoid folding the wrong load
  %2 = load <32 x half>, <32 x half>* %a2
  %3 = fadd <32 x half> %a1, %2
  %4 = bitcast i32 %mask to <32 x i1>
  %5 = call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %a0, <32 x half> %3, i32 0, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 4)
  %6 = and <32 x i1> %4, %5
  %7 = select <32 x i1> %6, <32 x half> %b0, <32 x half> %b1
  ret <32 x half> %7
}

define half @stack_fold_divsh(half %a0, half %a1) {
  ;CHECK-LABEL: stack_fold_divsh
  ;CHECK:       vdivsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv half %a0, %a1
  ret half %2
}

define <8 x half> @stack_fold_divsh_int(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_divsh_int
  ;CHECK:       vdivsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <8 x half> %a0, i32 0
  %3 = extractelement <8 x half> %a1, i32 0
  %4 = fdiv half %2, %3
  %5 = insertelement <8 x half> %a0, half %4, i32 0
  ret <8 x half> %5
}

define <32 x half> @stack_fold_maxph_zmm(<32 x half> %a0, <32 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half>, <32 x half>, i32) nounwind readnone

define <32 x half> @stack_fold_maxph_zmm_commuted(<32 x half> %a0, <32 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commuted:
  ;CHECK-NOT:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_maxph_zmm_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_k:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_maxph_zmm_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_k_commuted:
  ;CHECK-NOT:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_maxph_zmm_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_kz:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_maxph_zmm_kz_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask) #0 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_kz_commuted:
  ;CHECK-NOT:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_maxph_zmm_commutable(<32 x half> %a0, <32 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_maxph_zmm_commutable_commuted(<32 x half> %a0, <32 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable_commuted:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_maxph_zmm_commutable_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable_k:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_maxph_zmm_commutable_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable_k_commuted:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_maxph_zmm_commutable_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable_kz:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_maxph_zmm_commutable_kz_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask) #1 {
  ;CHECK-LABEL: stack_fold_maxph_zmm_commutable_kz_commuted:
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.max.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define half @stack_fold_maxsh(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsh:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_maxsh_commuted(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsh_commuted:
  ;CHECK-NOT:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define half @stack_fold_maxsh_commutable(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsh_commutable:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_maxsh_commutable_commuted(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsh_commutable_commuted:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define <8 x half> @stack_fold_maxsh_int(<8 x half> %a0, <8 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsh_int:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.max.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.max.sh.round(<8 x half>, <8 x half>, <8 x half>, i8, i32)

define <8 x half> @stack_fold_maxsh_mask(<8 x half> %a0, <8 x half> %a1, i8 %mask, <8 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_maxsh_mask:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.max.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_maxsh_maskz(<8 x half> %a0, <8 x half> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_maxsh_maskz:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.max.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %mask, i32 4)
  ret <8 x half> %2
}

define <32 x half> @stack_fold_minph_zmm(<32 x half> %a0, <32 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half>, <32 x half>, i32) nounwind readnone

define <32 x half> @stack_fold_minph_zmm_commuted(<32 x half> %a0, <32 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commuted:
  ;CHECK-NOT:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_minph_zmm_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm_k:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_minph_zmm_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm_k_commuted:
  ;CHECK-NOT:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_minph_zmm_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm_kz:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_minph_zmm_kz_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask) #0 {
  ;CHECK-LABEL: stack_fold_minph_zmm_kz_commuted:
  ;CHECK-NOT:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_minph_zmm_commutable(<32 x half> %a0, <32 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_minph_zmm_commutable_commuted(<32 x half> %a0, <32 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable_commuted:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_minph_zmm_commutable_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable_k:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_minph_zmm_commutable_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable_k_commuted:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_minph_zmm_commutable_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable_kz:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define <32 x half> @stack_fold_minph_zmm_commutable_kz_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask) #1 {
  ;CHECK-LABEL: stack_fold_minph_zmm_commutable_kz_commuted:
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.min.ph.512(<32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define half @stack_fold_minsh(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsh:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_minsh_commuted(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsh_commuted:
  ;CHECK-NOT:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define half @stack_fold_minsh_commutable(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsh_commutable:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_minsh_commutable_commuted(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsh_commutable_commuted:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define <8 x half> @stack_fold_minsh_int(<8 x half> %a0, <8 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsh_int:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.min.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.min.sh.round(<8 x half>, <8 x half>, <8 x half>, i8, i32)

define <8 x half> @stack_fold_minsh_mask(<8 x half> %a0, <8 x half> %a1, i8 %mask, <8 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_minsh_mask:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.min.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_minsh_maskz(<8 x half> %a0, <8 x half> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_minsh_maskz:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.min.sh.round(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %mask, i32 4)
  ret <8 x half> %2
}

define <32 x half> @stack_fold_mulph_zmm(<32 x half> %a0, <32 x half> %a1) {
  ;CHECK-LABEL: stack_fold_mulph_zmm
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <32 x half> %a0, %a1
  ret <32 x half> %2
}

define <32 x half> @stack_fold_mulph_zmm_k(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_mulph_zmm_k:
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <32 x half> %a0, %a1
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_mulph_zmm_k_commuted(<32 x half> %a0, <32 x half> %a1, i32 %mask, <32 x half>* %passthru) {
  ;CHECK-LABEL: stack_fold_mulph_zmm_k_commuted:
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <32 x half> %a1, %a0
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = load <32 x half>, <32 x half>* %passthru
  %5 = select <32 x i1> %3, <32 x half> %2, <32 x half> %4
  ret <32 x half> %5
}

define <32 x half> @stack_fold_mulph_zmm_kz(<32 x half> %a0, <32 x half> %a1, i32 %mask) {
  ;CHECK-LABEL: stack_fold_mulph_zmm_kz
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <32 x half> %a1, %a0
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %4
}

define half @stack_fold_mulsh(half %a0, half %a1) {
  ;CHECK-LABEL: stack_fold_mulsh
  ;CHECK-NOT:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul half %a0, %a1
  ret half %2
}

define <8 x half> @stack_fold_mulsh_int(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_mulsh_int
  ;CHECK-NOT:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <8 x half> %a0, i32 0
  %3 = extractelement <8 x half> %a1, i32 0
  %4 = fmul half %2, %3
  %5 = insertelement <8 x half> %a0, half %4, i32 0
  ret <8 x half> %5
}

define <32 x half> @stack_fold_subph_zmm(<32 x half> %a0, <32 x half> %a1) {
  ;CHECK-LABEL: stack_fold_subph_zmm
  ;CHECK:       vsubph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <32 x half> %a0, %a1
  ret <32 x half> %2
}

define half @stack_fold_subsh(half %a0, half %a1) {
  ;CHECK-LABEL: stack_fold_subsh
  ;CHECK:       vsubsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 2-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub half %a0, %a1
  ret half %2
}

define <8 x half> @stack_fold_subsh_int(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_subsh_int
  ;CHECK:       vsubsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <8 x half> %a0, i32 0
  %3 = extractelement <8 x half> %a1, i32 0
  %4 = fsub half %2, %3
  %5 = insertelement <8 x half> %a0, half %4, i32 0
  ret <8 x half> %5
}

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
