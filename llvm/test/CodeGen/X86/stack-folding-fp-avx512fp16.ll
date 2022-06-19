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
  ;CHECK:       vaddsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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
  ;CHECK:       vdivsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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

define i32 @stack_fold_fpclassph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_fpclassph:
  ;CHECK:       vfpclassphz $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i1> @llvm.x86.avx512fp16.fpclass.ph.512(<32 x half> %a0, i32 4)
  %3 = bitcast <32 x i1> %2 to i32
  ret i32 %3
}
declare <32 x i1> @llvm.x86.avx512fp16.fpclass.ph.512(<32 x half>, i32)

define i32 @stack_fold_fpclassph_mask(<32 x half> %a0, <32 x i1>* %p) {
  ;CHECK-LABEL: stack_fold_fpclassph_mask:
  ;CHECK:       vfpclassphz $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i1> @llvm.x86.avx512fp16.fpclass.ph.512(<32 x half> %a0, i32 4)
  %mask = load <32 x i1>, <32 x i1>* %p
  %3 = and <32 x i1> %2, %mask
  %4 = bitcast <32 x i1> %3 to i32
  ret i32 %4
}

define i8 @stack_fold_fpclasssh(<8 x half> %a0) {
  ;CHECK-LABEl: stack_fold_fpclasssh:
  ;CHECK:       vfpclasssh $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call i8 @llvm.x86.avx512fp16.mask.fpclass.sh(<8 x half> %a0, i32 4, i8 -1)
  ret i8 %2
}
declare i8 @llvm.x86.avx512fp16.mask.fpclass.sh(<8 x half>, i32, i8)

define i8 @stack_fold_fpclasssh_mask(<8 x half> %a0, i8* %p) {
  ;CHECK-LABEL: stack_fold_fpclasssh_mask:
  ;CHECK:       vfpclasssh $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %mask = load i8, i8* %p
  %2 = call i8 @llvm.x86.avx512fp16.mask.fpclass.sh(<8 x half> %a0, i32 4, i8 %mask)
  ret i8 %2
}

define <32 x half> @stack_fold_getexpph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getexpph:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.getexp.ph.512(<32 x half> %a0, <32 x half> undef, i32 -1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.getexp.ph.512(<32 x half>, <32 x half>, i32, i32)

define <32 x half> @stack_fold_getexpph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_mask:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.getexp.ph.512(<32 x half> %a0, <32 x half> %2, i32 %mask, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_getexpph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_maskz:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.getexp.ph.512(<32 x half> %a0, <32 x half> zeroinitializer, i32 %2, i32 4)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_getexpsh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_getexpsh:
  ;CHECK:       vgetexpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.getexp.sh(<8 x half>, <8 x half>, <8 x half>, i8, i32)

define <8 x half> @stack_fold_getexpsh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_getexpsh_mask:
  ;CHECK:       vgetexpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_getexpsh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_getexpsh_maskz:
  ;CHECK:       vgetexpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2, i32 4)
  ret <8 x half> %3
}

define <32 x half> @stack_fold_getmantph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getmantph:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.getmant.ph.512(<32 x half> %a0, i32 8, <32 x half> undef, i32 -1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.getmant.ph.512(<32 x half>, i32, <32 x half>, i32, i32)

define <32 x half> @stack_fold_getmantph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_mask:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.getmant.ph.512(<32 x half> %a0, i32 8, <32 x half> %2, i32 %mask, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_getmantph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_maskz:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.getmant.ph.512(<32 x half> %a0, i32 8, <32 x half> zeroinitializer, i32 %2, i32 4)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_getmantsh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_getmantsh:
  ;CHECK:       vgetmantsh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.sh(<8 x half> %a0, <8 x half> %a1, i32 8, <8 x half> undef, i8 -1, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.getmant.sh(<8 x half>, <8 x half>, i32, <8 x half>, i8, i32)

define <8 x half> @stack_fold_getmantsh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_getmantsh_mask:
  ;CHECK:       vgetmantsh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.sh(<8 x half> %a0, <8 x half> %a1, i32 8, <8 x half> %2, i8 %mask, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_getmantsh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_getmantsh_maskz:
  ;CHECK:       vgetmantsh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.sh(<8 x half> %a0, <8 x half> %a1, i32 8, <8 x half> zeroinitializer, i8 %2, i32 4)
  ret <8 x half> %3
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
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_maxsh_commuted(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsh_commuted:
  ;CHECK-NOT:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define half @stack_fold_maxsh_commutable(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsh_commutable:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp ogt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_maxsh_commutable_commuted(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsh_commutable_commuted:
  ;CHECK:       vmaxsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_minsh_commuted(half %a0, half %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsh_commuted:
  ;CHECK-NOT:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a1, %a0
  %3 = select i1 %2, half %a1, half %a0
  ret half %3
}

define half @stack_fold_minsh_commutable(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsh_commutable:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fcmp olt half %a0, %a1
  %3 = select i1 %2, half %a0, half %a1
  ret half %3
}

define half @stack_fold_minsh_commutable_commuted(half %a0, half %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsh_commutable_commuted:
  ;CHECK:       vminsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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
  ;CHECK-NOT:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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

define <32 x half> @stack_fold_rcpph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rcpph:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %a0, <32 x half> undef, i32 -1)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half>, <32 x half>, i32)

define <32 x half> @stack_fold_rcpph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_mask:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %a0, <32 x half> %2, i32 %mask)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_rcpph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_maskz:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %a0, <32 x half> zeroinitializer, i32 %2)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_rcpsh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_rcpsh:
  ;CHECK:       vrcpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half>, <8 x half>, <8 x half>, i8)

define <8 x half> @stack_fold_rcpsh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rcpsh_mask:
  ;CHECK:       vrcpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rcpsh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rcpsh_maskz:
  ;CHECK:       vrcpsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <32 x half> @stack_fold_reduceph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_reduceph:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.reduce.ph.512(<32 x half> %a0, i32 8, <32 x half> undef, i32 -1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.reduce.ph.512(<32 x half>, i32, <32 x half>, i32, i32)

define <32 x half> @stack_fold_reduceph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_mask:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.reduce.ph.512(<32 x half> %a0, i32 8, <32 x half> %2, i32 %mask, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_reduceph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_maskz:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.reduce.ph.512(<32 x half> %a0, i32 8, <32 x half> zeroinitializer, i32 %2, i32 4)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_reducesh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_reducesh:
  ;CHECK:       vreducesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 8, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.reduce.sh(<8 x half>, <8 x half>, <8 x half>, i8, i32, i32)

define <8 x half> @stack_fold_reducesh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_reducesh_mask:
  ;CHECK:       vreducesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 8, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_reducesh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_reducesh_maskz:
  ;CHECK:       vreducesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2, i32 8, i32 4)
  ret <8 x half> %3
}

define <32 x half> @stack_fold_rndscaleph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rndscaleph:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.512(<32 x half> %a0, i32 8, <32 x half> undef, i32 -1, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.512(<32 x half>, i32, <32 x half>, i32, i32)

define <32 x half> @stack_fold_rndscaleph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_mask:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.512(<32 x half> %a0, i32 8, <32 x half> %2, i32 %mask, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_rndscaleph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_maskz:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.512(<32 x half> %a0, i32 8, <32 x half> zeroinitializer, i32 %2, i32 4)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_rndscalesh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_rndscalesh:
  ;CHECK:       vrndscalesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 8, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rndscale.sh(<8 x half>, <8 x half>, <8 x half>, i8, i32, i32)

define <8 x half> @stack_fold_rndscalesh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rndscalesh_mask:
  ;CHECK:       vrndscalesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 8, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rndscalesh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rndscalesh_maskz:
  ;CHECK:       vrndscalesh $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2, i32 8, i32 4)
  ret <8 x half> %3
}

define <32 x half> @stack_fold_rsqrtph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtph:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %a0, <32 x half> undef, i32 -1)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half>, <32 x half>, i32)

define <32 x half> @stack_fold_rsqrtph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_mask:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %a0, <32 x half> %2, i32 %mask)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_rsqrtph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_maskz:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %a0, <32 x half> zeroinitializer, i32 %2)
  ret <32 x half> %3
}

define <8 x half> @stack_fold_rsqrtsh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_rsqrtsh:
  ;CHECK:       vrsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)

define <8 x half> @stack_fold_rsqrtsh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtsh_mask:
  ;CHECK:       vrsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rsqrtsh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtsh_maskz:
  ;CHECK:       vrsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <32 x half> @stack_fold_sqrtph(<32 x half> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtph:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.sqrt.v32f16(<32 x half> %a0)
  ret <32 x half> %2
}
declare <32 x half> @llvm.sqrt.v32f16(<32 x half>)

define <32 x half> @stack_fold_sqrtph_mask(<32 x half> %a0, <32 x half>* %passthru, i32 %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_mask:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <32 x half>, <32 x half>* %passthru
  %3 = call <32 x half> @llvm.sqrt.v32f16(<32 x half> %a0)
  %4 = bitcast i32 %mask to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %3, <32 x half> %2
  ret <32 x half> %5
}

define <32 x half> @stack_fold_sqrtph_maskz(<32 x half> %a0, i32* %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_maskz:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i32, i32* %mask
  %3 = call <32 x half> @llvm.sqrt.v32f16(<32 x half> %a0)
  %4 = bitcast i32 %2 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %3, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <8 x half> @stack_fold_sqrtsh(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_sqrtsh:
  ;CHECK:       vsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> undef, i8 -1, i32 4)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8, i32)

define <8 x half> @stack_fold_sqrtsh_mask(<8 x half> %a0, <8 x half> %a1, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_sqrtsh_mask:
  ;CHECK:       vsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> %2, i8 %mask, i32 4)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_sqrtsh_maskz(<8 x half> %a0, <8 x half> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_sqrtsh_maskz:
  ;CHECK:       vsqrtsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %a0, <8 x half> %a1, <8 x half> zeroinitializer, i8 %2, i32 4)
  ret <8 x half> %3
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
  ;CHECK:       vsubsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
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

define <16 x float> @stack_fold_fmulcph(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulcph:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> undef, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fmulcph_commute(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulcph_commute:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.512(<16 x float> %a1, <16 x float> %a0, <16 x float> undef, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @stack_fold_fmulcph_mask(<16 x float> %a0, <16 x float> %a1, <16 x float>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmulcph_mask:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x float>, <16 x float>* %passthru
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %2, i16 %mask, i32 4)
  ret <16 x float> %3
}

define <16 x float> @stack_fold_fmulcph_maskz(<16 x float> %a0, <16 x float> %a1, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmulcph_maskz:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 %2, i32 4)
  ret <16 x float> %3
}

define <16 x float> @stack_fold_fcmulcph(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulcph:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> undef, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fcmulcph_commute(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulcph_commute:
  ;CHECK:       vmovups {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Reload
  ;CHECK:       vfcmulcph {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}}
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.512(<16 x float> %a1, <16 x float> %a0, <16 x float> undef, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @stack_fold_fcmulcph_mask(<16 x float> %a0, <16 x float> %a1, <16 x float>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fcmulcph_mask:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x float>, <16 x float>* %passthru
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %2, i16 %mask, i32 4)
  ret <16 x float> %3
}

define <16 x float> @stack_fold_fcmulcph_maskz(<16 x float> %a0, <16 x float> %a1, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fcmulcph_maskz:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 %2, i32 4)
  ret <16 x float> %3
}

define <16 x float> @stack_fold_fmaddcph(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddcph:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> %a0, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fmaddcph_commute(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddcph_commute:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.512(<16 x float> %a2, <16 x float> %a1, <16 x float> %a0, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @stack_fold_fmaddcph_mask(<16 x float>* %p, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddcph_mask:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x float>, <16 x float>* %p
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> %a0, i16 %mask, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fmaddcph_maskz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddcph_maskz:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> zeroinitializer, i16 %2, i32 4)
  ret <16 x float> %3
}
declare <16 x float> @llvm.x86.avx512fp16.maskz.vfmadd.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @stack_fold_fcmaddcph(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddcph:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> %a0, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fcmaddcph_commute(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddcph_commute:
  ;CHECK:       vmovups {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Reload
  ;CHECK:       vfcmaddcph {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}}
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float> %a2, <16 x float> %a1, <16 x float> %a0, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @stack_fold_fcmaddcph_mask(<16 x float>* %p, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddcph_mask:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x float>, <16 x float>* %p
  %2 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> %a0, i16 %mask, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_fcmaddcph_maskz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddcph_maskz:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float> %a1, <16 x float> %a2, <16 x float> zeroinitializer, i16 %2, i32 4)
  ret <16 x float> %3
}
declare <16 x float> @llvm.x86.avx512fp16.maskz.vfcmadd.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <4 x float> @stack_fold_fmulcsh(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulcsh:
  ;CHECK:       vfmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> undef, i8 -1, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fmulcsh_commute(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulcsh_commute:
  ;CHECK:       vfmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.csh(<4 x float> %a1, <4 x float> %a0, <4 x float> undef, i8 -1, i32 4)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfmul.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

define <4 x float> @stack_fold_fmulcsh_mask(<4 x float> %a0, <4 x float> %a1, <4 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmulcsh_mask:
  ;CHECK:       vfmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <4 x float>, <4 x float>* %passthru
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> %2, i8 %mask, i32 4)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fmulcsh_maskz(<4 x float> %a0, <4 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmulcsh_maskz:
  ;CHECK:       vfmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %2, i32 4)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fcmulcsh(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulcsh:
  ;CHECK:       vfcmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> undef, i8 -1, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fcmulcsh_commute(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulcsh_commute:
  ;CHECK:       vmovaps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Reload
  ;CHECK:       vfcmulcsh {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.csh(<4 x float> %a1, <4 x float> %a0, <4 x float> undef, i8 -1, i32 4)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

define <4 x float> @stack_fold_fcmulcsh_mask(<4 x float> %a0, <4 x float> %a1, <4 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmulcsh_mask:
  ;CHECK:       vfcmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <4 x float>, <4 x float>* %passthru
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> %2, i8 %mask, i32 4)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fcmulcsh_maskz(<4 x float> %a0, <4 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmulcsh_maskz:
  ;CHECK:       vfcmulcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.csh(<4 x float> %a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %2, i32 4)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fmaddcsh(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddcsh:
  ;CHECK:       vfmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 -1, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fmaddcsh_commute(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddcsh_commute:
  ;CHECK:       vfmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %a2, <4 x float> %a1, <4 x float> %a0, i8 -1, i32 4)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

define <4 x float> @stack_fold_fmaddcsh_mask(<4 x float>* %p, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddcsh_mask:
  ;CHECK:       vfmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <4 x float>, <4 x float>* %p
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 %mask, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fmaddcsh_maskz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddcsh_maskz:
  ;CHECK:       vfmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> zeroinitializer, i8 %2, i32 4)
  ret <4 x float> %3
}
declare <4 x float> @llvm.x86.avx512fp16.maskz.vfmadd.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

define <4 x float> @stack_fold_fcmaddcsh(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddcsh:
  ;CHECK:       vfcmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 -1, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fcmaddcsh_commute(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddcsh_commute:
  ;CHECK:       vmovaps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Reload
  ;CHECK:       vfcmaddcsh {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %a2, <4 x float> %a1, <4 x float> %a0, i8 -1, i32 4)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

define <4 x float> @stack_fold_fcmaddcsh_mask(<4 x float>* %p, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddcsh_mask:
  ;CHECK:       vfcmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <4 x float>, <4 x float>* %p
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 %mask, i32 4)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fcmaddcsh_maskz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddcsh_maskz:
  ;CHECK:       vfcmaddcsh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %a1, <4 x float> %a2, <4 x float> zeroinitializer, i8 %2, i32 4)
  ret <4 x float> %3
}
declare <4 x float> @llvm.x86.avx512fp16.maskz.vfcmadd.csh(<4 x float>, <4 x float>, <4 x float>, i8, i32)

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
