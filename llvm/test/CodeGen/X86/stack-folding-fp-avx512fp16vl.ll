; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512vl,+avx512fp16 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <8 x half> @stack_fold_addph(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_addph
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <8 x half> %a0, %a1
  ret <8 x half> %2
}

define <16 x half> @stack_fold_addph_ymm(<16 x half> %a0, <16 x half> %a1) {
  ;CHECK-LABEL: stack_fold_addph_ymm
  ;CHECK:       vaddph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <16 x half> %a0, %a1
  ret <16 x half> %2
}

define i8 @stack_fold_cmpph(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_cmpph
  ;CHECK:       vcmpeqph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %a0, <8 x half> %a1, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %2 = bitcast <8 x i1> %res to i8
  ret i8 %2
}
declare <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half>, <8 x half>, i32,  <8 x i1>)

define i16 @stack_fold_cmpph_ymm(<16 x half> %a0, <16 x half> %a1) {
  ;CHECK-LABEL: stack_fold_cmpph_ymm
  ;CHECK:       vcmpeqph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %a0, <16 x half> %a1, i32 0, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %2 = bitcast <16 x i1> %res to i16
  ret i16 %2
}
declare <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half>, <16 x half>, i32, <16 x i1>)

define <8 x half> @stack_fold_divph(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_divph
  ;CHECK:       vdivph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <8 x half> %a0, %a1
  ret <8 x half> %2
}

define <16 x half> @stack_fold_divph_ymm(<16 x half> %a0, <16 x half> %a1) {
  ;CHECK-LABEL: stack_fold_divph_ymm
  ;CHECK:       vdivph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <16 x half> %a0, %a1
  ret <16 x half> %2
}

define i8 @stack_fold_fpclassph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_fpclassph:
  ;CHECK:       fpclassphx $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half> %a0, i32 4)
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}
declare <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half>, i32)

define i8 @stack_fold_fpclassph_mask(<8 x half> %a0, <8 x i1>* %p) {
  ;CHECK-LABEL: stack_fold_fpclassph_mask:
  ;CHECK:       fpclassphx $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half> %a0, i32 4)
  %mask = load <8 x i1>, <8 x i1>* %p
  %3 = and <8 x i1> %2, %mask
  %4 = bitcast <8 x i1> %3 to i8
  ret i8 %4
}

define i16 @stack_fold_fpclassph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_fpclassph_ymm:
  ;CHECK:       fpclassphy $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i1> @llvm.x86.avx512fp16.fpclass.ph.256(<16 x half> %a0, i32 4)
  %3 = bitcast <16 x i1> %2 to i16
  ret i16 %3
}
declare <16 x i1> @llvm.x86.avx512fp16.fpclass.ph.256(<16 x half>, i32)

define i16 @stack_fold_fpclassph_mask_ymm(<16 x half> %a0, <16 x i1>* %p) {
  ;CHECK-LABEL: stack_fold_fpclassph_mask_ymm:
  ;CHECK:       fpclassphy $4, {{-?[0-9]*}}(%rsp), {{%k[0-7]}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i1> @llvm.x86.avx512fp16.fpclass.ph.256(<16 x half> %a0, i32 4)
  %mask = load <16 x i1>, <16 x i1>* %p
  %3 = and <16 x i1> %2, %mask
  %4 = bitcast <16 x i1> %3 to i16
  ret i16 %4
}

define <8 x half> @stack_fold_getexpph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getexpph:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.ph.128(<8 x half> %a0, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.getexp.ph.128(<8 x half>, <8 x half>, i8)

define <8 x half> @stack_fold_getexpph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_mask:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.ph.128(<8 x half> %a0, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_getexpph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_maskz:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getexp.ph.128(<8 x half> %a0, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_getexpph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getexpph_ymm:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.getexp.ph.256(<16 x half> %a0, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.getexp.ph.256(<16 x half>, <16 x half>, i16)

define <16 x half> @stack_fold_getexpph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_mask_ymm:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.getexp.ph.256(<16 x half> %a0, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_getexpph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_getexpph_maskz_ymm:
  ;CHECK:       vgetexpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.getexp.ph.256(<16 x half> %a0, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_getmantph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getmantph:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.ph.128(<8 x half> %a0, i32 8, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.getmant.ph.128(<8 x half>, i32, <8 x half>, i8)

define <8 x half> @stack_fold_getmantph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_mask:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.ph.128(<8 x half> %a0, i32 8, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_getmantph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_maskz:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.getmant.ph.128(<8 x half> %a0, i32 8, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_getmantph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_getmantph_ymm:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.getmant.ph.256(<16 x half> %a0, i32 8, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.getmant.ph.256(<16 x half>, i32, <16 x half>, i16)

define <16 x half> @stack_fold_getmantph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_mask_ymm:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.getmant.ph.256(<16 x half> %a0, i32 8, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_getmantph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_getmantph_maskz_ymm:
  ;CHECK:       vgetmantph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.getmant.ph.256(<16 x half> %a0, i32 8, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_maxph(<8 x half> %a0, <8 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxph
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.max.ph.128(<8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.max.ph.128(<8 x half>, <8 x half>) nounwind readnone

define <8 x half> @stack_fold_maxph_commutable(<8 x half> %a0, <8 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxph_commutable
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.max.ph.128(<8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %2
}

define <16 x half> @stack_fold_maxph_ymm(<16 x half> %a0, <16 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxph_ymm
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.max.ph.256(<16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.max.ph.256(<16 x half>, <16 x half>) nounwind readnone

define <16 x half> @stack_fold_maxph_ymm_commutable(<16 x half> %a0, <16 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxph_ymm_commutable
  ;CHECK:       vmaxph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.max.ph.256(<16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %2
}

define <8 x half> @stack_fold_minph(<8 x half> %a0, <8 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minph
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.min.ph.128(<8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.min.ph.128(<8 x half>, <8 x half>) nounwind readnone

define <8 x half> @stack_fold_minph_commutable(<8 x half> %a0, <8 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minph_commutable
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.min.ph.128(<8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %2
}

define <16 x half> @stack_fold_minph_ymm(<16 x half> %a0, <16 x half> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minph_ymm
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.min.ph.256(<16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.min.ph.256(<16 x half>, <16 x half>) nounwind readnone

define <16 x half> @stack_fold_minph_ymm_commutable(<16 x half> %a0, <16 x half> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minph_ymm_commutable
  ;CHECK:       vminph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.min.ph.256(<16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %2
}

define <8 x half> @stack_fold_mulph(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_mulph
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <8 x half> %a0, %a1
  ret <8 x half> %2
}

define <16 x half> @stack_fold_mulph_ymm(<16 x half> %a0, <16 x half> %a1) {
  ;CHECK-LABEL: stack_fold_mulph_ymm
  ;CHECK:       vmulph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <16 x half> %a0, %a1
  ret <16 x half> %2
}

define <8 x half> @stack_fold_rcpph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rcpph:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half> %a0, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half>, <8 x half>, i8)

define <8 x half> @stack_fold_rcpph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_mask:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half> %a0, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rcpph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_maskz:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half> %a0, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_rcpph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rcpph_ymm:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half> %a0, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half>, <16 x half>, i16)

define <16 x half> @stack_fold_rcpph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_mask_ymm:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half> %a0, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_rcpph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_rcpph_maskz_ymm:
  ;CHECK:       vrcpph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half> %a0, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_reduceph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_reduceph:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.ph.128(<8 x half> %a0, i32 8, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.reduce.ph.128(<8 x half>, i32, <8 x half>, i8)

define <8 x half> @stack_fold_reduceph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_mask:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.ph.128(<8 x half> %a0, i32 8, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_reduceph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_maskz:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.reduce.ph.128(<8 x half> %a0, i32 8, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_reduceph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_reduceph_ymm:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.reduce.ph.256(<16 x half> %a0, i32 8, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.reduce.ph.256(<16 x half>, i32, <16 x half>, i16)

define <16 x half> @stack_fold_reduceph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_mask_ymm:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.reduce.ph.256(<16 x half> %a0, i32 8, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_reduceph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_reduceph_maskz_ymm:
  ;CHECK:       vreduceph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.reduce.ph.256(<16 x half> %a0, i32 8, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_rndscaleph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rndscaleph:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.128(<8 x half> %a0, i32 8, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.128(<8 x half>, i32, <8 x half>, i8)

define <8 x half> @stack_fold_rndscaleph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_mask:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.128(<8 x half> %a0, i32 8, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rndscaleph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_maskz:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.128(<8 x half> %a0, i32 8, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_rndscaleph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rndscaleph_ymm:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.256(<16 x half> %a0, i32 8, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.256(<16 x half>, i32, <16 x half>, i16)

define <16 x half> @stack_fold_rndscaleph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_mask_ymm:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.256(<16 x half> %a0, i32 8, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_rndscaleph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_rndscaleph_maskz_ymm:
  ;CHECK:       vrndscaleph $8, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rndscale.ph.256(<16 x half> %a0, i32 8, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_rsqrtph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtph:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half> %a0, <8 x half> undef, i8 -1)
  ret <8 x half> %2
}
declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half>, <8 x half>, i8)

define <8 x half> @stack_fold_rsqrtph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_mask:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half> %a0, <8 x half> %2, i8 %mask)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_rsqrtph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_maskz:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half> %a0, <8 x half> zeroinitializer, i8 %2)
  ret <8 x half> %3
}

define <16 x half> @stack_fold_rsqrtph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtph_ymm:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half> %a0, <16 x half> undef, i16 -1)
  ret <16 x half> %2
}
declare <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half>, <16 x half>, i16)

define <16 x half> @stack_fold_rsqrtph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_mask_ymm:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half> %a0, <16 x half> %2, i16 %mask)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_rsqrtph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_rsqrtph_maskz_ymm:
  ;CHECK:       vrsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half> %a0, <16 x half> zeroinitializer, i16 %2)
  ret <16 x half> %3
}

define <8 x half> @stack_fold_sqrtph(<8 x half> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtph:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a0)
  ret <8 x half> %2
}
declare <8 x half> @llvm.sqrt.v8f16(<8 x half>)

define <8 x half> @stack_fold_sqrtph_mask(<8 x half> %a0, <8 x half>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_mask:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x half>, <8 x half>* %passthru
  %3 = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a0)
  %4 = bitcast i8 %mask to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %3, <8 x half> %2
  ret <8 x half> %5
}

define <8 x half> @stack_fold_sqrtph_maskz(<8 x half> %a0, i8* %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_maskz:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a0)
  %4 = bitcast i8 %2 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %3, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <16 x half> @stack_fold_sqrtph_ymm(<16 x half> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtph_ymm:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.sqrt.v16f16(<16 x half> %a0)
  ret <16 x half> %2
}
declare <16 x half> @llvm.sqrt.v16f16(<16 x half>)

define <16 x half> @stack_fold_sqrtph_mask_ymm(<16 x half> %a0, <16 x half>* %passthru, i16 %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_mask_ymm:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <16 x half>, <16 x half>* %passthru
  %3 = call <16 x half> @llvm.sqrt.v16f16(<16 x half> %a0)
  %4 = bitcast i16 %mask to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %3, <16 x half> %2
  ret <16 x half> %5
}

define <16 x half> @stack_fold_sqrtph_maskz_ymm(<16 x half> %a0, i16* %mask) {
  ;CHECK-LABEL: stack_fold_sqrtph_maskz_ymm:
  ;CHECK:       vsqrtph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i16, i16* %mask
  %3 = call <16 x half> @llvm.sqrt.v16f16(<16 x half> %a0)
  %4 = bitcast i16 %2 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %3, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <8 x half> @stack_fold_subph(<8 x half> %a0, <8 x half> %a1) {
  ;CHECK-LABEL: stack_fold_subph
  ;CHECK:       vsubph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <8 x half> %a0, %a1
  ret <8 x half> %2
}

define <16 x half> @stack_fold_subph_ymm(<16 x half> %a0, <16 x half> %a1) {
  ;CHECK-LABEL: stack_fold_subph_ymm
  ;CHECK:       vsubph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <16 x half> %a0, %a1
  ret <16 x half> %2
}

define <4 x float> @stack_fold_fmulc(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulc:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> undef, i8 -1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float> @stack_fold_fmulc_mask(<4 x float> %a0, <4 x float> %a1, <4 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmulc_mask:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <4 x float>, <4 x float>* %passthru
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %2, i8 %mask)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fmulc_maskz(<4 x float> %a0, <4 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmulc_maskz:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %2)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fcmulc(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulc:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> undef, i8 -1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float> @stack_fold_fcmulc_mask(<4 x float> %a0, <4 x float> %a1, <4 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmulc_mask:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <4 x float>, <4 x float>* %passthru
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %2, i8 %mask)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fcmulc_maskz(<4 x float> %a0, <4 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmulc_maskz:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.128(<4 x float> %a0, <4 x float> %a1, <4 x float> zeroinitializer, i8 %2)
  ret <4 x float> %3
}

define <4 x float> @stack_fold_fmaddc(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddc:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 -1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float> @stack_fold_fmaddc_mask(<4 x float>* %p, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddc_mask:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <4 x float>, <4 x float>* %p
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 %mask)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fmaddc_maskz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddc_maskz:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> zeroinitializer, i8 %2)
  ret <4 x float> %3
}
declare <4 x float> @llvm.x86.avx512fp16.maskz.vfmadd.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float> @stack_fold_fcmaddc(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddc:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 -1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <4 x float> @stack_fold_fcmaddc_mask(<4 x float>* %p, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddc_mask:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <4 x float>, <4 x float>* %p
  %2 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> %a0, i8 %mask)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_fcmaddc_maskz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddc_maskz:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.128(<4 x float> %a1, <4 x float> %a2, <4 x float> zeroinitializer, i8 %2)
  ret <4 x float> %3
}
declare <4 x float> @llvm.x86.avx512fp16.maskz.vfcmadd.cph.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @stack_fold_fmulc_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fmulc_ymm:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> undef, i8 -1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float> @stack_fold_fmulc_mask_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmulc_mask_ymm:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x float>, <8 x float>* %passthru
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %2, i8 %mask)
  ret <8 x float> %3
}

define <8 x float> @stack_fold_fmulc_maskz_ymm(<8 x float> %a0, <8 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmulc_maskz_ymm:
  ;CHECK:       vfmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> zeroinitializer, i8 %2)
  ret <8 x float> %3
}

define <8 x float> @stack_fold_fcmulc_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_fcmulc_ymm:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> undef, i8 -1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float> @stack_fold_fcmulc_mask_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float>* %passthru, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmulc_mask_ymm:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x float>, <8 x float>* %passthru
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %2, i8 %mask)
  ret <8 x float> %3
}

define <8 x float> @stack_fold_fcmulc_maskz_ymm(<8 x float> %a0, <8 x float> %a1, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmulc_maskz_ymm:
  ;CHECK:       vfcmulcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmul.cph.256(<8 x float> %a0, <8 x float> %a1, <8 x float> zeroinitializer, i8 %2)
  ret <8 x float> %3
}

define <8 x float> @stack_fold_fmaddc_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddc_ymm:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> %a0, i8 -1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float> @stack_fold_fmaddc_mask_ymm(<8 x float>* %p, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddc_mask_ymm:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x float>, <8 x float>* %p
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> %a0, i8 %mask)
  ret <8 x float> %2
}

define <8 x float> @stack_fold_fmaddc_maskz_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddc_maskz_ymm:
  ;CHECK:       vfmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> zeroinitializer, i8 %2)
  ret <8 x float> %3
}
declare <8 x float> @llvm.x86.avx512fp16.maskz.vfmadd.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float> @stack_fold_fcmaddc_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ;CHECK-LABEL: stack_fold_fcmaddc_ymm:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> %a0, i8 -1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <8 x float> @stack_fold_fcmaddc_mask_ymm(<8 x float>* %p, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddc_mask_ymm:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x float>, <8 x float>* %p
  %2 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> %a0, i8 %mask)
  ret <8 x float> %2
}

define <8 x float> @stack_fold_fcmaddc_maskz_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fcmaddc_maskz_ymm:
  ;CHECK:       vfcmaddcph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load i8, i8* %mask
  %3 = call <8 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.256(<8 x float> %a1, <8 x float> %a2, <8 x float> zeroinitializer, i8 %2)
  ret <8 x float> %3
}
declare <8 x float> @llvm.x86.avx512fp16.maskz.vfcmadd.cph.256(<8 x float>, <8 x float>, <8 x float>, i8)

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
