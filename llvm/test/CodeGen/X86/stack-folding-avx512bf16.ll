; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512bf16,+avx512vl < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <32 x i16> @stack_fold_cvtne2ps2bf16(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cvtne2ps2bf16:
  ;CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  ret <32 x i16> %2
}
declare <32 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float>, <16 x float>)

define <32 x i16> @stack_fold_cvtne2ps2bf16_mask(<16 x float> %a0, <16 x float> %a1, <32 x i16>* %passthru, i32 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  %3 = bitcast i32 %U to <32 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <32 x i16>, <32 x i16>* %passthru
  %5 = select <32 x i1> %3, <32 x i16> %2, <32 x i16> %4
  ret <32 x i16> %5
}

define <32 x i16> @stack_fold_cvtne2ps2bf16_maskz(<16 x float> %a0, <16 x float> %a1, i32 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  %3 = bitcast i32 %U to <32 x i1>
  %4 = select <32 x i1> %3, <32 x i16> %2, <32 x i16> zeroinitializer
  ret <32 x i16> %4
}

define <16 x i16> @stack_fold_cvtneps2bf16(<16 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16:
; CHECK:       vcvtneps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float>)

define <16 x i16> @stack_fold_cvtneps2bf16_mask(<16 x float> %a0, <16 x i16>* %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask:
; CHECK:       vcvtneps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  %3 = bitcast i16 %U to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x i16>, <16 x i16>* %passthru
  %5 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> %4
  ret <16 x i16> %5
}

define <16 x i16> @stack_fold_cvtneps2bf16_maskz(<16 x float> %a0, i16 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz:
; CHECK:       vcvtneps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  %3 = bitcast i16 %U to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> zeroinitializer
  ret <16 x i16> %4
}

define <16 x float> @stack_fold_vdpbf16ps(<16 x float> %a0, <16 x i32> %a1, <16 x i32> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %a0, <16 x i32> %a1, <16 x i32> %a2)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float>, <16 x i32>, <16 x i32>)

define <16 x float> @stack_fold_vdpbf16ps_mask(<16 x float>* %a0, <16 x i32> %a1, <16 x i32> %a2, <16 x float>* %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <16 x float>, <16 x float>* %a0
  %3 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %2, <16 x i32> %a1, <16 x i32> %a2)
  %4 = bitcast i16 %U to <16 x i1>
  %5 = select <16 x i1> %4, <16 x float> %3, <16 x float> %2
  ret <16 x float> %5
}

define <16 x float> @stack_fold_vdpbf16ps_maskz(<16 x float> %a0, <16 x i32> %a1, <16 x i32> %a2, i16* %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %a0, <16 x i32> %a1, <16 x i32> %a2)
  %3 = load i16, i16* %U
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x float> %2, <16 x float> zeroinitializer
  ret <16 x float> %5
}



define <16 x i16> @stack_fold_cvtne2ps2bf16_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cvtne2ps2bf16_ymm:
  ;CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  ret <16 x i16> %2
}
declare <16 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float>, <8 x float>)

define <16 x i16> @stack_fold_cvtne2ps2bf16_mask_ymm(<8 x float> %a0, <8 x float> %a1, <16 x i16>* %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask_ymm:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  %3 = bitcast i16 %U to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x i16>, <16 x i16>* %passthru
  %5 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> %4
  ret <16 x i16> %5
}

define <16 x i16> @stack_fold_cvtne2ps2bf16_maskz_ymm(<8 x float> %a0, <8 x float> %a1, i16 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz_ymm:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  %3 = bitcast i16 %U to <16 x i1>
  %4 = select <16 x i1> %3, <16 x i16> %2, <16 x i16> zeroinitializer
  ret <16 x i16> %4
}

define <8 x i16> @stack_fold_cvtneps2bf16_ymm(<8 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_ymm:
; CHECK:       vcvtneps2bf16y {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float>)

define <8 x i16> @stack_fold_cvtneps2bf16_mask_ymm(<8 x float> %a0, <8 x i16>* %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask_ymm:
; CHECK:       vcvtneps2bf16y {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  %3 = bitcast i8 %U to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x i16>, <8 x i16>* %passthru
  %5 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> %4
  ret <8 x i16> %5
}

define <8 x i16> @stack_fold_cvtneps2bf16_maskz_ymm(<8 x float> %a0, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz_ymm:
; CHECK:       vcvtneps2bf16y {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x i16> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  %3 = bitcast i8 %U to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> zeroinitializer
  ret <8 x i16> %4
}

define <8 x float> @stack_fold_vdpbf16ps_ymm(<8 x float> %a0, <8 x i32> %a1, <8 x i32> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps_ymm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %a0, <8 x i32> %a1, <8 x i32> %a2)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float>, <8 x i32>, <8 x i32>)

define <8 x float> @stack_fold_vdpbf16ps_mask_ymm(<8 x float>* %a0, <8 x i32> %a1, <8 x i32> %a2, <8 x float>* %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask_ymm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <8 x float>, <8 x float>* %a0
  %3 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %2, <8 x i32> %a1, <8 x i32> %a2)
  %4 = bitcast i8 %U to <8 x i1>
  %5 = select <8 x i1> %4, <8 x float> %3, <8 x float> %2
  ret <8 x float> %5
}

define <8 x float> @stack_fold_vdpbf16ps_maskz_ymm(<8 x float> %a0, <8 x i32> %a1, <8 x i32> %a2, i8* %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz_ymm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %a0, <8 x i32> %a1, <8 x i32> %a2)
  %3 = load i8, i8* %U
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x float> %2, <8 x float> zeroinitializer
  ret <8 x float> %5
}




define <8 x i16> @stack_fold_cvtne2ps2bf16_xmm(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cvtne2ps2bf16_xmm:
  ;CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float>, <4 x float>)

define <8 x i16> @stack_fold_cvtne2ps2bf16_mask_xmm(<4 x float> %a0, <4 x float> %a1, <8 x i16>* %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask_xmm:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  %3 = bitcast i8 %U to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x i16>, <8 x i16>* %passthru
  %5 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> %4
  ret <8 x i16> %5
}

define <8 x i16> @stack_fold_cvtne2ps2bf16_maskz_xmm(<4 x float> %a0, <4 x float> %a1, i8 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz_xmm:
; CHECK:       vcvtne2ps2bf16 {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  %3 = bitcast i8 %U to <8 x i1>
  %4 = select <8 x i1> %3, <8 x i16> %2, <8 x i16> zeroinitializer
  ret <8 x i16> %4
}

define <8 x i16> @stack_fold_cvtneps2bf16_xmm(<4 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_xmm:
; CHECK:       vcvtneps2bf16x {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x i16> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x i16> undef, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float>, <8 x i16>, <4 x i1>)

define <8 x i16> @stack_fold_cvtneps2bf16_mask_xmm(<4 x float> %a0, <8 x i16>* %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask_xmm:
; CHECK:       vcvtneps2bf16x {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x i16>, <8 x i16>* %passthru
  %3 = bitcast i8 %U to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = tail call <8 x i16> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x i16> %2, <4 x i1> %4)
  ret <8 x i16> %5
}

define <8 x i16> @stack_fold_cvtneps2bf16_maskz_xmm(<4 x float> %a0, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz_xmm:
; CHECK:       vcvtneps2bf16x {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast i8 %U to <8 x i1>
  %3 = shufflevector <8 x i1> %2, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %4 = tail call <8 x i16> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x i16> zeroinitializer, <4 x i1> %3)
  ret <8 x i16> %4
}

define <4 x float> @stack_fold_vdpbf16ps_xmm(<4 x float> %a0, <4 x i32> %a1, <4 x i32> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps_xmm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %a0, <4 x i32> %a1, <4 x i32> %a2)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float>, <4 x i32>, <4 x i32>)

define <4 x float> @stack_fold_vdpbf16ps_mask_xmm(<4 x float>* %a0, <4 x i32> %a1, <4 x i32> %a2, <4 x float>* %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask_xmm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <4 x float>, <4 x float>* %a0
  %3 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %2, <4 x i32> %a1, <4 x i32> %a2)
  %4 = bitcast i8 %U to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x float> %3, <4 x float> %2
  ret <4 x float> %6
}

define <4 x float> @stack_fold_vdpbf16ps_maskz_xmm(<4 x float> %a0, <4 x i32> %a1, <4 x i32> %a2, i8* %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz_xmm:
; CHECK:       vdpbf16ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %a0, <4 x i32> %a1, <4 x i32> %a2)
  %3 = load i8, i8* %U
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x float> %2, <4 x float> zeroinitializer
  ret <4 x float> %6
}
