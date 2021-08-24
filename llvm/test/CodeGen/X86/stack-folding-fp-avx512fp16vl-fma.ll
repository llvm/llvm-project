; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512fp16,+avx512vl < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <8 x half> @stack_fold_fmadd123ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd123ph:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2)
  ret <8 x half> %2
}
declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)

define <8 x half> @stack_fold_fmadd213ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd213ph:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %a2)
  ret <8 x half> %2
}

define <8 x half> @stack_fold_fmadd231ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd231ph:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %a0)
  ret <8 x half> %2
}

define <8 x half> @stack_fold_fmadd321ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd321ph:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %a0)
  ret <8 x half> %2
}

define <8 x half> @stack_fold_fmadd132ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd132ph:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %a1)
  ret <8 x half> %2
}

define <8 x half> @stack_fold_fmadd312ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd312ph:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %2
}

define <8 x half> @stack_fold_fmadd123ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_mask:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd213ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_mask:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %a2)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd231ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_mask:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %a0)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd321ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_mask:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %a0)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd132ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_mask:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd312ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_mask:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmadd123ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_maskz:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmadd213ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_maskz:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmadd231ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_maskz:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmadd321ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_maskz:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmadd132ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_maskz:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmadd312ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_maskz:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub123ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub123ph:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub213ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub213ph:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub231ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub231ph:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub321ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub321ph:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub132ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub132ph:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub312ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub312ph:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fmsub123ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_mask:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub213ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_mask:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub231ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_mask:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub321ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_mask:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub132ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_mask:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub312ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_mask:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fmsub123ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_maskz:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a1, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub213ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_maskz:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a0, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub231ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_maskz:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a1, <8 x half> %a2, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub321ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_maskz:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a1, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub132ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_maskz:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a0, <8 x half> %a2, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fmsub312ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_maskz:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %a2, <8 x half> %a0, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd123ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a1, <8 x half> %a2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd213ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a0, <8 x half> %a2)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd231ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a2, <8 x half> %a0)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd321ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a1, <8 x half> %a0)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd132ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a2, <8 x half> %a1)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd312ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a0, <8 x half> %a1)
  ret <8 x half> %3
}

define <8 x half> @stack_fold_fnmadd123ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_mask:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a1, <8 x half> %a2)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd213ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_mask:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a0, <8 x half> %a2)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd231ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_mask:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a2, <8 x half> %a0)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd321ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_mask:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a1, <8 x half> %a0)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd132ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_mask:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a2, <8 x half> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd312ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_mask:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a0, <8 x half> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmadd123ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_maskz:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a1, <8 x half> %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd213ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_maskz:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a0, <8 x half> %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd231ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_maskz:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a2, <8 x half> %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd321ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_maskz:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a1, <8 x half> %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd132ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_maskz:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a2, <8 x half> %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmadd312ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_maskz:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg, <8 x half> %a0, <8 x half> %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub123ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = fneg <8 x half> %a2
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a1, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub213ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = fneg <8 x half> %a2
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a0, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub231ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a1
  %3 = fneg <8 x half> %a0
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a2, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub321ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = fneg <8 x half> %a0
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a1, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub132ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a0
  %3 = fneg <8 x half> %a1
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a2, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub312ph(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <8 x half> %a2
  %3 = fneg <8 x half> %a1
  %4 = call <8 x half> @llvm.fma.v8f16(<8 x half> %2, <8 x half> %a0, <8 x half> %3)
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub123ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_mask:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %neg1 = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a1, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub213ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_mask:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a2
  %neg1 = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a0, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub231ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_mask:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %neg1 = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a2, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub321ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_mask:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a0
  %neg1 = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a1, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub132ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_mask:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %neg1 = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a2, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub312ph_mask(<8 x half>* %p, <8 x half> %a1, <8 x half> %a2, i8 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_mask:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <8 x half>, <8 x half>* %p
  %neg = fneg <8 x half> %a1
  %neg1 = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a0, <8 x half> %neg)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x half> %2, <8 x half> %a0
  ret <8 x half> %4
}

define <8 x half> @stack_fold_fnmsub123ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_maskz:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %neg1 = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a1, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub213ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_maskz:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a2
  %neg1 = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a0, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub231ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_maskz:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %neg1 = fneg <8 x half> %a1
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a2, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub321ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_maskz:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a0
  %neg1 = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a1, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub132ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_maskz:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %neg1 = fneg <8 x half> %a0
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a2, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <8 x half> @stack_fold_fnmsub312ph_maskz(<8 x half> %a0, <8 x half> %a1, <8 x half> %a2, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_maskz:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <8 x half> %a1
  %neg1 = fneg <8 x half> %a2
  %2 = call <8 x half> @llvm.fma.v8f16(<8 x half> %neg1, <8 x half> %a0, <8 x half> %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x half> %2, <8 x half> zeroinitializer
  ret <8 x half> %5
}

define <16 x half> @stack_fold_fmadd123ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2)
  ret <16 x half> %2
}
declare <16 x half> @llvm.fma.v16f16(<16 x half>, <16 x half>, <16 x half>)

define <16 x half> @stack_fold_fmadd213ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %a2)
  ret <16 x half> %2
}

define <16 x half> @stack_fold_fmadd231ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %a0)
  ret <16 x half> %2
}

define <16 x half> @stack_fold_fmadd321ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %a0)
  ret <16 x half> %2
}

define <16 x half> @stack_fold_fmadd132ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %a1)
  ret <16 x half> %2
}

define <16 x half> @stack_fold_fmadd312ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %2
}

define <16 x half> @stack_fold_fmadd123ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_mask_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd213ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_mask_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %a2)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd231ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_mask_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %a0)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd321ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_mask_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %a0)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd132ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_mask_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd312ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_mask_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmadd123ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_maskz_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmadd213ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_maskz_ymm:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmadd231ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_maskz_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %a0)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmadd321ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_maskz_ymm:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %a0)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmadd132ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_maskz_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmadd312ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_maskz_ymm:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub123ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub213ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub231ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub321ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub132ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub312ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fmsub123ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_mask_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub213ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_mask_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub231ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_mask_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub321ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_mask_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub132ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_mask_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub312ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_mask_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fmsub123ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_maskz_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a1, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub213ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_maskz_ymm:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a0, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub231ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_maskz_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a1, <16 x half> %a2, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub321ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_maskz_ymm:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a1, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub132ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_maskz_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a0, <16 x half> %a2, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fmsub312ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_maskz_ymm:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %a2, <16 x half> %a0, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd123ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a1, <16 x half> %a2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd213ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a0, <16 x half> %a2)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd231ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a2, <16 x half> %a0)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd321ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a1, <16 x half> %a0)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd132ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a2, <16 x half> %a1)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd312ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a0, <16 x half> %a1)
  ret <16 x half> %3
}

define <16 x half> @stack_fold_fnmadd123ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_mask_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a1, <16 x half> %a2)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd213ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_mask_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a0, <16 x half> %a2)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd231ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_mask_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a2, <16 x half> %a0)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd321ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_mask_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a1, <16 x half> %a0)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd132ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_mask_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a2, <16 x half> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd312ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_mask_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a0, <16 x half> %a1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmadd123ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_maskz_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a1, <16 x half> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd213ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_maskz_ymm:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a0, <16 x half> %a2)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd231ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_maskz_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a2, <16 x half> %a0)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd321ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_maskz_ymm:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a1, <16 x half> %a0)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd132ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_maskz_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a2, <16 x half> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmadd312ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_maskz_ymm:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg, <16 x half> %a0, <16 x half> %a1)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub123ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = fneg <16 x half> %a2
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a1, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub213ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = fneg <16 x half> %a2
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a0, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub231ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a1
  %3 = fneg <16 x half> %a0
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a2, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub321ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = fneg <16 x half> %a0
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a1, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub132ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a0
  %3 = fneg <16 x half> %a1
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a2, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub312ph_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <16 x half> %a2
  %3 = fneg <16 x half> %a1
  %4 = call <16 x half> @llvm.fma.v16f16(<16 x half> %2, <16 x half> %a0, <16 x half> %3)
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub123ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_mask_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %neg1 = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a1, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub213ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_mask_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a2
  %neg1 = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a0, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub231ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_mask_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %neg1 = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a2, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub321ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_mask_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a0
  %neg1 = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a1, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub132ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_mask_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %neg1 = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a2, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub312ph_mask_ymm(<16 x half>* %p, <16 x half> %a1, <16 x half> %a2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_mask_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <16 x half>, <16 x half>* %p
  %neg = fneg <16 x half> %a1
  %neg1 = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a0, <16 x half> %neg)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x half> %2, <16 x half> %a0
  ret <16 x half> %4
}

define <16 x half> @stack_fold_fnmsub123ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_maskz_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %neg1 = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a1, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub213ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_maskz_ymm:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a2
  %neg1 = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a0, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub231ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_maskz_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %neg1 = fneg <16 x half> %a1
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a2, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub321ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_maskz_ymm:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a0
  %neg1 = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a1, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub132ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_maskz_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %neg1 = fneg <16 x half> %a0
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a2, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}

define <16 x half> @stack_fold_fnmsub312ph_maskz_ymm(<16 x half> %a0, <16 x half> %a1, <16 x half> %a2, i16* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_maskz_ymm:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <16 x half> %a1
  %neg1 = fneg <16 x half> %a2
  %2 = call <16 x half> @llvm.fma.v16f16(<16 x half> %neg1, <16 x half> %a0, <16 x half> %neg)
  %3 = load i16, i16* %mask
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x half> %2, <16 x half> zeroinitializer
  ret <16 x half> %5
}
