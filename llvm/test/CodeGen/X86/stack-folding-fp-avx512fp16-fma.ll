; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512fp16 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <32 x half> @stack_fold_fmadd123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd123ph:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2)
  ret <32 x half> %2
}
declare <32 x half> @llvm.fma.v32f16(<32 x half>, <32 x half>, <32 x half>)

define <32 x half> @stack_fold_fmadd213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd213ph:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmadd231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd231ph:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmadd321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd321ph:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmadd132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd132ph:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmadd312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmadd312ph:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmadd123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_mask:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_mask:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_mask:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_mask:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_mask:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_mask:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmadd123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123ph_maskz:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmadd213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213ph_maskz:
  ;CHECK:       vfmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmadd231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231ph_maskz:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmadd321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321ph_maskz:
  ;CHECK:       vfmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmadd132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132ph_maskz:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmadd312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312ph_maskz:
  ;CHECK:       vfmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub123ph:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub213ph:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub231ph:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub321ph:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub132ph:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsub312ph:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsub123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_mask:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_mask:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_mask:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_mask:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_mask:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_mask:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsub123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123ph_maskz:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a1, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213ph_maskz:
  ;CHECK:       vfmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a0, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231ph_maskz:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a1, <32 x half> %a2, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321ph_maskz:
  ;CHECK:       vfmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a1, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132ph_maskz:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a0, <32 x half> %a2, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsub312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312ph_maskz:
  ;CHECK:       vfmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %a2, <32 x half> %a0, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a1, <32 x half> %a2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a0, <32 x half> %a2)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a2, <32 x half> %a0)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a1, <32 x half> %a0)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a2, <32 x half> %a1)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a0, <32 x half> %a1)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fnmadd123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_mask:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a1, <32 x half> %a2)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_mask:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a0, <32 x half> %a2)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_mask:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a2, <32 x half> %a0)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_mask:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a1, <32 x half> %a0)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_mask:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a2, <32 x half> %a1)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_mask:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a0, <32 x half> %a1)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmadd123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123ph_maskz:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a1, <32 x half> %a2)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213ph_maskz:
  ;CHECK:       vfnmadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a0, <32 x half> %a2)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231ph_maskz:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a2, <32 x half> %a0)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321ph_maskz:
  ;CHECK:       vfnmadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a1, <32 x half> %a0)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132ph_maskz:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a2, <32 x half> %a1)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmadd312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312ph_maskz:
  ;CHECK:       vfnmadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg, <32 x half> %a0, <32 x half> %a1)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = fneg <32 x half> %a2
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a1, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = fneg <32 x half> %a2
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a0, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = fneg <32 x half> %a0
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a2, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = fneg <32 x half> %a0
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a1, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = fneg <32 x half> %a1
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a2, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = fneg <32 x half> %a1
  %4 = call <32 x half> @llvm.fma.v32f16(<32 x half> %2, <32 x half> %a0, <32 x half> %3)
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_mask:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %neg1 = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a1, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_mask:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %neg1 = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a0, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_mask:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %neg1 = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a2, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_mask:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %neg1 = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a1, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_mask:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %neg1 = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a2, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_mask:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %neg1 = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a0, <32 x half> %neg)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fnmsub123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123ph_maskz:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %neg1 = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a1, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213ph_maskz:
  ;CHECK:       vfnmsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %neg1 = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a0, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231ph_maskz:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %neg1 = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a2, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321ph_maskz:
  ;CHECK:       vfnmsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %neg1 = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a1, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132ph_maskz:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %neg1 = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a2, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fnmsub312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312ph_maskz:
  ;CHECK:       vfnmsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %neg1 = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.fma.v32f16(<32 x half> %neg1, <32 x half> %a0, <32 x half> %neg)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define half @stack_fold_fmadd123sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd123sh:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %a2)
  ret half %2
}
declare half @llvm.fma.f16(half, half, half)

define half @stack_fold_fmadd213sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd213sh:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %a2)
  ret half %2
}

define half @stack_fold_fmadd231sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd231sh:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %a0)
  ret half %2
}

define half @stack_fold_fmadd321sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd321sh:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %a0)
  ret half %2
}

define half @stack_fold_fmadd132sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd132sh:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %a1)
  ret half %2
}

define half @stack_fold_fmadd312sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmadd312sh:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %a1)
  ret half %2
}

define half @stack_fold_fmsub123sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub123sh:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = call half @llvm.fma.f16(half %a0, half %a1, half %2)
  ret half %3
}

define half @stack_fold_fmsub213sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub213sh:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = call half @llvm.fma.f16(half %a1, half %a0, half %2)
  ret half %3
}

define half @stack_fold_fmsub231sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub231sh:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = call half @llvm.fma.f16(half %a1, half %a2, half %2)
  ret half %3
}

define half @stack_fold_fmsub321sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub321sh:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = call half @llvm.fma.f16(half %a2, half %a1, half %2)
  ret half %3
}

define half @stack_fold_fmsub132sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub132sh:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = call half @llvm.fma.f16(half %a0, half %a2, half %2)
  ret half %3
}

define half @stack_fold_fmsub312sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fmsub312sh:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = call half @llvm.fma.f16(half %a2, half %a0, half %2)
  ret half %3
}

define half @stack_fold_fnmadd123sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd123sh:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = call half @llvm.fma.f16(half %2, half %a1, half %a2)
  ret half %3
}

define half @stack_fold_fnmadd213sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd213sh:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = call half @llvm.fma.f16(half %2, half %a0, half %a2)
  ret half %3
}

define half @stack_fold_fnmadd231sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd231sh:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = call half @llvm.fma.f16(half %2, half %a2, half %a0)
  ret half %3
}

define half @stack_fold_fnmadd321sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd321sh:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = call half @llvm.fma.f16(half %2, half %a1, half %a0)
  ret half %3
}

define half @stack_fold_fnmadd132sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd132sh:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = call half @llvm.fma.f16(half %2, half %a2, half %a1)
  ret half %3
}

define half @stack_fold_fnmadd312sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmadd312sh:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = call half @llvm.fma.f16(half %2, half %a0, half %a1)
  ret half %3
}

define half @stack_fold_fnmsub123sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub123sh:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = fneg half %a2
  %4 = call half @llvm.fma.f16(half %2, half %a1, half %3)
  ret half %4
}

define half @stack_fold_fnmsub213sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub213sh:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = fneg half %a2
  %4 = call half @llvm.fma.f16(half %2, half %a0, half %3)
  ret half %4
}

define half @stack_fold_fnmsub231sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub231sh:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a1
  %3 = fneg half %a0
  %4 = call half @llvm.fma.f16(half %2, half %a2, half %3)
  ret half %4
}

define half @stack_fold_fnmsub321sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub321sh:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = fneg half %a0
  %4 = call half @llvm.fma.f16(half %2, half %a1, half %3)
  ret half %4
}

define half @stack_fold_fnmsub132sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub132sh:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a0
  %3 = fneg half %a1
  %4 = call half @llvm.fma.f16(half %2, half %a2, half %3)
  ret half %4
}

define half @stack_fold_fnmsub312sh(half %a0, half %a1, half %a2) {
  ;CHECK-LABEL: stack_fold_fnmsub312sh:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg half %a2
  %3 = fneg half %a1
  %4 = call half @llvm.fma.f16(half %2, half %a0, half %3)
  ret half %4
}

define <8 x half> @stack_fold_fmadd123sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd123sh_int:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %a2)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd213sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd213sh_int:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %a2)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd231sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd231sh_int:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %a0)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd321sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd321sh_int:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %a0)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd132sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd132sh_int:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %a1)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd312sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmadd312sh_int:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %a1)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub123sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub123sh_int:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub213sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub213sh_int:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub231sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub231sh_int:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub321sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub321sh_int:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub132sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub132sh_int:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub312sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fmsub312sh_int:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd123sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd123sh_int:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a2)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd213sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd213sh_int:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a2)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd231sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd231sh_int:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a0)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd321sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd321sh_int:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a0)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd132sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd132sh_int:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a1)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd312sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmadd312sh_int:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a1)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub123sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub123sh_int:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub213sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub213sh_int:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub231sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub231sh_int:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub321sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub321sh_int:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub132sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub132sh_int:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub312sh_int(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v) {
  ;CHECK-LABEL: stack_fold_fnmsub312sh_int:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %res = insertelement <8 x half> %a0v, half %2, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd123sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123sh_intk:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd213sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213sh_intk:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd231sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231sh_intk:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd321sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321sh_intk:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd132sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132sh_intk:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd312sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312sh_intk:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub123sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123sh_intk:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub213sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213sh_intk:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub231sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231sh_intk:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub321sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321sh_intk:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub132sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132sh_intk:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub312sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312sh_intk:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd123sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123sh_intk:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd213sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213sh_intk:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd231sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231sh_intk:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd321sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321sh_intk:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd132sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132sh_intk:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd312sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312sh_intk:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub123sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123sh_intk:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub213sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213sh_intk:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub231sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231sh_intk:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub321sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321sh_intk:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub132sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132sh_intk:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub312sh_intk(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312sh_intk:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half %a0
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd123sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd123sh_intkz:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd213sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd213sh_intkz:
  ;CHECK:       vfmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd231sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd231sh_intkz:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd321sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd321sh_intkz:
  ;CHECK:       vfmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd132sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd132sh_intkz:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmadd312sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmadd312sh_intkz:
  ;CHECK:       vfmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub123sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub123sh_intkz:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a0, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub213sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub213sh_intkz:
  ;CHECK:       vfmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %2 = call half @llvm.fma.f16(half %a1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub231sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub231sh_intkz:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub321sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub321sh_intkz:
  ;CHECK:       vfmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %2 = call half @llvm.fma.f16(half %a2, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub132sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub132sh_intkz:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a0, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fmsub312sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fmsub312sh_intkz:
  ;CHECK:       vfmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %2 = call half @llvm.fma.f16(half %a2, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd123sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd123sh_intkz:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd213sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd213sh_intkz:
  ;CHECK:       vfnmadd213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a2)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd231sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd231sh_intkz:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd321sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd321sh_intkz:
  ;CHECK:       vfnmadd231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %a0)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd132sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd132sh_intkz:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmadd312sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmadd312sh_intkz:
  ;CHECK:       vfnmadd132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %a1)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub123sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub123sh_intkz:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub213sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub213sh_intkz:
  ;CHECK:       vfnmsub213sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a2
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub231sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub231sh_intkz:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a1
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub321sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub321sh_intkz:
  ;CHECK:       vfnmsub231sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a0
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a1, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub132sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub132sh_intkz:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a0
  %2 = call half @llvm.fma.f16(half %neg1, half %a2, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <8 x half> @stack_fold_fnmsub312sh_intkz(<8 x half> %a0v, <8 x half> %a1v, <8 x half> %a2v, i8* %mask) {
  ;CHECK-LABEL: stack_fold_fnmsub312sh_intkz:
  ;CHECK:       vfnmsub132sh {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = extractelement <8 x half> %a0v, i64 0
  %a1 = extractelement <8 x half> %a1v, i64 0
  %a2 = extractelement <8 x half> %a2v, i64 0
  %neg = fneg half %a1
  %neg1 = fneg half %a2
  %2 = call half @llvm.fma.f16(half %neg1, half %a0, half %neg)
  %3 = load i8, i8* %mask
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = extractelement <8 x i1> %4, i64 0
  %6 = select i1 %5, half %2, half zeroinitializer
  %res = insertelement <8 x half> %a0v, half %6, i64 0
  ret <8 x half> %res
}

define <32 x half> @stack_fold_fmaddsub123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub123ph:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32 4)
  ret <32 x half> %2
}
declare <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half>, <32 x half>, <32 x half>, i32)

define <32 x half> @stack_fold_fmaddsub213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub213ph:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmaddsub231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub231ph:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmaddsub321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub321ph:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmaddsub132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub132ph:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmaddsub312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmaddsub312ph:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1, i32 4)
  ret <32 x half> %2
}

define <32 x half> @stack_fold_fmaddsub123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub123ph_mask:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub213ph_mask:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub231ph_mask:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub321ph_mask:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub132ph_mask:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub312ph_mask:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmaddsub123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub123ph_maskz:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmaddsub213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub213ph_maskz:
  ;CHECK:       vfmaddsub213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %a2, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmaddsub231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub231ph_maskz:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %a0, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmaddsub321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub321ph_maskz:
  ;CHECK:       vfmaddsub231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %a0, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmaddsub132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub132ph_maskz:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %a1, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmaddsub312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmaddsub312ph_maskz:
  ;CHECK:       vfmaddsub132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %a1, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd123ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd123ph:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd213ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd213ph:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a2
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd231ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd231ph:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd321ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd321ph:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a0
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd132ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd132ph:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd312ph(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2) {
  ;CHECK-LABEL: stack_fold_fmsubadd312ph:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fneg <32 x half> %a1
  %3 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %2, i32 4)
  ret <32 x half> %3
}

define <32 x half> @stack_fold_fmsubadd123ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd123ph_mask:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd213ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd213ph_mask:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd231ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd231ph_mask:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd321ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd321ph_mask:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd132ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd132ph_mask:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd312ph_mask(<32 x half>* %p, <32 x half> %a1, <32 x half> %a2, i32 %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd312ph_mask:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %a0 = load <32 x half>, <32 x half>* %p
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %neg, i32 4)
  %3 = bitcast i32 %mask to <32 x i1>
  %4 = select <32 x i1> %3, <32 x half> %2, <32 x half> %a0
  ret <32 x half> %4
}

define <32 x half> @stack_fold_fmsubadd123ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd123ph_maskz:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a1, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd213ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd213ph_maskz:
  ;CHECK:       vfmsubadd213ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a2
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a0, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd231ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd231ph_maskz:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a1, <32 x half> %a2, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd321ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd321ph_maskz:
  ;CHECK:       vfmsubadd231ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a0
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a1, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd132ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd132ph_maskz:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a0, <32 x half> %a2, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}

define <32 x half> @stack_fold_fmsubadd312ph_maskz(<32 x half> %a0, <32 x half> %a1, <32 x half> %a2, i32* %mask) {
  ;CHECK-LABEL: stack_fold_fmsubadd312ph_maskz:
  ;CHECK:       vfmsubadd132ph {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %neg = fneg <32 x half> %a1
  %2 = call <32 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %a2, <32 x half> %a0, <32 x half> %neg, i32 4)
  %3 = load i32, i32* %mask
  %4 = bitcast i32 %3 to <32 x i1>
  %5 = select <32 x i1> %4, <32 x half> %2, <32 x half> zeroinitializer
  ret <32 x half> %5
}
