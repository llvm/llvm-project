; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512vl,+avx512bw,+avx512dq,+avx512vbmi < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

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

define <4 x i32> @stack_fold_extracti32x4(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x4
  ;CHECK:       vextracti32x4 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <8 x i32> %a0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %2 = shufflevector <8 x i32> %1, <8 x i32> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_extracti64x2(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_extracti64x2
  ;CHECK:       vextracti64x2 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <4 x i64> %a0, <i64 1, i64 1, i64 1, i64 1>
  %2 = shufflevector <4 x i64> %1, <4 x i64> %a1, <2 x i32> <i32 2, i32 3>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <2 x i64> %2
}

define <8 x i32> @stack_fold_inserti32x4(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_inserti32x4
  ;CHECK:       vinserti32x4 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; add forces execution domain
  %3 = add <8 x i32> %2, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %3
}

define <4 x i64> @stack_fold_inserti64x2(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_inserti64x2
  ;CHECK:       vinserti64x2 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <2 x i64> %a0, <2 x i64> %a1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; add forces execution domain
  %3 = add <4 x i64> %2, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %3
}

define <4 x float> @stack_fold_vpermt2ps(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2ps
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x float> @llvm.x86.avx512.mask.vpermi2var.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.vpermi2var.ps.128(<4 x float>, <4 x i32>, <4 x float>, i8)

define <4 x float> @stack_fold_vpermi2ps(<4 x i32> %x0, <4 x float> %x1, <4 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2ps
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x float> @llvm.x86.avx512.mask.vpermt2var.ps.128(<4 x i32> %x0, <4 x float> %x1, <4 x float> %x2, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.vpermt2var.ps.128(<4 x i32>, <4 x float>, <4 x float>, i8)

define <2 x double> @stack_fold_vpermt2pd(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2pd
  ;CHECK:       vpermt2pd {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x double> @llvm.x86.avx512.mask.vpermi2var.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2, i8 -1)
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.mask.vpermi2var.pd.128(<2 x double>, <2 x i64>, <2 x double>, i8)

define <2 x double> @stack_fold_vpermi2pd(<2 x i64> %x0, <2 x double> %x1, <2 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2pd
  ;CHECK:       vpermi2pd {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x double> @llvm.x86.avx512.mask.vpermt2var.pd.128(<2 x i64> %x0, <2 x double> %x1, <2 x double> %x2, i8 -1)
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.mask.vpermt2var.pd.128(<2 x i64>, <2 x double>, <2 x double>, i8)

define <4 x i32> @stack_fold_vpermt2d(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2d
  ;CHECK:       vpermt2d {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpermi2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.vpermi2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32> @stack_fold_vpermi2d(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2d
  ;CHECK:       vpermi2d {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 -1)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.vpermt2var.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <2 x i64> @stack_fold_vpermt2q(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2q
  ;CHECK:       vpermt2q {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x i64> @llvm.x86.avx512.mask.vpermi2var.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.vpermi2var.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <2 x i64> @stack_fold_vpermi2q(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2q
  ;CHECK:       vpermi2q {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x i64> @llvm.x86.avx512.mask.vpermt2var.q.128(<2 x i64> %x0, <2 x i64> %x1, <2 x i64> %x2, i8 -1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.vpermt2var.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <8 x i16> @stack_fold_vpermt2w(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2w
  ;CHECK:       vpermt2w {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <8 x i16> @stack_fold_vpermi2w(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2w
  ;CHECK:       vpermi2w {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16> %x0, <8 x i16> %x1, <8 x i16> %x2, i8 -1)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i8> @stack_fold_vpermt2b(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2b
  ;CHECK:       vpermt2b {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <16 x i8> @stack_fold_vpermi2b(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2b
  ;CHECK:       vpermi2b {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.128(<16 x i8> %x0, <16 x i8> %x1, <16 x i8> %x2, i16 -1)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <8 x float> @stack_fold_vpermt2ps_ymm(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2ps_ymm
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x float> @llvm.x86.avx512.mask.vpermi2var.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.vpermi2var.ps.256(<8 x float>, <8 x i32>, <8 x float>, i8)

define <8 x float> @stack_fold_vpermi2ps_ymm(<8 x i32> %x0, <8 x float> %x1, <8 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2ps_ymm
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x float> @llvm.x86.avx512.mask.vpermt2var.ps.256(<8 x i32> %x0, <8 x float> %x1, <8 x float> %x2, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.vpermt2var.ps.256(<8 x i32>, <8 x float>, <8 x float>, i8)

define <4 x double> @stack_fold_vpermt2pd_ymm(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2pd_ymm
  ;CHECK:       vpermt2pd {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x double> @llvm.x86.avx512.mask.vpermi2var.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2, i8 -1)
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx512.mask.vpermi2var.pd.256(<4 x double>, <4 x i64>, <4 x double>, i8)

define <4 x double> @stack_fold_vpermi2pd_ymm(<4 x i64> %x0, <4 x double> %x1, <4 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2pd_ymm
  ;CHECK:       vpermi2pd {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x double> @llvm.x86.avx512.mask.vpermt2var.pd.256(<4 x i64> %x0, <4 x double> %x1, <4 x double> %x2, i8 -1)
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx512.mask.vpermt2var.pd.256(<4 x i64>, <4 x double>, <4 x double>, i8)

define <8 x i32> @stack_fold_vpermt2d_ymm(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2d_ymm
  ;CHECK:       vpermt2d {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpermi2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.vpermi2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32> @stack_fold_vpermi2d_ymm(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2d_ymm
  ;CHECK:       vpermi2d {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 -1)
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.vpermt2var.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i64> @stack_fold_vpermt2q_ymm(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2q_ymm
  ;CHECK:       vpermt2q {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i64> @llvm.x86.avx512.mask.vpermi2var.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.vpermi2var.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <4 x i64> @stack_fold_vpermi2q_ymm(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2q_ymm
  ;CHECK:       vpermi2q {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i64> @llvm.x86.avx512.mask.vpermt2var.q.256(<4 x i64> %x0, <4 x i64> %x1, <4 x i64> %x2, i8 -1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.vpermt2var.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define <16 x i16> @stack_fold_vpermt2w_ymm(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2w_ymm
  ;CHECK:       vpermt2w {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.vpermi2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i16> @stack_fold_vpermi2w_ymm(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2w_ymm
  ;CHECK:       vpermi2w {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16> %x0, <16 x i16> %x1, <16 x i16> %x2, i16 -1)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.vpermt2var.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <32 x i8> @stack_fold_vpermt2b_ymm(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2b_ymm
  ;CHECK:       vpermt2b {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <32 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.vpermi2var.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <32 x i8> @stack_fold_vpermi2b_ymm(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2b_ymm
  ;CHECK:       vpermi2b {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <32 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.256(<32 x i8> %x0, <32 x i8> %x1, <32 x i8> %x2, i32 -1)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.vpermt2var.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <4 x i32> @stack_fold_pmovsxbd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbd
  ;CHECK:       vpmovsxbd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pmovsxbq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbq
  ;CHECK:       vpmovsxbq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i8> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i16> @stack_fold_pmovsxbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbw
  ;CHECK:       vpmovsxbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i16>
  ret <8 x i16> %3
}

define <2 x i64> @stack_fold_pmovsxdq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxdq
  ;CHECK:       vpmovsxdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i32> @stack_fold_pmovsxwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwd
  ;CHECK:       vpmovsxwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pmovsxwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwq
  ;CHECK:       vpmovsxwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i16> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i32> @stack_fold_pmovzxbd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbd
  ;CHECK:       vpmovzxbd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 17, i32 18, i32 1, i32 19, i32 20, i32 21, i32 2, i32 22, i32 23, i32 24, i32 3, i32 25, i32 26, i32 27>
  %3 = bitcast <16 x i8> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pmovzxbq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbq
  ;CHECK:       vpmovzxbq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 1, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
  %3 = bitcast <16 x i8> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i16> @stack_fold_pmovzxbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbw
  ;CHECK:       vpmovzxbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %3 = bitcast <16 x i8> %2 to <8 x i16>
  ret <8 x i16> %3
}

define <2 x i64> @stack_fold_pmovzxdq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxdq
  ;CHECK:       vpmovzxdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %3 = bitcast <4 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i32> @stack_fold_pmovzxwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwd
  ;CHECK:       vpmovzxwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %3 = bitcast <8 x i16> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pmovzxwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwq
  ;CHECK:       vpmovzxwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 9, i32 10, i32 1, i32 11, i32 12, i32 13>
  %3 = bitcast <8 x i16> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i32> @stack_fold_pmovsxbd_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbd_ymm
  ;CHECK:       vpmovsxbd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <4 x i64> @stack_fold_pmovsxbq_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbq_ymm
  ;CHECK:       pmovsxbq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <16 x i16> @stack_fold_pmovsxbw_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxbw_ymm
  ;CHECK:       vpmovsxbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <16 x i8> %a0 to <16 x i16>
  ret <16 x i16> %2
}

define <4 x i64> @stack_fold_pmovsxdq_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxdq_ymm
  ;CHECK:       vpmovsxdq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <4 x i32> %a0 to <4 x i64>
  ret <4 x i64> %2
}

define <8 x i32> @stack_fold_pmovsxwd_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwd_ymm
  ;CHECK:       vpmovsxwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sext <8 x i16> %a0 to <8 x i32>
  ret <8 x i32> %2
}

define <4 x i64> @stack_fold_pmovsxwq_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovsxwq_ymm
  ;CHECK:       vpmovsxwq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i32> @stack_fold_pmovzxbd_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbd_ymm
  ;CHECK:       vpmovzxbd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = zext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <4 x i64> @stack_fold_pmovzxbq_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbq_ymm
  ;CHECK:       vpmovzxbq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <16 x i16> @stack_fold_pmovzxbw_ymm(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxbw_ymm
  ;CHECK:       vpmovzxbw {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <16 x i8> %a0 to <16 x i16>
  ret <16 x i16> %2
}

define <4 x i64> @stack_fold_pmovzxdq_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxdq_ymm
  ;CHECK:       vpmovzxdq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <4 x i32> %a0 to <4 x i64>
  ret <4 x i64> %2
}

define <8 x i32> @stack_fold_pmovzxwd_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pmovzxwd_ymm
  ;CHECK:       vpmovzxwd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = zext <8 x i16> %a0 to <8 x i32>
  ret <8 x i32> %2
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
