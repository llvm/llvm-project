; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512dq < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <64 x i8> @stack_fold_paddb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddb
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <64 x i8> %a0, %a1
  ret <64 x i8> %2
}

define <16 x i32> @stack_fold_paddd(<16 x i32> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_paddd
  ;CHECK:       vpaddd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <16 x i32> %a0, %a1
  ret <16 x i32> %2
}

define <8 x i64> @stack_fold_paddq(<8 x i64> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_paddq
  ;CHECK:       vpaddq {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <8 x i64> %a0, %a1
  ret <8 x i64> %2
}

define <64 x i8> @stack_fold_paddsb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddsb
  ;CHECK:       vpaddsb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <64 x i8> @llvm.x86.avx512.mask.padds.b.512(<64 x i8> %a0, <64 x i8> %a1, <64 x i8> undef, i64 -1)
  ret <64 x i8> %2
}
declare <64 x i8> @llvm.x86.avx512.mask.padds.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64) nounwind readnone

define <32 x i16> @stack_fold_paddsw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddsw
  ;CHECK:       vpaddsw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a0, <32 x i16> %a1, <32 x i16> undef, i32 -1)
  ret <32 x i16> %2
}
declare <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <64 x i8> @stack_fold_paddusb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddusb
  ;CHECK:       vpaddusb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <64 x i8> @llvm.x86.avx512.mask.paddus.b.512(<64 x i8> %a0, <64 x i8> %a1, <64 x i8> undef, i64 -1)
  ret <64 x i8> %2
}
declare <64 x i8> @llvm.x86.avx512.mask.paddus.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64) nounwind readnone

define <32 x i16> @stack_fold_paddusw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddusw
  ;CHECK:       vpaddusw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a0, <32 x i16> %a1, <32 x i16> undef, i32 -1)
  ret <32 x i16> %2
}
declare <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @stack_fold_paddw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddw
  ;CHECK:       vpaddw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = add <32 x i16> %a0, %a1
  ret <32 x i16> %2
}

define i64 @stack_fold_pcmpeqb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqb
  ;CHECK:       vpcmpeqb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <64 x i8> %a0, %a1
  %3 = bitcast <64 x i1> %2 to i64
  ret i64 %3
}

define i16 @stack_fold_pcmpeqd(<16 x i32> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqd
  ;CHECK:       vpcmpeqd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <16 x i32> %a0, %a1
  %3 = bitcast <16 x i1> %2 to i16
  ret i16 %3
}

define i8 @stack_fold_pcmpeqq(<8 x i64> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqq
  ;CHECK:       vpcmpeqq {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <8 x i64> %a0, %a1
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}

define i32 @stack_fold_pcmpeqw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqw
  ;CHECK:       vpcmpeqw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-7]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = icmp eq <32 x i16> %a0, %a1
  %3 = bitcast <32 x i1> %2 to i32
  ret i32 %3
}

define <64 x i8> @stack_fold_psubb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubb
  ;CHECK:       vpsubb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <64 x i8> %a0, %a1
  ret <64 x i8> %2
}

define <16 x i32> @stack_fold_psubd(<16 x i32> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psubd
  ;CHECK:       vpsubd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <16 x i32> %a0, %a1
  ret <16 x i32> %2
}

define <8 x i64> @stack_fold_psubq(<8 x i64> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psubq
  ;CHECK:       vpsubq {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <8 x i64> %a0, %a1
  ret <8 x i64> %2
}

define <64 x i8> @stack_fold_psubsb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubsb
  ;CHECK:       vpsubsb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <64 x i8> @llvm.x86.avx512.mask.psubs.b.512(<64 x i8> %a0, <64 x i8> %a1, <64 x i8> undef, i64 -1)
  ret <64 x i8> %2
}
declare <64 x i8> @llvm.x86.avx512.mask.psubs.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64) nounwind readnone

define <32 x i16> @stack_fold_psubsw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubsw
  ;CHECK:       vpsubsw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a0, <32 x i16> %a1, <32 x i16> undef, i32 -1)
  ret <32 x i16> %2
}
declare <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <64 x i8> @stack_fold_psubusb(<64 x i8> %a0, <64 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubusb
  ;CHECK:       vpsubusb {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <64 x i8> @llvm.x86.avx512.mask.psubus.b.512(<64 x i8> %a0, <64 x i8> %a1, <64 x i8> undef, i64 -1)
  ret <64 x i8> %2
}
declare <64 x i8> @llvm.x86.avx512.mask.psubus.b.512(<64 x i8>, <64 x i8>, <64 x i8>, i64) nounwind readnone

define <32 x i16> @stack_fold_psubusw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubusw
  ;CHECK:       vpsubusw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a0, <32 x i16> %a1, <32 x i16> undef, i32 -1)
  ret <32 x i16> %2
}
declare <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @stack_fold_psubw(<32 x i16> %a0, <32 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubw
  ;CHECK:       vpsubw {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sub <32 x i16> %a0, %a1
  ret <32 x i16> %2
}

define <16 x i32> @stack_fold_ternlogd(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2) {
  ;CHECK-LABEL: stack_fold_ternlogd
  ;CHECK:       vpternlogd $33, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x i32> @llvm.x86.avx512.mask.pternlog.d.512(<16 x i32> %x0, <16 x i32> %x1, <16 x i32> %x2, i32 33, i16 -1)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.pternlog.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i32, i16)

define <8 x i64> @stack_fold_ternlogq(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2) {
  ;CHECK-LABEL: stack_fold_ternlogq
  ;CHECK:       vpternlogq $33, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i64> @llvm.x86.avx512.mask.pternlog.q.512(<8 x i64> %x0, <8 x i64> %x1, <8 x i64> %x2, i32 33, i8 -1)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.pternlog.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i32, i8)

define <16 x i8> @stack_fold_vpmovdb(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovdb
  ;CHECK:       vpmovdb %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmov.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i16> @stack_fold_vpmovdw(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovdw
  ;CHECK:       vpmovdw %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32> %a0, <16 x i16> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i16> %1
}
declare <16 x i16> @llvm.x86.avx512.mask.pmov.dw.512(<16 x i32>, <16 x i16>, i16)

define <8 x i32> @stack_fold_vpmovqd(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovqd
  ;CHECK:       vpmovqd %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64> %a0, <8 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i32> %1
}
declare <8 x i32> @llvm.x86.avx512.mask.pmov.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i16> @stack_fold_vpmovqw(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovqw
  ;CHECK:       vpmovqw %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmov.qw.512(<8 x i64>, <8 x i16>, i8)

define <32 x i8> @stack_fold_vpmovwb(<32 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovwb
  ;CHECK:       vpmovwb %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16> %a0, <32 x i8> undef, i32 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <32 x i8> %1
}
declare <32 x i8> @llvm.x86.avx512.mask.pmov.wb.512(<32 x i16>, <32 x i8>, i32)

define <16 x i8> @stack_fold_vpmovsdb(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsdb
  ;CHECK:       vpmovsdb %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i16> @stack_fold_vpmovsdw(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsdw
  ;CHECK:       vpmovsdw %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32> %a0, <16 x i16> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i16> %1
}
declare <16 x i16> @llvm.x86.avx512.mask.pmovs.dw.512(<16 x i32>, <16 x i16>, i16)

define <8 x i32> @stack_fold_vpmovsqd(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsqd
  ;CHECK:       vpmovsqd %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64> %a0, <8 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i32> %1
}
declare <8 x i32> @llvm.x86.avx512.mask.pmovs.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i16> @stack_fold_vpmovsqw(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovsqw
  ;CHECK:       vpmovsqw %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmovs.qw.512(<8 x i64>, <8 x i16>, i8)

define <32 x i8> @stack_fold_vpmovswb(<32 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovswb
  ;CHECK:       vpmovswb %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16> %a0, <32 x i8> undef, i32 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <32 x i8> %1
}
declare <32 x i8> @llvm.x86.avx512.mask.pmovs.wb.512(<32 x i16>, <32 x i8>, i32)

define <16 x i8> @stack_fold_vpmovusdb(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusdb
  ;CHECK:       vpmovusdb %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32> %a0, <16 x i8> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i8> %1
}
declare <16 x i8> @llvm.x86.avx512.mask.pmovus.db.512(<16 x i32>, <16 x i8>, i16)

define <16 x i16> @stack_fold_vpmovusdw(<16 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusdw
  ;CHECK:       vpmovusdw %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32> %a0, <16 x i16> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i16> %1
}
declare <16 x i16> @llvm.x86.avx512.mask.pmovus.dw.512(<16 x i32>, <16 x i16>, i16)

define <8 x i32> @stack_fold_vpmovusqd(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusqd
  ;CHECK:       vpmovusqd %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64> %a0, <8 x i32> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i32> %1
}
declare <8 x i32> @llvm.x86.avx512.mask.pmovus.qd.512(<8 x i64>, <8 x i32>, i8)

define <8 x i16> @stack_fold_vpmovusqw(<8 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovusqw
  ;CHECK:       vpmovusqw %zmm0, {{-?[0-9]*}}(%rsp) # 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64> %a0, <8 x i16> undef, i8 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.avx512.mask.pmovus.qw.512(<8 x i64>, <8 x i16>, i8)

define <32 x i8> @stack_fold_vpmovuswb(<32 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vpmovuswb
  ;CHECK:       vpmovuswb %zmm0, {{-?[0-9]*}}(%rsp) # 32-byte Folded Spill
  %1 = call <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16> %a0, <32 x i8> undef, i32 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <32 x i8> %1
}
declare <32 x i8> @llvm.x86.avx512.mask.pmovus.wb.512(<32 x i16>, <32 x i8>, i32)

define <4 x i32> @stack_fold_extracti32x4(<16 x i32> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x4
  ;CHECK:       vextracti32x4 $3, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <16 x i32> %a0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %2 = shufflevector <16 x i32> %1, <16 x i32> %a1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_extracti64x2(<8 x i64> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_extracti64x2
  ;CHECK:       vextracti64x2 $3, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  ; add forces execution domain
  %1 = add <8 x i64> %a0, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %2 = shufflevector <8 x i64> %1, <8 x i64> %a1, <2 x i32> <i32 6, i32 7>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <2 x i64> %2
}

define <8 x i32> @stack_fold_extracti32x8(<16 x i32> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x8
  ;CHECK:       vextracti32x8 $1, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 32-byte Folded Spill
  ; add forces execution domain
  %1 = add <16 x i32> %a0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %2 = shufflevector <16 x i32> %1, <16 x i32> %a1, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x i32> %2
}

define <4 x i64> @stack_fold_extracti64x4(<8 x i64> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_extracti64x4
  ;CHECK:       vextracti64x4 $1, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 32-byte Folded Spill
  ; add forces execution domain
  %1 = add <8 x i64> %a0, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %2 = shufflevector <8 x i64> %1, <8 x i64> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x i64> %2
}

define <16 x i32> @stack_fold_inserti32x8(<8 x i32> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_inserti32x8
  ;CHECK:       vinserti32x8 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x i32> %a0, <8 x i32> %a1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ; add forces execution domain
  %3 = add <16 x i32> %2, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x i32> %3
}

define <8 x i64> @stack_fold_inserti64x4(<4 x i64> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_inserti64x4
  ;CHECK:       vinserti64x4 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i64> %a0, <4 x i64> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; add forces execution domain
  %3 = add <8 x i64> %2, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  ret <8 x i64> %3
}
