; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512vl,+avx512dq < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x double> @stack_fold_addpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_addpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd_ymm
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_addps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_addps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps_ymm
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <8 x float> %a0, %a1
  ret <8 x float> %2
}

define <2 x double> @stack_fold_andnpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, <i64 -1, i64 -1>
  %5 = and <2 x i64> %4, %3
  %6 = bitcast <2 x i64> %5 to <2 x double>
  ; fadd forces execution domain
  %7 = fadd <2 x double> %6, <double 0x0, double 0x0>
  ret <2 x double> %7
}

define <4 x double> @stack_fold_andnpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd_ymm
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %5 = and <4 x i64> %4, %3
  %6 = bitcast <4 x i64> %5 to <4 x double>
  ; fadd forces execution domain
  %7 = fadd <4 x double> %6, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %7
}

define <4 x float> @stack_fold_andnps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps
  ;CHECK:       vandnps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, <i64 -1, i64 -1>
  %5 = and <2 x i64> %4, %3
  %6 = bitcast <2 x i64> %5 to <4 x float>
  ; fadd forces execution domain
  %7 = fadd <4 x float> %6, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %7
}

define <8 x float> @stack_fold_andnps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps_ymm
  ;CHECK:       vandnps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %5 = and <4 x i64> %4, %3
  %6 = bitcast <4 x i64> %5 to <8 x float>
  ; fadd forces execution domain
  %7 = fadd <8 x float> %6, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %7
}

define <2 x double> @stack_fold_andpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd
  ;CHECK:       vandpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_andpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd_ymm
  ;CHECK:       vandpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = and <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_andps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <4 x i32>
  %3 = bitcast <4 x float> %a1 to <4 x i32>
  %4 = and <4 x i32> %2, %3
  %5 = bitcast <4 x i32> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_andps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps_ymm
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <8 x i32>
  %3 = bitcast <8 x float> %a1 to <8 x i32>
  %4 = and <8 x i32> %2, %3
  %5 = bitcast <8 x i32> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

define i8 @stack_fold_cmppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-9]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double> %a0, <2 x double> %a1, i32 0)
  %2 = shufflevector <2 x i1> %res, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}
declare <2 x i1> @llvm.x86.avx512.cmp.pd.128(<2 x double>, <2 x double>, i32)

define i8 @stack_fold_cmppd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd_ymm
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%k[0-9]}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double> %a0, <4 x double> %a1, i32 0)
  %2 = shufflevector <4 x i1> %res, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}
declare <4 x i1> @llvm.x86.avx512.cmp.pd.256(<4 x double>, <4 x double>, i32)

define i8 @stack_fold_cmpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float> %a0, <4 x float> %a1, i32 0)
  %2 = shufflevector <4 x i1> %res, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast <8 x i1> %2 to i8
  ret i8 %3
}
declare <4 x i1> @llvm.x86.avx512.cmp.ps.128(<4 x float>, <4 x float>, i32)

define i8 @stack_fold_cmpps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps_ymm
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float> %a0, <8 x float> %a1, i32 0)
  %2 = bitcast <8 x i1> %res to i8
  ret i8 %2
}
declare <8 x i1> @llvm.x86.avx512.cmp.ps.256(<8 x float>, <8 x float>, i32)

define <2 x double> @stack_fold_divpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_divpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd_ymm
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_divps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_divps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps_ymm
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv <8 x float> %a0, %a1
  ret <8 x float> %2
}

define <2 x double> @stack_fold_cvtdq2pd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sitofp <2 x i32> %2 to <2 x double>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_cvtdq2pd_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd_ymm
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sitofp <4 x i32> %a0 to <4 x double>
  ret <4 x double> %2
}

define <2 x double> @stack_fold_cvtudq2pd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtudq2pd
  ;CHECK:   vcvtudq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = uitofp <2 x i32> %2 to <2 x double>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_cvtudq2pd_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtudq2pd_ymm
  ;CHECK:   vcvtudq2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = uitofp <4 x i32> %a0 to <4 x double>
  ret <4 x double> %2
}

define <2 x float> @stack_fold_cvtpd2ps(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2psx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fptrunc <2 x double> %a0 to <2 x float>
  ret <2 x float> %2
}

define <4 x float> @stack_fold_cvtpd2ps_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps_ymm
  ;CHECK:   vcvtpd2psy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fptrunc <4 x double> %a0 to <4 x float>
  ret <4 x float> %2
}

define <2 x double> @stack_fold_maxpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone

define <2 x double> @stack_fold_maxpd_commutable(<2 x double> %a0, <2 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_commutable
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}

define <4 x double> @stack_fold_maxpd_ymm(<4 x double> %a0, <4 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd_ymm
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x double> @stack_fold_maxpd_ymm_commutable(<4 x double> %a0, <4 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_ymm_commutable
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}

define <4 x float> @stack_fold_maxps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_maxps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_commutable
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define <8 x float> @stack_fold_maxps_ymm(<8 x float> %a0, <8 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps_ymm
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <8 x float> @stack_fold_maxps_ymm_commutable(<8 x float> %a0, <8 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_ymm_commutable
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}

define <4 x float> @stack_fold_minps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_minps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_commutable
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define <8 x float> @stack_fold_minps_ymm(<8 x float> %a0, <8 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps_ymm
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <8 x float> @stack_fold_minps_ymm_commutable(<8 x float> %a0, <8 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_ymm_commutable
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}

define <2 x double> @stack_fold_mulpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_mulpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd_ymm
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_mulps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_mulps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps_ymm
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul <8 x float> %a0, %a1
  ret <8 x float> %2
}

define <2 x double> @stack_fold_orpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_orpd
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_orpd_ymm(<4 x double> %a0, <4 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_orpd_ymm
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = or <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_orps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <4 x i32>
  %3 = bitcast <4 x float> %a1 to <4 x i32>
  %4 = or <4 x i32> %2, %3
  %5 = bitcast <4 x i32> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_orps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps_ymm
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <8 x i32>
  %3 = bitcast <8 x float> %a1 to <8 x i32>
  %4 = or <8 x i32> %2, %3
  %5 = bitcast <8 x i32> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

define <4 x double> @stack_fold_shuff64x2_maskz(<4 x double> %a, <4 x double> %b, i8 %mask) {
  ;CHECK-LABEL: stack_fold_shuff64x2_maskz
  ;CHECK:   vshuff64x2 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x double> %2, <4 x double> zeroinitializer
  ret <4 x double> %5
}

define <8 x float> @stack_fold_shuff32x4_maskz(<8 x float> %a, <8 x float> %b, i8 %mask) {
  ;CHECK-LABEL: stack_fold_shuff32x4_maskz
  ;CHECK:   vshuff32x4 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x float> %2, <8 x float> zeroinitializer
  ret <8 x float> %4
}

define <4 x float> @stack_fold_shufps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps
  ;CHECK:       vshufps $200, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  ret <4 x float> %2
}

define <4 x float> @stack_fold_shufps_mask(<4 x float>* %passthru, <4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_shufps_mask
  ;CHECK:       vshufps $200, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = load <4 x float>, <4 x float>* %passthru
  %6 = select <4 x i1> %4, <4 x float> %2, <4 x float> %5
  ret <4 x float> %6
}

define <4 x float> @stack_fold_shufps_maskz(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_shufps_maskz
  ;CHECK:       vshufps $200, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = select <4 x i1> %4, <4 x float> %2, <4 x float> zeroinitializer
  ret <4 x float> %5
}

define <8 x float> @stack_fold_shufps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps_ymm
  ;CHECK:       vshufps $148, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 0, i32 1, i32 9, i32 10, i32 4, i32 5, i32 13, i32 14>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_subpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_subpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd_ymm
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_subps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_subps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps_ymm
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <8 x float> %a0, %a1
  ret <8 x float> %2
}

define <2 x double> @stack_fold_xorpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_xorpd
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_xorpd_ymm(<4 x double> %a0, <4 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_xorpd_ymm
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_xorps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <4 x i32>
  %3 = bitcast <4 x float> %a1 to <4 x i32>
  %4 = xor <4 x i32> %2, %3
  %5 = bitcast <4 x i32> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_xorps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps_ymm
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <8 x i32>
  %3 = bitcast <8 x float> %a1 to <8 x i32>
  %4 = xor <8 x i32> %2, %3
  %5 = bitcast <8 x i32> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

define <4 x float> @stack_fold_extractf32x4(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_extractf32x4
  ;CHECK:       vextractf128 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = shufflevector <8 x float> %a0, <8 x float> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x float> %1
}

define <2 x double> @stack_fold_extractf64x2(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_extractf64x2
  ;CHECK:       vextractf128 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = shufflevector <4 x double> %a0, <4 x double> %a1, <2 x i32> <i32 2, i32 3>
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <2 x double> %1
}

define <8 x float> @stack_fold_insertf32x4(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertf32x4
  ;CHECK:       vinsertf128 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %2
}

define <4 x double> @stack_fold_insertf64x2(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_insertf64x2
  ;CHECK:       vinsertf128 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %2
}

define <4 x float> @stack_fold_vpermt2ps(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2ps
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512.vpermi2var.ps.128(<4 x float> %x0, <4 x i32> %x1, <4 x float> %x2)
  ret <4 x float> %2
}

define <4 x float> @stack_fold_vpermi2ps(<4 x i32> %x0, <4 x float> %x1, <4 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2ps
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx512.vpermi2var.ps.128(<4 x float> %x1, <4 x i32> %x0, <4 x float> %x2)
  ret <4 x float> %2
}

define <2 x double> @stack_fold_vpermt2pd(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2pd
  ;CHECK:       vpermt2pd {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x double> @llvm.x86.avx512.vpermi2var.pd.128(<2 x double> %x0, <2 x i64> %x1, <2 x double> %x2)
  ret <2 x double> %2
}

define <2 x double> @stack_fold_vpermi2pd(<2 x i64> %x0, <2 x double> %x1, <2 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2pd
  ;CHECK:       vpermi2pd {{-?[0-9]*}}(%rsp), %xmm1, %xmm0 # 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x double> @llvm.x86.avx512.vpermi2var.pd.128(<2 x double> %x1, <2 x i64> %x0, <2 x double> %x2)
  ret <2 x double> %2
}

define <8 x float> @stack_fold_vpermt2ps_ymm(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2ps_ymm
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512.vpermi2var.ps.256(<8 x float> %x0, <8 x i32> %x1, <8 x float> %x2)
  ret <8 x float> %2
}

define <8 x float> @stack_fold_vpermi2ps_ymm(<8 x i32> %x0, <8 x float> %x1, <8 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2ps_ymm
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx512.vpermi2var.ps.256(<8 x float> %x1, <8 x i32> %x0, <8 x float> %x2)
  ret <8 x float> %2
}

define <4 x double> @stack_fold_vpermt2pd_ymm(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2pd_ymm
  ;CHECK:       vpermt2pd {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx512.vpermi2var.pd.256(<4 x double> %x0, <4 x i64> %x1, <4 x double> %x2)
  ret <4 x double> %2
}

define <4 x double> @stack_fold_vpermi2pd_ymm(<4 x i64> %x0, <4 x double> %x1, <4 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2pd_ymm
  ;CHECK:       vpermi2pd {{-?[0-9]*}}(%rsp), %ymm1, %ymm0 # 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx512.vpermi2var.pd.256(<4 x double> %x1, <4 x i64> %x0, <4 x double> %x2)
  ret <4 x double> %2
}

define <4 x double> @stack_fold_permpd(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permpd
  ;CHECK:   vpermpd $235, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> undef, <4 x i32> <i32 3, i32 2, i32 2, i32 3>
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %3
}

define <4 x double> @stack_fold_permpdvar(<4 x i64> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_permpdvar
  ;CHECK:   vpermpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx512.permvar.df.256(<4 x double> %a1, <4 x i64> %a0)
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, zeroinitializer
  ret <4 x double> %3
}
declare <4 x double> @llvm.x86.avx512.permvar.df.256(<4 x double>, <4 x i64>) nounwind readonly

define <8 x float> @stack_fold_permps(<8 x i32> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_permps
  ;CHECK:       vpermps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx2.permps(<8 x float> %a1, <8 x i32> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx2.permps(<8 x float>, <8 x i32>) nounwind readonly

define <2 x double> @stack_fold_permilpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd
  ;CHECK:   vpermilpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_permilpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd_ymm
  ;CHECK:   vpermilpd $5, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x double> %2
}

define <2 x double> @stack_fold_permilpdvar(<2 x double> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permilpdvar
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double> %a0, <2 x i64> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double>, <2 x i64>) nounwind readnone

define <4 x double> @stack_fold_permilpdvar_ymm(<4 x double> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permilpdvar_ymm
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double> %a0, <4 x i64> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double>, <4 x i64>) nounwind readnone

define <4 x float> @stack_fold_permilps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_permilps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps_ymm
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x float> %2
}

define <4 x float> @stack_fold_permilpsvar(<4 x float> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permilpsvar
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %a0, <4 x i32> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float>, <4 x i32>) nounwind readnone

define <8 x float> @stack_fold_permilpsvar_ymm(<8 x float> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permilpsvar_ymm
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float> %a0, <8 x i32> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float>, <8 x i32>) nounwind readnone

define <8 x float> @stack_fold_permilpsvar_ymm_maskz(<8 x float> %a0, <8 x i32> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permilpsvar_ymm_maskz
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float> %a0, <8 x i32> %a1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x float> %2, <8 x float> zeroinitializer
  ret <8 x float> %4
}

declare <4 x float> @llvm.x86.avx512.vpermi2var.ps.128(<4 x float>, <4 x i32>, <4 x float>)
declare <2 x double> @llvm.x86.avx512.vpermi2var.pd.128(<2 x double>, <2 x i64>, <2 x double>)
declare <8 x float> @llvm.x86.avx512.vpermi2var.ps.256(<8 x float>, <8 x i32>, <8 x float>)
declare <4 x double> @llvm.x86.avx512.vpermi2var.pd.256(<4 x double>, <4 x i64>, <4 x double>)

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
