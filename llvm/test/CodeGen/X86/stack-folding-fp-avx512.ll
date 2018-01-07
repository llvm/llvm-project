; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512dq < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <8 x double> @stack_fold_addpd_zmm(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd_zmm
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <8 x double> %a0, %a1
  ret <8 x double> %2
}

define <8 x double> @stack_fold_addpd_zmm_kz(<8 x double> %a0, <8 x double> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_addpd_zmm_kz
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <8 x double> %a1, %a0
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> zeroinitializer
  ret <8 x double> %4
}

define <16 x float> @stack_fold_addps_zmm(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps_zmm
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <16 x float> %a0, %a1
  ret <16 x float> %2
}

define <16 x float> @stack_fold_addps_zmm_kz(<16 x float> %a0, <16 x float> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_addps_zmm_kz
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd <16 x float> %a1, %a0
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x float> %2, <16 x float> zeroinitializer
  ret <16 x float> %4
}

define double @stack_fold_addsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_addsd
  ;CHECK:       vaddsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_addsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsd_int
  ;CHECK:       vaddsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fadd double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_addss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_addss
  ;CHECK:       vaddss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fadd float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_addss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addss_int
  ;CHECK:       vaddss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fadd float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <8 x double> @stack_fold_andnpd_zmm(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd_zmm
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x double> %a0 to <8 x i64>
  %3 = bitcast <8 x double> %a1 to <8 x i64>
  %4 = xor <8 x i64> %2, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %5 = and <8 x i64> %4, %3
  %6 = bitcast <8 x i64> %5 to <8 x double>
  ; fadd forces execution domain
  %7 = fadd <8 x double> %6, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %7
}

define <16 x float> @stack_fold_andnps_zmm(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps_zmm
  ;CHECK:       vandnps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <16 x float> %a0 to <16 x i32>
  %3 = bitcast <16 x float> %a1 to <16 x i32>
  %4 = xor <16 x i32> %2, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %5 = and <16 x i32> %4, %3
  %6 = bitcast <16 x i32> %5 to <16 x float>
  ; fadd forces execution domain
  %7 = fadd <16 x float> %6, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <16 x float> %7
}

define <8 x double> @stack_fold_andpd_zmm(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd_zmm
  ;CHECK:       vandpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x double> %a0 to <8 x i64>
  %3 = bitcast <8 x double> %a1 to <8 x i64>
  %4 = and <8 x i64> %2, %3
  %5 = bitcast <8 x i64> %4 to <8 x double>
  ; fadd forces execution domain
  %6 = fadd <8 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %6
}

define <16 x float> @stack_fold_andps_zmm(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps_zmm
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <16 x float> %a0 to <16 x i32>
  %3 = bitcast <16 x float> %a1 to <16 x i32>
  %4 = and <16 x i32> %2, %3
  %5 = bitcast <16 x i32> %4 to <16 x float>
  ; fadd forces execution domain
  %6 = fadd <16 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <16 x float> %6
}

define i8 @stack_fold_cmppd(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-9]}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %a0, <8 x double> %a1, i32 0, i8 -1, i32 4)
  ret i8 %res
}
declare i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> , <8 x double> , i32, i8, i32)

define i16 @stack_fold_cmpps(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%k[0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %a0, <16 x float> %a1, i32 0, i16 -1, i32 4)
  ret i16 %res
}
declare i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> , <16 x float> , i32, i16, i32)

define <2 x double> @stack_fold_divsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divsd_int
  ;CHECK:       vdivsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fdiv double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_divss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_divss
  ;CHECK:       vdivss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fdiv float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_divss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divss_int
  ;CHECK:       vdivss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fdiv float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <8 x double> @stack_fold_cvtdq2pd(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = sitofp <8 x i32> %a0 to <8 x double>
  ret <8 x double> %2
}

define <8 x double> @stack_fold_cvtudq2pd(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtudq2pd
  ;CHECK:   vcvtudq2pd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = uitofp <8 x i32> %a0 to <8 x double>
  ret <8 x double> %2
}

define <8 x float> @stack_fold_cvtpd2ps(<8 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fptrunc <8 x double> %a0 to <8 x float>
  ret <8 x float> %2
}

define <16 x float> @stack_fold_cvtph2ps(<16 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_cvtph2ps
  ;CHECK:   vcvtph2ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> undef, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16>, <16 x float>, i16, i32) nounwind readonly

define <16 x i16> @stack_fold_cvtps2ph(<16 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2ph
  ;CHECK:   vcvtps2ph $0, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 32-byte Folded Spill
  %1 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %a0, i32 0, <16 x i16> undef, i16 -1)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <16 x i16> %1
}
declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float>, i32, <16 x i16>, i16) nounwind readonly

define <4 x float> @stack_fold_insertps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertps
  ;CHECK:       vinsertps $17, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  ;CHECK-NEXT:                                                                              {{.*#+}} xmm0 = zero,mem[0],xmm0[2,3]
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a0, <4 x float> %a1, i8 209)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x double> @stack_fold_maxpd_zmm(<8 x double> %a0, <8 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd_zmm
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %2
}
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @stack_fold_maxpd_zmm_commutable(<8 x double> %a0, <8 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_zmm_commutable
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %2
}

define <8 x double> @stack_fold_maxpd_zmm_commutable_kz(<8 x double> %a0, <8 x double> %a1, i8 %mask) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_zmm_commutable_kz
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a1, <8 x double> %a0, <8 x double> zeroinitializer, i8 %mask, i32 4)
  ret <8 x double> %2
}

define <16 x float> @stack_fold_maxps_zmm(<16 x float> %a0, <16 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps_zmm
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @stack_fold_maxps_zmm_commutable(<16 x float> %a0, <16 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_zmm_commutable
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_maxps_zmm_commutable_kz(<16 x float> %a0, <16 x float> %a1, i16 %mask) #1 {
  ;CHECK-LABEL: stack_fold_maxps_zmm_commutable_kz
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a1, <16 x float> %a0, <16 x float> zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %2
}

define <8 x double> @stack_fold_minpd_zmm(<8 x double> %a0, <8 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minpd_zmm
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %2
}
declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @stack_fold_minpd_zmm_commutable(<8 x double> %a0, <8 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minpd_zmm_commutable
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %2
}

define <8 x double> @stack_fold_minpd_zmm_commutable_kz(<8 x double> %a0, <8 x double> %a1, i8 %mask) #1 {
  ;CHECK-LABEL: stack_fold_minpd_zmm_commutable_kz
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a1, <8 x double> %a0, <8 x double> zeroinitializer, i8 %mask, i32 4)
  ret <8 x double> %2
}

define <16 x float> @stack_fold_minps_zmm(<16 x float> %a0, <16 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps_zmm
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @stack_fold_minps_zmm_commutable(<16 x float> %a0, <16 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_zmm_commutable
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %2
}

define <16 x float> @stack_fold_minps_zmm_commutable_kz(<16 x float> %a0, <16 x float> %a1, i16 %mask) #1 {
  ;CHECK-LABEL: stack_fold_minps_zmm_commutable_kz
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a1, <16 x float> %a0, <16 x float> zeroinitializer, i16 %mask, i32 4)
  ret <16 x float> %2
}

define double @stack_fold_mulsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_mulsd
  ;CHECK:       vmulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_mulsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulsd_int
  ;CHECK:       vmulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fmul double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_mulss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_mulss
  ;CHECK:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fmul float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_mulss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulss_int
  ;CHECK:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fmul float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <8 x double> @stack_fold_orpd_zmm(<8 x double> %a0, <8 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_orpd_zmm
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x double> %a0 to <8 x i64>
  %3 = bitcast <8 x double> %a1 to <8 x i64>
  %4 = or <8 x i64> %2, %3
  %5 = bitcast <8 x i64> %4 to <8 x double>
  ; fadd forces execution domain
  %6 = fadd <8 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %6
}

define <16 x float> @stack_fold_orps_zmm(<16 x float> %a0, <16 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_orps_zmm
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <16 x float> %a0 to <16 x i32>
  %3 = bitcast <16 x float> %a1 to <16 x i32>
  %4 = or <16 x i32> %2, %3
  %5 = bitcast <16 x i32> %4 to <16 x float>
  ; fadd forces execution domain
  %6 = fadd <16 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <16 x float> %6
}

define <8 x double> @stack_fold_subpd_zmm(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd_zmm
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <8 x double> %a0, %a1
  ret <8 x double> %2
}

define <16 x float> @stack_fold_subps_zmm(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps_zmm
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub <16 x float> %a0, %a1
  ret <16 x float> %2
}

define double @stack_fold_subsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_subsd
  ;CHECK:       vsubsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_subsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subsd_int
  ;CHECK:       vsubsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fsub double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_subss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_subss
  ;CHECK:       vsubss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = fsub float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_subss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subss_int
  ;CHECK:       vsubss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fsub float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <8 x double> @stack_fold_xorpd_zmm(<8 x double> %a0, <8 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_xorpd_zmm
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <8 x double> %a0 to <8 x i64>
  %3 = bitcast <8 x double> %a1 to <8 x i64>
  %4 = xor <8 x i64> %2, %3
  %5 = bitcast <8 x i64> %4 to <8 x double>
  ; fadd forces execution domain
  %6 = fadd <8 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %6
}

define <16 x float> @stack_fold_xorps_zmm(<16 x float> %a0, <16 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_xorps_zmm
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast <16 x float> %a0 to <16 x i32>
  %3 = bitcast <16 x float> %a1 to <16 x i32>
  %4 = xor <16 x i32> %2, %3
  %5 = bitcast <16 x i32> %4 to <16 x float>
  ; fadd forces execution domain
  %6 = fadd <16 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <16 x float> %6
}

define i32 @stack_fold_extractps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_extractps
  ;CHECK:       vextractps $1, {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 4-byte Folded Spill
  ;CHECK:       movl    {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Reload
  %1 = extractelement <4 x float> %a0, i32 1
  %2 = bitcast float %1 to i32
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret i32 %2
}

define <4 x float> @stack_fold_extracti32x4(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x4
  ;CHECK:       vextractf32x4 $3, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = shufflevector <16 x float> %a0, <16 x float> %a1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x float> %1
}

define <2 x double> @stack_fold_extractf64x2(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_extractf64x2
  ;CHECK:       vextractf32x4 $3, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = shufflevector <8 x double> %a0, <8 x double> %a1, <2 x i32> <i32 6, i32 7>
  %2 = tail call <2 x double> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <2 x double> %1
}

define <8 x float> @stack_fold_extracti32x8(<16 x float> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_extracti32x8
  ;CHECK:       vextractf64x4 $1, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 32-byte Folded Spill
  %1 = shufflevector <16 x float> %a0, <16 x float> %a1, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <8 x float> %1
}

define <4 x double> @stack_fold_extractf64x4(<8 x double> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_extractf64x4
  ;CHECK:       vextractf64x4 $1, {{%zmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 32-byte Folded Spill
  %1 = shufflevector <8 x double> %a0, <8 x double> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %2 = tail call <2 x double> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ret <4 x double> %1
}

define <16 x float> @stack_fold_insertf32x8(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertf32x8
  ;CHECK:       vinsertf64x4 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x float> %2
}

define <8 x double> @stack_fold_insertf64x4(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_insertf64x4
  ;CHECK:       vinsertf64x4 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %2
}

define <8 x double> @stack_fold_insertf64x4_mask(<8 x double> %passthru, <4 x double> %a0, <4 x double> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_insertf64x4_mask
  ;CHECK:       vinsertf64x4 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> %passthru
  ret <8 x double> %4
}

define <8 x double> @stack_fold_insertf64x4_maskz(<4 x double> %a0, <4 x double> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_insertf64x4_maskz
  ;CHECK:       vinsertf64x4 $1, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[1-7]}}} {z} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> zeroinitializer
  ret <8 x double> %4
}

define <16 x float> @stack_fold_vpermt2ps(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2ps
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), %zmm1, %zmm0 # 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float> %x0, <16 x i32> %x1, <16 x float> %x2, i16 -1)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float>, <16 x i32>, <16 x float>, i16)

define <16 x float> @stack_fold_vpermi2ps(<16 x i32> %x0, <16 x float> %x1, <16 x float> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2ps
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), %zmm1, %zmm0 # 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermt2var.ps.512(<16 x i32> %x0, <16 x float> %x1, <16 x float> %x2, i16 -1)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.vpermt2var.ps.512(<16 x i32>, <16 x float>, <16 x float>, i16)

define <16 x float> @stack_fold_vpermi2ps_mask(<16 x float> %x0, <16 x i32>* %x1, <16 x float> %x2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpermi2ps_mask
  ;CHECK:       vpermi2ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %x1b = load <16 x i32>, <16 x i32>* %x1
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermi2var.ps.512(<16 x float> %x0, <16 x i32> %x1b, <16 x float> %x2, i16 %mask)
  ret <16 x float> %res
}

define <16 x float> @stack_fold_vpermt2ps_mask(<16 x i32>* %x0, <16 x float> %x1, <16 x float> %x2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpermt2ps_mask
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %x0b = load <16 x i32>, <16 x i32>* %x0
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermt2var.ps.512(<16 x i32> %x0b, <16 x float> %x1, <16 x float> %x2, i16 %mask)
  ret <16 x float> %res
}

define <16 x float> @stack_fold_vpermt2ps_maskz(<16 x i32>* %x0, <16 x float> %x1, <16 x float> %x2, i16 %mask) {
  ;CHECK-LABEL: stack_fold_vpermt2ps_maskz
  ;CHECK:       vpermt2ps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %x0b = load <16 x i32>, <16 x i32>* %x0
  %res = call <16 x float> @llvm.x86.avx512.maskz.vpermt2var.ps.512(<16 x i32> %x0b, <16 x float> %x1, <16 x float> %x2, i16 %mask)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.maskz.vpermt2var.ps.512(<16 x i32>, <16 x float>, <16 x float>, i16)

define <8 x double> @stack_fold_vpermt2pd(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermt2pd
  ;CHECK:       vpermt2pd {{-?[0-9]*}}(%rsp), %zmm1, %zmm0 # 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %x0, <8 x i64> %x1, <8 x double> %x2, i8 -1)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double>, <8 x i64>, <8 x double>, i8)

define <8 x double> @stack_fold_vpermi2pd(<8 x i64> %x0, <8 x double> %x1, <8 x double> %x2) {
  ;CHECK-LABEL: stack_fold_vpermi2pd
  ;CHECK:       vpermi2pd {{-?[0-9]*}}(%rsp), %zmm1, %zmm0 # 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %res = call <8 x double> @llvm.x86.avx512.mask.vpermt2var.pd.512(<8 x i64> %x0, <8 x double> %x1, <8 x double> %x2, i8 -1)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.vpermt2var.pd.512(<8 x i64>, <8 x double>, <8 x double>, i8)

define <8 x double> @stack_fold_permpd(<8 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permpd
  ;CHECK:   vpermpd $235, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 3, i32 2, i32 2, i32 3, i32 7, i32 6, i32 6, i32 7>
  ; fadd forces execution domain
  %3 = fadd <8 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %3
}

define <8 x double> @stack_fold_permpd_mask(<8 x double>* %passthru, <8 x double> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permpd_mask
  ;CHECK:   vpermpd $235, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 3, i32 2, i32 2, i32 3, i32 7, i32 6, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x double>, <8 x double>* %passthru
  %5 = select <8 x i1> %3, <8 x double> %2, <8 x double> %4
  ; fadd forces execution domain
  %6 = fadd <8 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %6
}

define <8 x double> @stack_fold_permpd_maskz(<8 x double> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permpd_maskz
  ;CHECK:   vpermpd $235, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 3, i32 2, i32 2, i32 3, i32 7, i32 6, i32 6, i32 7>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> zeroinitializer
  ret <8 x double> %4
}

define <8 x double> @stack_fold_permpdvar(<8 x i64> %a0, <8 x double> %a1) {
  ;CHECK-LABEL: stack_fold_permpdvar
  ;CHECK:   vpermpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.permvar.df.512(<8 x double> %a1, <8 x i64> %a0, <8 x double> undef, i8 -1)
  ; fadd forces execution domain
  %3 = fadd <8 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0, double 0x0>
  ret <8 x double> %3
}
declare <8 x double> @llvm.x86.avx512.mask.permvar.df.512(<8 x double>, <8 x i64>, <8 x double>, i8) nounwind readonly

define <16 x float> @stack_fold_permps(<16 x i32> %a0, <16 x float> %a1) {
  ;CHECK-LABEL: stack_fold_permps
  ;CHECK:       vpermps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.permvar.sf.512(<16 x float> %a1, <16 x i32> %a0, <16 x float> undef, i16 -1)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512.mask.permvar.sf.512(<16 x float>, <16 x i32>, <16 x float>, i16) nounwind readonly

define <8 x double> @stack_fold_permilpd_zmm(<8 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd_zmm
  ;CHECK:   vpermilpd $85, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x double> %2
}

define <8 x double> @stack_fold_permilpd_zmm_mask(<8 x double>* %passthru, <8 x double> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permilpd_zmm_mask
  ;CHECK:   vpermilpd $85, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  %3 = bitcast i8 %mask to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x double>, <8 x double>* %passthru
  %5 = select <8 x i1> %3, <8 x double> %2, <8 x double> %4
  ret <8 x double> %5
}

define <8 x double> @stack_fold_permilpd_zmm_maskz(<8 x double> %a0, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permilpd_zmm_maskz
  ;CHECK:   vpermilpd $85, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <8 x double> %a0, <8 x double> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> zeroinitializer
  ret <8 x double> %4
}

define <8 x double> @stack_fold_permilpdvar_zmm(<8 x double> %a0, <8 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permilpdvar_zmm
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %a0, <8 x i64> %a1, <8 x double> undef, i8 -1)
  ret <8 x double> %2
}
declare <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double>, <8 x i64>, <8 x double>, i8) nounwind readnone

define <8 x double> @stack_fold_permilpdvar_zmm_mask(<8 x double>* %passthru, <8 x double> %a0, <8 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permilpdvar_zmm_mask
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %a0, <8 x i64> %a1, <8 x double> undef, i8 -1)
  %3 = bitcast i8 %mask to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x double>, <8 x double>* %passthru
  %5 = select <8 x i1> %3, <8 x double> %2, <8 x double> %4
  ret <8 x double> %5
}

define <8 x double> @stack_fold_permilpdvar_zmm_maskz(<8 x double> %a0, <8 x i64> %a1, i8 %mask) {
  ;CHECK-LABEL: stack_fold_permilpdvar_zmm_maskz
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x double> @llvm.x86.avx512.mask.vpermilvar.pd.512(<8 x double> %a0, <8 x i64> %a1, <8 x double> undef, i8 -1)
  %3 = bitcast i8 %mask to <8 x i1>
  %4 = select <8 x i1> %3, <8 x double> %2, <8 x double> zeroinitializer
  ret <8 x double> %4
}

define <16 x float> @stack_fold_permilps_zmm(<16 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps_zmm
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x float> %a0, <16 x float> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  ret <16 x float> %2
}

define <16 x float> @stack_fold_permilps_zmm_mask(<16 x float>* %passthru, <16 x float> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_permilps_zmm_mask
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x float> %a0, <16 x float> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  %3 = bitcast i16 %mask to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x float>, <16 x float>* %passthru
  %5 = select <16 x i1> %3, <16 x float> %2, <16 x float> %4
  ret <16 x float> %5
}

define <16 x float> @stack_fold_permilps_zmm_maskz(<16 x float> %a0, i16 %mask) {
  ;CHECK-LABEL: stack_fold_permilps_zmm_maskz
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = shufflevector <16 x float> %a0, <16 x float> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x float> %2, <16 x float> zeroinitializer
  ret <16 x float> %4
}

define <16 x float> @stack_fold_permilpsvar_zmm(<16 x float> %a0, <16 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permilpsvar_zmm
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %a0, <16 x i32> %a1, <16 x float> undef, i16 -1)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float>, <16 x i32>, <16 x float>, i16) nounwind readnone

define <16 x float> @stack_fold_permilpsvar_zmm_mask(<16 x float>* %passthru, <16 x float> %a0, <16 x i32> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_permilpsvar_zmm_mask
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %a0, <16 x i32> %a1, <16 x float> undef, i16 -1)
  %3 = bitcast i16 %mask to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x float>, <16 x float>* %passthru
  %5 = select <16 x i1> %3, <16 x float> %2, <16 x float> %4
  ret <16 x float> %5
}

define <16 x float> @stack_fold_permilpsvar_zmm_maskz(<16 x float> %a0, <16 x i32> %a1, i16 %mask) {
  ;CHECK-LABEL: stack_fold_permilpsvar_zmm_maskz
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%zmm[0-9][0-9]*}}, {{%zmm[0-9][0-9]*}} {{{%k[0-7]}}} {z} {{.*#+}} 64-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float> %a0, <16 x i32> %a1, <16 x float> undef, i16 -1)
  %3 = bitcast i16 %mask to <16 x i1>
  %4 = select <16 x i1> %3, <16 x float> %2, <16 x float> zeroinitializer
  ret <16 x float> %4
}

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
