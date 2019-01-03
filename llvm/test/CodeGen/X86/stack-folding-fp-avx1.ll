; RUN: llc -O3 -verify-machineinstrs -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx,+f16c < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x double> @stack_fold_addpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_addpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd_ymm
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_addps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_addps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps_ymm
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_addsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_addsd
  ;CHECK:       vaddsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_addsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsd_int
  ;CHECK:       vaddsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fadd double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}
declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_addss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_addss
  ;CHECK:       vaddss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_addss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addss_int
  ;CHECK:       vaddss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fadd float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}
declare <4 x float> @llvm.x86.sse.add.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_addsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsubpd
  ;CHECK:       vaddsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_addsubpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsubpd_ymm
  ;CHECK:       vaddsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_addsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addsubps
  ;CHECK:       vaddsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_addsubps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addsubps_ymm
  ;CHECK:       vaddsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_andnpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_andps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps_ymm
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = and <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

define <2 x double> @stack_fold_blendpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_blendpd
  ;CHECK:       vblendpd $2, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <2 x i1> <i1 1, i1 0>, <2 x double> %a0, <2 x double> %a1
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_blendpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_blendpd_ymm
  ;CHECK:       vblendpd $6, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <4 x i1> <i1 1, i1 0, i1 0, i1 1>, <4 x double> %a0, <4 x double> %a1
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %3}

define <4 x float> @stack_fold_blendps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_blendps
  ;CHECK:       vblendps $6, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <4 x i1> <i1 1, i1 0, i1 0, i1 1>, <4 x float> %a0, <4 x float> %a1
  ; fadd forces execution domain
  %3 = fadd <4 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %3
}

define <8 x float> @stack_fold_blendps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_blendps_ymm
  ;CHECK:       vblendps $102, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <8 x i1> <i1 1, i1 0, i1 0, i1 1, i1 1, i1 0, i1 0, i1 1>, <8 x float> %a0, <8 x float> %a1
  ; fadd forces execution domain
  %3 = fadd <8 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %3
}

define <2 x double> @stack_fold_blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %c) {
  ;CHECK-LABEL: stack_fold_blendvpd
  ;CHECK:       vblendvpd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a1, <2 x double> %c, <2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_blendvpd_ymm(<4 x double> %a0, <4 x double> %a1, <4 x double> %c) {
  ;CHECK-LABEL: stack_fold_blendvpd_ymm
  ;CHECK:       vblendvpd {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a1, <4 x double> %c, <4 x double> %a0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %c) {
  ;CHECK-LABEL: stack_fold_blendvps
  ;CHECK:       vblendvps {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a1, <4 x float> %c, <4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_blendvps_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %c) {
  ;CHECK-LABEL: stack_fold_blendvps_ymm
  ;CHECK:       vblendvps {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a1, <8 x float> %c, <8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_cmppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @stack_fold_cmppd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd_ymm
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %a0, <4 x double> %a1, i8 0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone

define <4 x float> @stack_fold_cmpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @stack_fold_cmpps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps_ymm
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a1, i8 0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

define i32 @stack_fold_cmpsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd
  ;CHECK:       vcmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq double %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <2 x double> @stack_fold_cmpsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd_int
  ;CHECK:       vcmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone

define i32 @stack_fold_cmpss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_cmpss
  ;CHECK:       vcmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq float %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <4 x float> @stack_fold_cmpss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpss_int
  ;CHECK:       vcmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone

; TODO stack_fold_comisd

define i32 @stack_fold_comisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_comisd_int
  ;CHECK:       vcomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.comieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.comieq.sd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_comiss

define i32 @stack_fold_comiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_comiss_int
  ;CHECK:       vcomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.comieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.comieq.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_cvtdq2pd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sitofp <2 x i32> %2 to <2 x double>
  ret <2 x double> %3
}
define <2 x double> @stack_fold_cvtdq2pd_int(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd_int
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> %a0, <2 x i32> <i32 0, i32 1>
  %cvt = sitofp <2 x i32> %2 to <2 x double>
  ret <2 x double> %cvt
}

define <4 x double> @stack_fold_cvtdq2pd_ymm(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd_ymm
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <4 x i32> %a0 to <4 x double>
  ret <4 x double> %2
}

define <4 x double> @stack_fold_cvtdq2pd_ymm_int(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd_ymm_int
  ;CHECK:   vcvtdq2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %cvt = sitofp <4 x i32> %a0 to <4 x double>
  ret <4 x double> %cvt
}

define <4 x float> @stack_fold_cvtdq2ps(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <4 x i32> %a0 to <4 x float>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_cvtdq2ps_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps_ymm
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <8 x i32> %a0 to <8 x float>
  ret <8 x float> %2
}

define <4 x i32> @stack_fold_cvtpd2dq(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2dq
  ;CHECK:   vcvtpd2dqx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double>) nounwind readnone

define <4 x i32> @stack_fold_cvtpd2dq_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2dq_ymm
  ;CHECK:   vcvtpd2dqy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double>) nounwind readnone

define <2 x float> @stack_fold_cvtpd2ps(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2psx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc <2 x double> %a0 to <2 x float>
  ret <2 x float> %2
}

define <4 x float> @stack_fold_cvtpd2ps_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps_ymm
  ;CHECK:   vcvtpd2psy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc <4 x double> %a0 to <4 x float>
  ret <4 x float> %2
}

define <4 x float> @stack_fold_cvtph2ps(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_cvtph2ps
  ;CHECK:   vcvtph2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16>) nounwind readonly

define <8 x float> @stack_fold_cvtph2ps_ymm(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_cvtph2ps_ymm
  ;CHECK:   vcvtph2ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readonly

define <4 x i32> @stack_fold_cvtps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2dq
  ;CHECK:  vcvtps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float>) nounwind readnone

define <8 x i32> @stack_fold_cvtps2dq_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2dq_ymm
  ;CHECK:  vcvtps2dq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float> %a0)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float>) nounwind readnone

define <2 x double> @stack_fold_cvtps2pd(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd
  ;CHECK:   vcvtps2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %3 = fpext <2 x float> %2 to <2 x double>
  ret <2 x double> %3
}

define <2 x double> @stack_fold_cvtps2pd_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd_int
  ;CHECK:   vcvtps2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a0, <2 x i32> <i32 0, i32 1>
  %cvtps2pd = fpext <2 x float> %2 to <2 x double>
  ret <2 x double> %cvtps2pd
}

define <4 x double> @stack_fold_cvtps2pd_ymm(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd_ymm
  ;CHECK:   vcvtps2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fpext <4 x float> %a0 to <4 x double>
  ret <4 x double> %2
}

define <4 x double> @stack_fold_cvtps2pd_ymm_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd_ymm_int
  ;CHECK:   vcvtps2pd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %cvtps2pd = fpext <4 x float> %a0 to <4 x double>
  ret <4 x double> %cvtps2pd
}

define <8 x i16> @stack_fold_cvtps2ph_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2ph_ymm
  ;CHECK:   vcvtps2ph $0, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %a0, i32 0)
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  ret <8 x i16> %1
}
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readonly

; TODO stack_fold_cvtsd2si

define i32 @stack_fold_cvtsd2si_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtsd2si_int
  ;CHECK:  vcvtsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone

; TODO stack_fold_cvtsd2si64

define i64 @stack_fold_cvtsd2si64_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtsd2si64_int
  ;CHECK:  vcvtsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define double @stack_fold_cvtsi2sd(i32 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi2sd
  ;CHECK:  vcvtsi2sdl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to double
  ret double %2
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define <2 x double> @stack_fold_cvtsi2sd_int(i32 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi2sd_int
  ;CHECK:  vcvtsi2sdl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to double
  %3 = insertelement <2 x double> zeroinitializer, double %2, i64 0
  ret <2 x double> %3
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define double @stack_fold_cvtsi642sd(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642sd
  ;CHECK:  vcvtsi2sdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to double
  ret double %2
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define <2 x double> @stack_fold_cvtsi642sd_int(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642sd_int
  ;CHECK:  vcvtsi2sdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to double
  %3 = insertelement <2 x double> zeroinitializer, double %2, i64 0
  ret <2 x double> %3
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define float @stack_fold_cvtsi2ss(i32 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi2ss
  ;CHECK:  vcvtsi2ssl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to float
  ret float %2
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define <4 x float> @stack_fold_cvtsi2ss_int(i32 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi2ss_int
  ;CHECK:  vcvtsi2ssl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to float
  %3 = insertelement <4 x float> zeroinitializer, float %2, i64 0
  ret <4 x float> %3
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define float @stack_fold_cvtsi642ss(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642ss
  ;CHECK:  vcvtsi2ssq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to float
  ret float %2
}

; TODO: This fold shouldn't require optsize. Not folding doesn't prevent reading an undef register since the registers are a mix of XMM and GPR.
define <4 x float> @stack_fold_cvtsi642ss_int(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642ss_int
  ;CHECK:  vcvtsi2ssq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to float
  %3 = insertelement <4 x float> zeroinitializer, float %2, i64 0
  ret <4 x float> %3
}

; TODO stack_fold_cvtss2si

define i32 @stack_fold_cvtss2si_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtss2si_int
  ;CHECK:  vcvtss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.cvtss2si(<4 x float> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone

; TODO stack_fold_cvtss2si64

define i64 @stack_fold_cvtss2si64_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtss2si64_int
  ;CHECK:  vcvtss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone

define <4 x i32> @stack_fold_cvttpd2dq(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttpd2dq
  ;CHECK:  vcvttpd2dqx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double>) nounwind readnone

define <4 x i32> @stack_fold_cvttpd2dq_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttpd2dq_ymm
  ;CHECK:  vcvttpd2dqy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <4 x double> %a0 to <4 x i32>
  ret <4 x i32> %2
}

define <4 x i32> @stack_fold_cvttps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2dq
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <4 x float> %a0 to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i32> @stack_fold_cvttps2dq_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2dq_ymm
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <8 x float> %a0 to <8 x i32>
  ret <8 x i32> %2
}

define i32 @stack_fold_cvttsd2si(double %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si
  ;CHECK:  vcvttsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi double %a0 to i32
  ret i32 %2
}

define i32 @stack_fold_cvttsd2si_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si_int
  ;CHECK:  vcvttsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone

define i64 @stack_fold_cvttsd2si64(double %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si64
  ;CHECK:  vcvttsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi double %a0 to i64
  ret i64 %2
}

define i64 @stack_fold_cvttsd2si64_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si64_int
  ;CHECK:  vcvttsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone

define i32 @stack_fold_cvttss2si(float %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si
  ;CHECK:  vcvttss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi float %a0 to i32
  ret i32 %2
}

define i32 @stack_fold_cvttss2si_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si_int
  ;CHECK:  vcvttss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.cvttss2si(<4 x float> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone

define i64 @stack_fold_cvttss2si64(float %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si64
  ;CHECK:  vcvttss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi float %a0 to i64
  ret i64 %2
}

define i64 @stack_fold_cvttss2si64_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si64_int
  ;CHECK:  vcvttss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_divpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_divpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd_ymm
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_divps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_divps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps_ymm
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_divsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_divsd
  ;CHECK:       vdivsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_divsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divsd_int
  ;CHECK:       vdivsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fdiv double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_divss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_divss
  ;CHECK:       vdivss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_divss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divss_int
  ;CHECK:       vdivss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fdiv float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <2 x double> @stack_fold_dppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_dppd
  ;CHECK:       vdppd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.dppd(<2 x double> %a0, <2 x double> %a1, i8 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.dppd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x float> @stack_fold_dpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_dpps
  ;CHECK:       vdpps $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.dpps(<4 x float> %a0, <4 x float> %a1, i8 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.dpps(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @stack_fold_dpps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_dpps_ymm
  ;CHECK:       vdpps $7, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.dp.ps.256(<8 x float> %a0, <8 x float> %a1, i8 7)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.dp.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

define <4 x float> @stack_fold_extractf128(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_extractf128
  ;CHECK:       vextractf128 $1, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %1 = shufflevector <8 x float> %a0, <8 x float> %a1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %2 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  ret <4 x float> %1
}

define i32 @stack_fold_extractps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_extractps
  ;CHECK:       vextractps $1, {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 4-byte Folded Spill
  ;CHECK:       movl    {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Reload
  ; fadd forces execution domain
  %1 = fadd <4 x float> %a0, %a1
  %2 = extractelement <4 x float> %1, i32 1
  %3 = bitcast float %2 to i32
  %4 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret i32 %3
}

define <2 x double> @stack_fold_haddpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_haddpd
  ;CHECK:       vhaddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_haddpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_haddpd_ymm
  ;CHECK:       vhaddpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_haddps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_haddps
  ;CHECK:       vhaddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_haddps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_haddps_ymm
  ;CHECK:       vhaddps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_hsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_hsubpd
  ;CHECK:       vhsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_hsubpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_hsubpd_ymm
  ;CHECK:       vhsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_hsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_hsubps
  ;CHECK:       vhsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_hsubps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_hsubps_ymm
  ;CHECK:       vhsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <8 x float> @stack_fold_insertf128(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertf128
  ;CHECK:       vinsertf128 $1, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %2
}

define <4 x float> @stack_fold_insertps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertps
  ;CHECK:       vinsertps $17, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  ;CHECK-NEXT:                                                                              {{.*#+}} xmm0 = zero,mem[0],xmm0[2,3]
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a0, <4 x float> %a1, i8 209)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i8) nounwind readnone

define <2 x double> @stack_fold_maxpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone

define <2 x double> @stack_fold_maxpd_commutable(<2 x double> %a0, <2 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_commutable
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}

define <4 x double> @stack_fold_maxpd_ymm(<4 x double> %a0, <4 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd_ymm
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x double> @stack_fold_maxpd_ymm_commutable(<4 x double> %a0, <4 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_ymm_commutable
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}

define <4 x float> @stack_fold_maxps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_maxps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_commutable
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define <8 x float> @stack_fold_maxps_ymm(<8 x float> %a0, <8 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps_ymm
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <8 x float> @stack_fold_maxps_ymm_commutable(<8 x float> %a0, <8 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_ymm_commutable
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}

define double @stack_fold_maxsd(double %a0, double %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsd
  ;CHECK:       vmaxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define double @stack_fold_maxsd_commutable(double %a0, double %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsd_commutable
  ;CHECK:       vmaxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define <2 x double> @stack_fold_maxsd_int(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsd_int
  ;CHECK:       vmaxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_maxss(float %a0, float %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxss
  ;CHECK:       vmaxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define float @stack_fold_maxss_commutable(float %a0, float %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxss_commutable
  ;CHECK:       vmaxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define <4 x float> @stack_fold_maxss_int(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxss_int
  ;CHECK:       vmaxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_minpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minpd
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <2 x double> @stack_fold_minpd_commutable(<2 x double> %a0, <2 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minpd_commutable
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}

define <4 x double> @stack_fold_minpd_ymm(<4 x double> %a0, <4 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minpd_ymm
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x double> @stack_fold_minpd_ymm_commutable(<4 x double> %a0, <4 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minpd_ymm_commutable
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}

define <4 x float> @stack_fold_minps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_minps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_commutable
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define <8 x float> @stack_fold_minps_ymm(<8 x float> %a0, <8 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps_ymm
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <8 x float> @stack_fold_minps_ymm_commutable(<8 x float> %a0, <8 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_ymm_commutable
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}

define double @stack_fold_minsd(double %a0, double %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsd
  ;CHECK:       vminsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define double @stack_fold_minsd_commutable(double %a0, double %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsd_commutable
  ;CHECK:       vminsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define <2 x double> @stack_fold_minsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_minsd_int
  ;CHECK:       vminsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_minss(float %a0, float %a1) #0 {
  ;CHECK-LABEL: stack_fold_minss
  ;CHECK:       vminss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define float @stack_fold_minss_commutable(float %a0, float %a1) #1 {
  ;CHECK-LABEL: stack_fold_minss_commutable
  ;CHECK:       vminss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define <4 x float> @stack_fold_minss_int(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minss_int
  ;CHECK:       vminss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_movddup(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_movddup
  ;CHECK:   vmovddup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_movddup_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_movddup_ymm
  ;CHECK:   vmovddup {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x double> %2
}

; TODO stack_fold_movhpd (load / store)
; TODO stack_fold_movhps (load / store)

; TODO stack_fold_movlpd (load / store)
; TODO stack_fold_movlps (load / store)

define <4 x float> @stack_fold_movshdup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movshdup
  ;CHECK:   vmovshdup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_movshdup_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movshdup_ymm
  ;CHECK:   vmovshdup {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %2
}

define <4 x float> @stack_fold_movsldup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movsldup
  ;CHECK:   vmovsldup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_movsldup_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movsldup_ymm
  ;CHECK:   vmovsldup {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_mulpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_mulpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd_ymm
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_mulps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_mulps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps_ymm
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_mulsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_mulsd
  ;CHECK:       vmulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_mulsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulsd_int
  ;CHECK:       vmulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fmul double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_mulss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_mulss
  ;CHECK:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_mulss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulss_int
  ;CHECK:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fmul float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define <2 x double> @stack_fold_orpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_orpd
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_orpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_orpd_ymm
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_orps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps_ymm
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = or <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

define <8 x float> @stack_fold_perm2f128(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_perm2f128
  ;CHECK:   vperm2f128 $33, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_permilpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd
  ;CHECK:   vpermilpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_permilpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd_ymm
  ;CHECK:   vpermilpd $5, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x double> %2
}

define <2 x double> @stack_fold_permilpdvar(<2 x double> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permilpdvar
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double> %a0, <2 x i64> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.avx.vpermilvar.pd(<2 x double>, <2 x i64>) nounwind readnone

define <4 x double> @stack_fold_permilpdvar_ymm(<4 x double> %a0, <4 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_permilpdvar_ymm
  ;CHECK:       vpermilpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double> %a0, <4 x i64> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.vpermilvar.pd.256(<4 x double>, <4 x i64>) nounwind readnone

define <4 x float> @stack_fold_permilps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_permilps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps_ymm
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x float> %2
}

define <4 x float> @stack_fold_permilpsvar(<4 x float> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permilpsvar
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %a0, <4 x i32> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float>, <4 x i32>) nounwind readnone

define <8 x float> @stack_fold_permilpsvar_ymm(<8 x float> %a0, <8 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_permilpsvar_ymm
  ;CHECK:       vpermilps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float> %a0, <8 x i32> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.vpermilvar.ps.256(<8 x float>, <8 x i32>) nounwind readnone

; TODO stack_fold_rcpps

define <4 x float> @stack_fold_rcpps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rcpps_int
  ;CHECK:       vrcpps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rcpps_ymm

define <8 x float> @stack_fold_rcpps_ymm_int(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rcpps_ymm_int
  ;CHECK:       vrcpps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

; TODO stack_fold_rcpss
; TODO stack_fold_rcpss_int

define <2 x double> @stack_fold_roundpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_roundpd
  ;CHECK:  vroundpd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %a0, i32 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone

define <4 x double> @stack_fold_roundpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_roundpd_ymm
  ;CHECK:  vroundpd $7, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %a0, i32 7)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone

define <4 x float> @stack_fold_roundps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_roundps
  ;CHECK:  vroundps $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %a0, i32 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone

define <8 x float> @stack_fold_roundps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_roundps_ymm
  ;CHECK:  vroundps $7, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %a0, i32 7)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone

define double @stack_fold_roundsd(double %a0) optsize {
  ;CHECK-LABEL: stack_fold_roundsd
  ;CHECK:       vroundsd $9, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call double @llvm.floor.f64(double %a0)
  ret double %2
}
declare double @llvm.floor.f64(double) nounwind readnone

define <2 x double> @stack_fold_roundsd_int(<2 x double> %a0, <2 x double> %a1) optsize {
  ;CHECK-LABEL: stack_fold_roundsd_int
  ;CHECK:       vroundsd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %a0, <2 x double> %a1, i32 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define float @stack_fold_roundss(float %a0) optsize {
  ;CHECK-LABEL: stack_fold_roundss
  ;CHECK:       vroundss $9, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call float @llvm.floor.f32(float %a0)
  ret float %2
}
declare float @llvm.floor.f32(float) nounwind readnone

define <4 x float> @stack_fold_roundss_int(<4 x float> %a0, <4 x float> %a1) optsize {
  ;CHECK-LABEL: stack_fold_roundss_int
  ;CHECK:       vroundss $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %a0, <4 x float> %a1, i32 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

; TODO stack_fold_rsqrtps

define <4 x float> @stack_fold_rsqrtps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtps_int
  ;CHECK:       vrsqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rsqrtps_ymm

define <8 x float> @stack_fold_rsqrtps_ymm_int(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtps_ymm_int
  ;CHECK:       vrsqrtps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone

; TODO stack_fold_rsqrtss
; TODO stack_fold_rsqrtss_int

define <2 x double> @stack_fold_shufpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_shufpd
  ;CHECK:       vshufpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 2>
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_shufpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_shufpd_ymm
  ;CHECK:       vshufpd $5, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 1, i32 4, i32 3, i32 6>
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %3
}

define <4 x float> @stack_fold_shufps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps
  ;CHECK:       vshufps $200, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_shufps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps_ymm
  ;CHECK:       vshufps $148, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 0, i32 1, i32 9, i32 10, i32 4, i32 5, i32 13, i32 14>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_sqrtpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtpd
  ;CHECK:       vsqrtpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)

define <4 x double> @stack_fold_sqrtpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtpd_ymm
  ;CHECK:       vsqrtpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %a0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)

define <4 x float> @stack_fold_sqrtps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtps
  ;CHECK:       vsqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <8 x float> @stack_fold_sqrtps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtps_ymm
  ;CHECK:       vsqrtps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>)

define double @stack_fold_sqrtsd(double %a0) optsize {
  ;CHECK-LABEL: stack_fold_sqrtsd
  ;CHECK:       vsqrtsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call double @llvm.sqrt.f64(double %a0)
  ret double %2
}
declare double @llvm.sqrt.f64(double) nounwind readnone

; TODO stack_fold_sqrtsd_int

define float @stack_fold_sqrtss(float %a0) optsize {
  ;CHECK-LABEL: stack_fold_sqrtss
  ;CHECK:       vsqrtss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call float @llvm.sqrt.f32(float %a0)
  ret float %2
}
declare float @llvm.sqrt.f32(float) nounwind readnone

; TODO stack_fold_sqrtss_int

define <2 x double> @stack_fold_subpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_subpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd_ymm
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_subps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_subps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps_ymm
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_subsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_subsd
  ;CHECK:       vsubsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_subsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subsd_int
  ;CHECK:       vsubsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <2 x double> %a0, i32 0
  %3 = extractelement <2 x double> %a1, i32 0
  %4 = fsub double %2, %3
  %5 = insertelement <2 x double> %a0, double %4, i32 0
  ret <2 x double> %5
}

define float @stack_fold_subss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_subss
  ;CHECK:       vsubss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_subss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subss_int
  ;CHECK:       vsubss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = extractelement <4 x float> %a0, i32 0
  %3 = extractelement <4 x float> %a1, i32 0
  %4 = fsub float %2, %3
  %5 = insertelement <4 x float> %a0, float %4, i32 0
  ret <4 x float> %5
}

define i32 @stack_fold_testpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_testpd
  ;CHECK:       vtestpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.avx.vtestc.pd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.avx.vtestc.pd(<2 x double>, <2 x double>) nounwind readnone

define i32 @stack_fold_testpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_testpd_ymm
  ;CHECK:       vtestpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.avx.vtestc.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.avx.vtestc.pd.256(<4 x double>, <4 x double>) nounwind readnone

define i32 @stack_fold_testps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_testps
  ;CHECK:       vtestps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.avx.vtestc.ps(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.avx.vtestc.ps(<4 x float>, <4 x float>) nounwind readnone

define i32 @stack_fold_testps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_testps_ymm
  ;CHECK:       vtestps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.avx.vtestc.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.avx.vtestc.ps.256(<8 x float>, <8 x float>) nounwind readnone

define i32 @stack_fold_ucomisd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd
  ;CHECK:       vucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq double %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd_int
  ;CHECK:       vucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.ucomieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.ucomieq.sd(<2 x double>, <2 x double>) nounwind readnone

define i32 @stack_fold_ucomiss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss
  ;CHECK:       vucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq float %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss_int
  ;CHECK:       vucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.ucomieq.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_unpckhpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhpd
  ;CHECK:       vunpckhpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 3>
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_unpckhpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhpd_ymm
  ;CHECK:       vunpckhpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %3
}

define <4 x float> @stack_fold_unpckhps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhps
  ;CHECK:       vunpckhps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ; fadd forces execution domain
  %3 = fadd <4 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %3
}

define <8 x float> @stack_fold_unpckhps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhps_ymm
  ;CHECK:       vunpckhps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ; fadd forces execution domain
  %3 = fadd <8 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %3
}

define <2 x double> @stack_fold_unpcklpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklpd
  ;CHECK:       vunpcklpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 0, i32 2>
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x double> @stack_fold_unpcklpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklpd_ymm
  ;CHECK:       vunpcklpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ; fadd forces execution domain
  %3 = fadd <4 x double> %2, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %3
}

define <4 x float> @stack_fold_unpcklps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklps
  ;CHECK:       vunpcklps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ; fadd forces execution domain
  %3 = fadd <4 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %3
}

define <8 x float> @stack_fold_unpcklps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklps_ymm
  ;CHECK:       vunpcklps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ; fadd forces execution domain
  %3 = fadd <8 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %3
}

define <2 x double> @stack_fold_xorpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_xorpd
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_xorpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_xorpd_ymm
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
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
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_xorps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps_ymm
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ; fadd forces execution domain
  %6 = fadd <8 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0>
  ret <8 x float> %6
}

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
