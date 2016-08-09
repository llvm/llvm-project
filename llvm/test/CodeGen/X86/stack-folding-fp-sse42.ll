; RUN: llc -O3 -verify-machineinstrs -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+sse4.2 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x double> @stack_fold_addpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd
  ;CHECK:       addpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x float> @stack_fold_addps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps
  ;CHECK:       addps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <4 x float> %a0, %a1
  ret <4 x float> %2
}

define double @stack_fold_addsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_addsd
  ;CHECK:       addsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_addsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsd_int
  ;CHECK:       addsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.add.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_addss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_addss
  ;CHECK:       addss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_addss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addss_int
  ;CHECK:       addss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.add.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.add.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_addsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsubpd
  ;CHECK:       addsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x float> @stack_fold_addsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addsubps
  ;CHECK:       addsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_andnpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd
  ;CHECK:       andnpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
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

define <4 x float> @stack_fold_andnps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps
  ;CHECK:       andnps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
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

define <2 x double> @stack_fold_andpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd
  ;CHECK:       andpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x float> @stack_fold_andps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps
  ;CHECK:       andps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

define <2 x double> @stack_fold_blendpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_blendpd
  ;CHECK:       blendpd $2, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <2 x i1> <i1 1, i1 0>, <2 x double> %a0, <2 x double> %a1
  ret <2 x double> %2
}

define <4 x float> @stack_fold_blendps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_blendps
  ;CHECK:       blendps $6, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <4 x i1> <i1 1, i1 0, i1 0, i1 1>, <4 x float> %a0, <4 x float> %a1
  ret <4 x float> %2
}

define <2 x double> @stack_fold_blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %c) {
  ;CHECK-LABEL: stack_fold_blendvpd
  ;CHECK:       blendvpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a1, <2 x double> %c, <2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <4 x float> @stack_fold_blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %c) {
  ;CHECK-LABEL: stack_fold_blendvps
  ;CHECK:       blendvps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a1, <4 x float> %c, <4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_cmppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd
  ;CHECK:       cmpeqpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x float> @stack_fold_cmpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps
  ;CHECK:       cmpeqps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

define i32 @stack_fold_cmpsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd
  ;CHECK:       cmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq double %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <2 x double> @stack_fold_cmpsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd_int
  ;CHECK:       cmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone

define i32 @stack_fold_cmpss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_cmpss
  ;CHECK:       cmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq float %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <4 x float> @stack_fold_cmpss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpss_int
  ;CHECK:       cmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone

; TODO stack_fold_comisd

define i32 @stack_fold_comisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_comisd_int
  ;CHECK:       comisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.comieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.comieq.sd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_comiss

define i32 @stack_fold_comiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_comiss_int
  ;CHECK:       comiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.comieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.comieq.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_cvtdq2pd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd
  ;CHECK:       cvtdq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sitofp <2 x i32> %2 to <2 x double>
  ret <2 x double> %3
}

define <2 x double> @stack_fold_cvtdq2pd_int(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2pd_int
  ;CHECK:       cvtdq2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32>) nounwind readnone

define <4 x float> @stack_fold_cvtdq2ps(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps
  ;CHECK:       cvtdq2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <4 x i32> %a0 to <4 x float>
  ret <4 x float> %2
}

define <4 x i32> @stack_fold_cvtpd2dq(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2dq
  ;CHECK:       cvtpd2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double>) nounwind readnone

define <2 x float> @stack_fold_cvtpd2ps(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:       cvtpd2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc <2 x double> %a0 to <2 x float>
  ret <2 x float> %2
}

define <4 x i32> @stack_fold_cvtps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2dq
  ;CHECK:       cvtps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_cvtps2pd(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd
  ;CHECK:       cvtps2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %3 = fpext <2 x float> %2 to <2 x double>
  ret <2 x double> %3
}

define <2 x double> @stack_fold_cvtps2pd_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pd_int
  ;CHECK:       cvtps2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtps2pd(<4 x float> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtps2pd(<4 x float>) nounwind readnone

; TODO stack_fold_cvtsd2si

define i32 @stack_fold_cvtsd2si_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtsd2si_int
  ;CHECK:       cvtsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone

; TODO stack_fold_cvtsd2si64

define i64 @stack_fold_cvtsd2si64_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtsd2si64_int
  ;CHECK:       cvtsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

define float @stack_fold_cvtsd2ss(double %a0) minsize {
  ;CHECK-LABEL: stack_fold_cvtsd2ss
  ;CHECK:       cvtsd2ss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc double %a0 to float
  ret float %2
}

define <4 x float> @stack_fold_cvtsd2ss_int(<2 x double> %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsd2ss_int
  ;CHECK:       cvtsd2ss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float> <float 0x0, float 0x0, float 0x0, float 0x0>, <2 x double> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float>, <2 x double>) nounwind readnone

define double @stack_fold_cvtsi2sd(i32 %a0) minsize {
  ;CHECK-LABEL: stack_fold_cvtsi2sd
  ;CHECK:       cvtsi2sdl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to double
  ret double %2
}

define <2 x double> @stack_fold_cvtsi2sd_int(i32 %a0) {
  ;CHECK-LABEL: stack_fold_cvtsi2sd_int
  ;CHECK:       cvtsi2sdl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double> <double 0x0, double 0x0>, i32 %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double>, i32) nounwind readnone

define double @stack_fold_cvtsi642sd(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642sd
  ;CHECK:       cvtsi2sdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to double
  ret double %2
}

define <2 x double> @stack_fold_cvtsi642sd_int(i64 %a0) {
  ;CHECK-LABEL: stack_fold_cvtsi642sd_int
  ;CHECK:       cvtsi2sdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double> <double 0x0, double 0x0>, i64 %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone

define float @stack_fold_cvtsi2ss(i32 %a0) minsize {
  ;CHECK-LABEL: stack_fold_cvtsi2ss
  ;CHECK:       cvtsi2ssl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i32 %a0 to float
  ret float %2
}

define <4 x float> @stack_fold_cvtsi2ss_int(i32 %a0) {
  ;CHECK-LABEL: stack_fold_cvtsi2ss_int
  ;CHECK:  cvtsi2ssl {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float> <float 0x0, float 0x0, float 0x0, float 0x0>, i32 %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float>, i32) nounwind readnone

define float @stack_fold_cvtsi642ss(i64 %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtsi642ss
  ;CHECK:       cvtsi2ssq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = sitofp i64 %a0 to float
  ret float %2
}

define <4 x float> @stack_fold_cvtsi642ss_int(i64 %a0) {
  ;CHECK-LABEL: stack_fold_cvtsi642ss_int
  ;CHECK:  cvtsi2ssq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float> <float 0x0, float 0x0, float 0x0, float 0x0>, i64 %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone

define double @stack_fold_cvtss2sd(float %a0) minsize {
  ;CHECK-LABEL: stack_fold_cvtss2sd
  ;CHECK:       cvtss2sd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fpext float %a0 to double
  ret double %2
}

define <2 x double> @stack_fold_cvtss2sd_int(<4 x float> %a0) optsize {
  ;CHECK-LABEL: stack_fold_cvtss2sd_int
  ;CHECK:       cvtss2sd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double> <double 0x0, double 0x0>, <4 x float> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double>, <4 x float>) nounwind readnone

; TODO stack_fold_cvtss2si

define i32 @stack_fold_cvtss2si_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtss2si_int
  ;CHECK:       cvtss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.cvtss2si(<4 x float> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone

; TODO stack_fold_cvtss2si64

define i64 @stack_fold_cvtss2si64_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtss2si64_int
  ;CHECK:       cvtss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone

define <4 x i32> @stack_fold_cvttpd2dq(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttpd2dq
  ;CHECK:       cvttpd2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double>) nounwind readnone

define <4 x i32> @stack_fold_cvttps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2dq
  ;CHECK:       cvttps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <4 x float> %a0 to <4 x i32>
  ret <4 x i32> %2
}

define i32 @stack_fold_cvttsd2si(double %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si
  ;CHECK:       cvttsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi double %a0 to i32
  ret i32 %2
}

define i32 @stack_fold_cvttsd2si_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si_int
  ;CHECK:       cvttsd2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone

define i64 @stack_fold_cvttsd2si64(double %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si64
  ;CHECK:       cvttsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi double %a0 to i64
  ret i64 %2
}

define i64 @stack_fold_cvttsd2si64_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttsd2si64_int
  ;CHECK:       cvttsd2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone

define i32 @stack_fold_cvttss2si(float %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si
  ;CHECK:       cvttss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi float %a0 to i32
  ret i32 %2
}

define i32 @stack_fold_cvttss2si_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si_int
  ;CHECK:       cvttss2si {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.cvttss2si(<4 x float> %a0)
  ret i32 %2
}
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone

define i64 @stack_fold_cvttss2si64(float %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si64
  ;CHECK:       cvttss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi float %a0 to i64
  ret i64 %2
}

define i64 @stack_fold_cvttss2si64_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttss2si64_int
  ;CHECK:       cvttss2si {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %a0)
  ret i64 %2
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_divpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd
  ;CHECK:       divpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x float> @stack_fold_divps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps
  ;CHECK:       divps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <4 x float> %a0, %a1
  ret <4 x float> %2
}

define double @stack_fold_divsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_divsd
  ;CHECK:       divsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_divsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divsd_int
  ;CHECK:       divsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.div.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.div.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_divss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_divss
  ;CHECK:       divss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_divss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divss_int
  ;CHECK:       divss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.div.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.div.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_dppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_dppd
  ;CHECK:       dppd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.dppd(<2 x double> %a0, <2 x double> %a1, i8 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.dppd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x float> @stack_fold_dpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_dpps
  ;CHECK:       dpps $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.dpps(<4 x float> %a0, <4 x float> %a1, i8 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.dpps(<4 x float>, <4 x float>, i8) nounwind readnone

define i32 @stack_fold_extractps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_extractps
  ;CHECK:       extractps $1, {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp) {{.*#+}} 4-byte Folded Spill
  ;CHECK:       movl    {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Reload
  %1 = extractelement <4 x float> %a0, i32 1
  %2 = bitcast float %1 to i32
  %3 = tail call <2 x i64> asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  ret i32 %2
}

define <2 x double> @stack_fold_haddpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_haddpd
  ;CHECK:       haddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x float> @stack_fold_haddps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_haddps
  ;CHECK:       haddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_hsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_hsubpd
  ;CHECK:       hsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x float> @stack_fold_hsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_hsubps
  ;CHECK:       hsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_insertps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_insertps
  ;CHECK:       insertps $17, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  ;CHECK-NEXT:                                                        {{.*#+}} xmm0 = zero,mem[0],xmm0[2,3]
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a0, <4 x float> %a1, i8 209)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i8) nounwind readnone

define <2 x double> @stack_fold_maxpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxpd
  ;CHECK:       maxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone

define <2 x double> @stack_fold_maxpd_commutable(<2 x double> %a0, <2 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxpd_commutable
  ;CHECK:       maxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}

define <4 x float> @stack_fold_maxps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxps
  ;CHECK:       maxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_maxps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxps_commutable
  ;CHECK:       maxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define double @stack_fold_maxsd(double %a0, double %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsd
  ;CHECK:       maxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define double @stack_fold_maxsd_commutable(double %a0, double %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxsd_commutable
  ;CHECK:       maxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define <2 x double> @stack_fold_maxsd_int(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxsd_int
  ;CHECK:       maxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_maxss(float %a0, float %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxss
  ;CHECK:       maxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define float @stack_fold_maxss_commutable(float %a0, float %a1) #1 {
  ;CHECK-LABEL: stack_fold_maxss_commutable
  ;CHECK:       maxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ogt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define <4 x float> @stack_fold_maxss_int(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_maxss_int
  ;CHECK:       maxss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_minpd(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minpd
  ;CHECK:       minpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <2 x double> @stack_fold_minpd_commutable(<2 x double> %a0, <2 x double> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minpd_commutable
  ;CHECK:       minpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}

define <4 x float> @stack_fold_minps(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minps
  ;CHECK:       minps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @stack_fold_minps_commutable(<4 x float> %a0, <4 x float> %a1) #1 {
  ;CHECK-LABEL: stack_fold_minps_commutable
  ;CHECK:       minps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}

define double @stack_fold_minsd(double %a0, double %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsd
  ;CHECK:       minsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define double @stack_fold_minsd_commutable(double %a0, double %a1) #1 {
  ;CHECK-LABEL: stack_fold_minsd_commutable
  ;CHECK:       minsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt double %a0, %a1
  %3 = select i1 %2, double %a0, double %a1
  ret double %3
}

define <2 x double> @stack_fold_minsd_int(<2 x double> %a0, <2 x double> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minsd_int
  ;CHECK:       minsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_minss(float %a0, float %a1) #0 {
  ;CHECK-LABEL: stack_fold_minss
  ;CHECK:       minss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define float @stack_fold_minss_commutable(float %a0, float %a1) #1 {
  ;CHECK-LABEL: stack_fold_minss_commutable
  ;CHECK:       minss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp olt float %a0, %a1
  %3 = select i1 %2, float %a0, float %a1
  ret float %3
}

define <4 x float> @stack_fold_minss_int(<4 x float> %a0, <4 x float> %a1) #0 {
  ;CHECK-LABEL: stack_fold_minss_int
  ;CHECK:       minss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_movddup(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_movddup
  ;CHECK:   movddup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %2
}
; TODO stack_fold_movhpd (load / store)
; TODO stack_fold_movhps (load / store)

; TODO stack_fold_movlpd (load / store)
; TODO stack_fold_movlps (load / store)

define <4 x float> @stack_fold_movshdup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movshdup
  ;CHECK:       movshdup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x float> %2
}

define <4 x float> @stack_fold_movsldup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movsldup
  ;CHECK:       movsldup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x float> %2
}

define <2 x double> @stack_fold_mulpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd
  ;CHECK:       mulpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x float> @stack_fold_mulps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps
  ;CHECK:       mulps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <4 x float> %a0, %a1
  ret <4 x float> %2
}

define double @stack_fold_mulsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_mulsd
  ;CHECK:       mulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_mulsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulsd_int
  ;CHECK:       mulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.mul.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.mul.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_mulss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_mulss
  ;CHECK:       mulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_mulss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulss_int
  ;CHECK:       mulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.mul.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_orpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_orpd
  ;CHECK:       orpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x float> @stack_fold_orps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps
  ;CHECK:       orps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

; TODO stack_fold_rcpps

define <4 x float> @stack_fold_rcpps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rcpps_int
  ;CHECK:       rcpps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rcpss
; TODO stack_fold_rcpss_int

define <2 x double> @stack_fold_roundpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_roundpd
  ;CHECK:       roundpd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %a0, i32 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone

define <4 x float> @stack_fold_roundps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_roundps
  ;CHECK:       roundps $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %a0, i32 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone

define double @stack_fold_roundsd(double %a0) optsize {
  ;CHECK-LABEL: stack_fold_roundsd
  ;CHECK:       roundsd $9, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call double @llvm.floor.f64(double %a0)
  ret double %2
}
declare double @llvm.floor.f64(double) nounwind readnone

define <2 x double> @stack_fold_roundsd_int(<2 x double> %a0, <2 x double> %a1) optsize {
  ;CHECK-LABEL: stack_fold_roundsd_int
  ;CHECK:       roundsd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %a0, <2 x double> %a1, i32 7)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define float @stack_fold_roundss(float %a0) minsize {
  ;CHECK-LABEL: stack_fold_roundss
  ;CHECK:       roundss $9, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call float @llvm.floor.f32(float %a0)
  ret float %2
}
declare float @llvm.floor.f32(float) nounwind readnone

define <4 x float> @stack_fold_roundss_int(<4 x float> %a0, <4 x float> %a1) optsize {
  ;CHECK-LABEL: stack_fold_roundss_int
  ;CHECK:       roundss $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %a0, <4 x float> %a1, i32 7)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

; TODO stack_fold_rsqrtps

define <4 x float> @stack_fold_rsqrtps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtps_int
  ;CHECK:       rsqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rsqrtss
; TODO stack_fold_rsqrtss_int

define <2 x double> @stack_fold_shufpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_shufpd
  ;CHECK:       shufpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %2
}

define <4 x float> @stack_fold_shufps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps
  ;CHECK:       shufps $200, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  ret <4 x float> %2
}

define <2 x double> @stack_fold_sqrtpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtpd
  ;CHECK:       sqrtpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <4 x float> @stack_fold_sqrtps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtps
  ;CHECK:       sqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define double @stack_fold_sqrtsd(double %a0) optsize {
  ;CHECK-LABEL: stack_fold_sqrtsd
  ;CHECK:       sqrtsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call double @llvm.sqrt.f64(double %a0)
  ret double %2
}
declare double @llvm.sqrt.f64(double) nounwind readnone

; TODO stack_fold_sqrtsd_int
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define float @stack_fold_sqrtss(float %a0) minsize {
  ;CHECK-LABEL: stack_fold_sqrtss
  ;CHECK:       sqrtss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call float @llvm.sqrt.f32(float %a0)
  ret float %2
}
declare float @llvm.sqrt.f32(float) nounwind readnone

; TODO stack_fold_sqrtss_int
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_subpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd
  ;CHECK:       subpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x float> @stack_fold_subps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps
  ;CHECK:       subps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <4 x float> %a0, %a1
  ret <4 x float> %2
}

define double @stack_fold_subsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_subsd
  ;CHECK:       subsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub double %a0, %a1
  ret double %2
}

define <2 x double> @stack_fold_subsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subsd_int
  ;CHECK:       subsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.sub.sd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.sub.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_subss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_subss
  ;CHECK:       subss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub float %a0, %a1
  ret float %2
}

define <4 x float> @stack_fold_subss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subss_int
  ;CHECK:       subss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.sub.ss(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>) nounwind readnone

define i32 @stack_fold_ucomisd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd
  ;CHECK:       ucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq double %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd_int
  ;CHECK:       ucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.ucomieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.ucomieq.sd(<2 x double>, <2 x double>) nounwind readnone

define i32 @stack_fold_ucomiss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss
  ;CHECK:       ucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq float %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss_int
  ;CHECK:       ucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.ucomieq.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_unpckhpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhpd
  ;CHECK:       unpckhpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 3>
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x float> @stack_fold_unpckhps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhps
  ;CHECK:       unpckhps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ; fadd forces execution domain
  %3 = fadd <4 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %3
}

define <2 x double> @stack_fold_unpcklpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklpd
  ;CHECK:       unpcklpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 0, i32 2>
  ; fadd forces execution domain
  %3 = fadd <2 x double> %2, <double 0x0, double 0x0>
  ret <2 x double> %3
}

define <4 x float> @stack_fold_unpcklps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklps
  ;CHECK:       unpcklps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ; fadd forces execution domain
  %3 = fadd <4 x float> %2, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %3
}

define <2 x double> @stack_fold_xorpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_xorpd
  ;CHECK:       xorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x float> @stack_fold_xorps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps
  ;CHECK:       xorps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ; fadd forces execution domain
  %6 = fadd <4 x float> %5, <float 0x0, float 0x0, float 0x0, float 0x0>
  ret <4 x float> %6
}

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
