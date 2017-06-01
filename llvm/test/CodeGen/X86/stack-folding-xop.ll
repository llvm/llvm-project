; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx,+xop < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x double> @stack_fold_vfrczpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczpd
  ;CHECK:       vfrczpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double>) nounwind readnone

define <4 x double> @stack_fold_vfrczpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczpd_ymm
  ;CHECK:       vfrczpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double> %a0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double>) nounwind readnone

define <4 x float> @stack_fold_vfrczps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczps
  ;CHECK:       vfrczps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float>) nounwind readnone

define <8 x float> @stack_fold_vfrczps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczps_ymm
  ;CHECK:       vfrczps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float>) nounwind readnone

define <2 x double> @stack_fold_vfrczsd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczsd
  ;CHECK:       vfrczsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double>) nounwind readnone

define <4 x float> @stack_fold_vfrczss(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_vfrczss
  ;CHECK:       vfrczss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float>) nounwind readnone

define <2 x i64> @stack_fold_vpcmov_rm(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpcmov_rm
  ;CHECK:       vpcmov {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2)
  ret <2 x i64> %2
}
define <2 x i64> @stack_fold_vpcmov_mr(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpcmov_mr
  ;CHECK:       vpcmov {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64> %a0, <2 x i64> %a2, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64>, <2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @stack_fold_vpcmov_rm_ymm(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpcmov_rm_ymm
  ;CHECK:       vpcmov {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %a2)
  ret <4 x i64> %2
}
define <4 x i64> @stack_fold_vpcmov_mr_ymm(<4 x i64> %a0, <4 x i64> %a1, <4 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpcmov_mr_ymm
  ;CHECK:       vpcmov {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %a0, <4 x i64> %a2, <4 x i64> %a1)
  ret <4 x i64> %2
}
declare <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64>, <4 x i64>, <4 x i64>) nounwind readnone

define <16 x i8> @stack_fold_vpcomb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomb
  ;CHECK:       vpcomltb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %a0, <16 x i8> %a1, i8 0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8>, <16 x i8>, i8) nounwind readnone

define <4 x i32> @stack_fold_vpcomd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomd
  ;CHECK:       vpcomltd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %a0, <4 x i32> %a1, i8 0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <2 x i64> @stack_fold_vpcomq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomq
  ;CHECK:       vpcomltq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %a0, <2 x i64> %a1, i8 0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <16 x i8> @stack_fold_vpcomub(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomub
  ;CHECK:       vpcomltub {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %a0, <16 x i8> %a1, i8 0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8>, <16 x i8>, i8) nounwind readnone

define <4 x i32> @stack_fold_vpcomud(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomud
  ;CHECK:       vpcomltud {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %a0, <4 x i32> %a1, i8 0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <2 x i64> @stack_fold_vpcomuq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomuq
  ;CHECK:       vpcomltuq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %a0, <2 x i64> %a1, i8 0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <8 x i16> @stack_fold_vpcomuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomuw
  ;CHECK:       vpcomltuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %a0, <8 x i16> %a1, i8 0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16>, <8 x i16>, i8) nounwind readnone

define <8 x i16> @stack_fold_vpcomw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpcomw
  ;CHECK:       vpcomltw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %a0, <8 x i16> %a1, i8 0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16>, <8 x i16>, i8) nounwind readnone

define <2 x double> @stack_fold_vpermil2pd_rm(<2 x double> %a0, <2 x double> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2pd_rm
  ;CHECK:       vpermil2pd $0, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %a0, <2 x double> %a1, <2 x i64> %a2, i8 0)
  ret <2 x double> %2
}
define <2 x double> @stack_fold_vpermil2pd_mr(<2 x double> %a0, <2 x i64> %a1, <2 x double> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2pd_mr
  ;CHECK:       vpermil2pd $0, {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %a0, <2 x double> %a2, <2 x i64> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double>, <2 x double>, <2 x i64>, i8) nounwind readnone

define <4 x double> @stack_fold_vpermil2pd_rm_ymm(<4 x double> %a0, <4 x double> %a1, <4 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2pd_rm
  ;CHECK:       vpermil2pd $0, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %a0, <4 x double> %a1, <4 x i64> %a2, i8 0)
  ret <4 x double> %2
}
define <4 x double> @stack_fold_vpermil2pd_mr_ymm(<4 x double> %a0, <4 x i64> %a1, <4 x double> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2pd_mr
  ;CHECK:       vpermil2pd $0, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %a0, <4 x double> %a2, <4 x i64> %a1, i8 0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double>, <4 x double>, <4 x i64>, i8) nounwind readnone

define <4 x float> @stack_fold_vpermil2ps_rm(<4 x float> %a0, <4 x float> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2ps_rm
  ;CHECK:       vpermil2ps $0, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float> %a0, <4 x float> %a1, <4 x i32> %a2, i8 0)
  ret <4 x float> %2
}
define <4 x float> @stack_fold_vpermil2ps_mr(<4 x float> %a0, <4 x i32> %a1, <4 x float> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2ps_mr
  ;CHECK:       vpermil2ps $0, {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float> %a0, <4 x float> %a2, <4 x i32> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float>, <4 x float>, <4 x i32>, i8) nounwind readnone

define <8 x float> @stack_fold_vpermil2ps_rm_ymm(<8 x float> %a0, <8 x float> %a1, <8 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2ps_rm
  ;CHECK:       vpermil2ps $0, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float> %a0, <8 x float> %a1, <8 x i32> %a2, i8 0)
  ret <8 x float> %2
}
define <8 x float> @stack_fold_vpermil2ps_mr_ymm(<8 x float> %a0, <8 x i32> %a1, <8 x float> %a2) {
  ;CHECK-LABEL: stack_fold_vpermil2ps_mr
  ;CHECK:       vpermil2ps $0, {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float> %a0, <8 x float> %a2, <8 x i32> %a1, i8 0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float>, <8 x float>, <8 x i32>, i8) nounwind readnone

define <4 x i32> @stack_fold_vphaddbd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddbd
  ;CHECK:       vphaddbd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vphaddbd(<16 x i8> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vphaddbd(<16 x i8>) nounwind readnone

define <2 x i64> @stack_fold_vphaddbq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddbq
  ;CHECK:       vphaddbq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphaddbq(<16 x i8> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphaddbq(<16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_vphaddbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddbw
  ;CHECK:       vphaddbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vphaddbw(<16 x i8> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vphaddbw(<16 x i8>) nounwind readnone

define <2 x i64> @stack_fold_vphadddq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vphadddq
  ;CHECK:       vphadddq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphadddq(<4 x i32> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphadddq(<4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_vphaddubd(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddubd
  ;CHECK:       vphaddubd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vphaddubd(<16 x i8> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vphaddubd(<16 x i8>) nounwind readnone

define <2 x i64> @stack_fold_vphaddubq(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddubq
  ;CHECK:       vphaddubq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphaddubq(<16 x i8> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphaddubq(<16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_vphaddubw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddubw
  ;CHECK:       vphaddubw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vphaddubw(<16 x i8> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vphaddubw(<16 x i8>) nounwind readnone

define <2 x i64> @stack_fold_vphaddudq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddudq
  ;CHECK:       vphaddudq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphaddudq(<4 x i32> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphaddudq(<4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_vphadduwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vphadduwd
  ;CHECK:       vphadduwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vphadduwd(<8 x i16> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vphadduwd(<8 x i16>) nounwind readnone

define <2 x i64> @stack_fold_vphadduwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vphadduwq
  ;CHECK:       vphadduwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphadduwq(<8 x i16> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphadduwq(<8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_vphaddwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddwd
  ;CHECK:       vphaddwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vphaddwd(<8 x i16> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vphaddwd(<8 x i16>) nounwind readnone

define <2 x i64> @stack_fold_vphaddwq(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vphaddwq
  ;CHECK:       vphaddwq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphaddwq(<8 x i16> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphaddwq(<8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_vphsubbw(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vphsubbw
  ;CHECK:       vphsubbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vphsubbw(<16 x i8> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vphsubbw(<16 x i8>) nounwind readnone

define <2 x i64> @stack_fold_vphsubdq(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vphsubdq
  ;CHECK:       vphsubdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_vphsubwd(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vphsubwd
  ;CHECK:       vphsubwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_vpmacsdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacsdd
  ;CHECK:       vpmacsdd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmacsdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmacsdd(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_vpmacsdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacsdqh
  ;CHECK:       vpmacsdqh {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpmacsdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpmacsdqh(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_vpmacsdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacsdql
  ;CHECK:       vpmacsdql {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpmacsdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpmacsdql(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <4 x i32> @stack_fold_vpmacssdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacssdd
  ;CHECK:       vpmacssdd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmacssdd(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmacssdd(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_vpmacssdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacssdqh
  ;CHECK:       vpmacssdqh {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpmacssdqh(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpmacssdqh(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_vpmacssdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacssdql
  ;CHECK:       vpmacssdql {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpmacssdql(<4 x i32> %a0, <4 x i32> %a1, <2 x i64> %a2)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpmacssdql(<4 x i32>, <4 x i32>, <2 x i64>) nounwind readnone

define <4 x i32> @stack_fold_vpmacsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacsswd
  ;CHECK:       vpmacsswd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmacsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmacsswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_vpmacssww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacssww
  ;CHECK:       vpmacssww {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpmacssww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpmacssww(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_vpmacswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacswd
  ;CHECK:       vpmacswd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmacswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmacswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_vpmacsww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2) {
  ;CHECK-LABEL: stack_fold_vpmacsww
  ;CHECK:       vpmacsww {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpmacsww(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %a2)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpmacsww(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_vpmadcsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmadcsswd
  ;CHECK:       vpmadcsswd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmadcsswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmadcsswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <4 x i32> @stack_fold_vpmadcswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2) {
  ;CHECK-LABEL: stack_fold_vpmadcswd
  ;CHECK:       vpmadcswd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16> %a0, <8 x i16> %a1, <4 x i32> %a2)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16>, <8 x i16>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_vpperm_rm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) {
  ;CHECK-LABEL: stack_fold_vpperm_rm
  ;CHECK:       vpperm {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2)
  ret <16 x i8> %2
}
define <16 x i8> @stack_fold_vpperm_mr(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) {
  ;CHECK-LABEL: stack_fold_vpperm_mr
  ;CHECK:       vpperm {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %a0, <16 x i8> %a2, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vpperm(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @stack_fold_vprotb(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_vprotb
  ;CHECK:       vprotb $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vprotbi(<16 x i8> %a0, i8 7)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vprotbi(<16 x i8>, i8) nounwind readnone

define <16 x i8> @stack_fold_vprotb_rm(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vprotb_rm
  ;CHECK:       vprotb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vprotb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
define <16 x i8> @stack_fold_vprotb_mr(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vprotb_mr
  ;CHECK:       vprotb {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vprotb(<16 x i8> %a1, <16 x i8> %a0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vprotb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_vprotd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_vprotd
  ;CHECK:       vprotd $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vprotdi(<4 x i32> %a0, i8 7)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vprotdi(<4 x i32>, i8) nounwind readnone

define <4 x i32> @stack_fold_vprotd_rm(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vprotd_rm
  ;CHECK:       vprotd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vprotd(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
define <4 x i32> @stack_fold_vprotd_mr(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vprotd_mr
  ;CHECK:       vprotd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vprotd(<4 x i32> %a1, <4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vprotd(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_vprotq(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_vprotq
  ;CHECK:       vprotq $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vprotqi(<2 x i64> %a0, i8 7)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vprotqi(<2 x i64>, i8) nounwind readnone

define <2 x i64> @stack_fold_vprotq_rm(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vprotq_rm
  ;CHECK:       vprotq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vprotq(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
define <2 x i64> @stack_fold_vprotq_mr(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vprotq_mr
  ;CHECK:       vprotq {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vprotq(<2 x i64> %a1, <2 x i64> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vprotq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @stack_fold_vprotw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_vprotw
  ;CHECK:       vprotw $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vprotwi(<8 x i16> %a0, i8 7)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vprotwi(<8 x i16>, i8) nounwind readnone

define <8 x i16> @stack_fold_vprotw_rm(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vprotw_rm
  ;CHECK:       vprotw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vprotw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
define <8 x i16> @stack_fold_vprotw_mr(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vprotw_mr
  ;CHECK:       vprotw {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vprotw(<8 x i16> %a1, <8 x i16> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vprotw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_vpshab_rm(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpshab_rm
  ;CHECK:       vpshab {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpshab(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
define <16 x i8> @stack_fold_vpshab_mr(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpshab_mr
  ;CHECK:       vpshab {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpshab(<16 x i8> %a1, <16 x i8> %a0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vpshab(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_vpshad_rm(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpshad_rm
  ;CHECK:       vpshad {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpshad(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
define <4 x i32> @stack_fold_vpshad_mr(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpshad_mr
  ;CHECK:       vpshad {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpshad(<4 x i32> %a1, <4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpshad(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_vpshaq_rm(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpshaq_rm
  ;CHECK:       vpshaq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
define <2 x i64> @stack_fold_vpshaq_mr(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpshaq_mr
  ;CHECK:       vpshaq {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64> %a1, <2 x i64> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @stack_fold_vpshaw_rm(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpshaw_rm
  ;CHECK:       vpshaw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
define <8 x i16> @stack_fold_vpshaw_mr(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpshaw_mr
  ;CHECK:       vpshaw {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16> %a1, <8 x i16> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_vpshlb_rm(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlb_rm
  ;CHECK:       vpshlb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
define <16 x i8> @stack_fold_vpshlb_mr(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlb_mr
  ;CHECK:       vpshlb {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8> %a1, <16 x i8> %a0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_vpshld_rm(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpshld_rm
  ;CHECK:       vpshld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpshld(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
define <4 x i32> @stack_fold_vpshld_mr(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_vpshld_mr
  ;CHECK:       vpshld {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.xop.vpshld(<4 x i32> %a1, <4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.xop.vpshld(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_vpshlq_rm(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlq_rm
  ;CHECK:       vpshlq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
define <2 x i64> @stack_fold_vpshlq_mr(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlq_mr
  ;CHECK:       vpshlq {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64> %a1, <2 x i64> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @stack_fold_vpshlw_rm(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlw_rm
  ;CHECK:       vpshlw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
define <8 x i16> @stack_fold_vpshlw_mr(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_vpshlw_mr
  ;CHECK:       vpshlw {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %a1, <8 x i16> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16>, <8 x i16>) nounwind readnone
