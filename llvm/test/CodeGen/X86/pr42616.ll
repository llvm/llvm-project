; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=sse2 | FileCheck %s

define <2 x double> @pr42616(<2 x double> %a0, <2 x double> %a1, ptr %p) {
  ;CHECK-LABEL: pr42616
  ;CHECK:       movhpd (%esp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = load <2 x double>, ptr %p, align 1
  %3 = shufflevector <2 x double> %a1, <2 x double> %2, <2 x i32> <i32 2, i32 0>
  %4 = fadd <2 x double> %a0, %3
  ret <2 x double> %4
}
