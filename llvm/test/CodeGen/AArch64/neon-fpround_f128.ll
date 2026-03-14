; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

define <1 x double> @test_fpround_v1f128(ptr %a) {
; CHECK-LABEL: test_fpround_v1f128:
; CHECK: bl __trunctfdf2
  %b = load <1 x fp128>, ptr %a
  %c = fptrunc <1 x fp128> %b to <1 x double>
  ret <1 x double> %c
}

define <2 x double> @test_fpround_v2f128(ptr %a) {
; CHECK-LABEL: test_fpround_v2f128:
; CHECK: bl __trunctfdf2
; CHECK: bl __trunctfdf2
  %b = load <2 x fp128>, ptr %a
  %c = fptrunc <2 x fp128> %b to <2 x double>
  ret <2 x double> %c
}
