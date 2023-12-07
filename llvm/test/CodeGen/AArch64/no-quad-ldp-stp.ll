; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+slow-paired-128 -verify-machineinstrs -asm-verbose=false | FileCheck %s --check-prefixes=CHECK,SLOW
; RUN: llc < %s -mtriple=aarch64-eabi -mcpu=exynos-m3         -verify-machineinstrs -asm-verbose=false | FileCheck %s --check-prefixes=CHECK,FAST

; CHECK-LABEL: test_nopair_st
; SLOW: str
; SLOW: stur
; SLOW-NOT: stp
; FAST: stp
define void @test_nopair_st(ptr %ptr, <2 x double> %v1, <2 x double> %v2) {
  store <2 x double> %v2, ptr %ptr, align 16
  %add.ptr = getelementptr inbounds double, ptr %ptr, i64 -2
  store <2 x double> %v1, ptr %add.ptr, align 16
  ret void
}

; CHECK-LABEL: test_nopair_ld
; SLOW: ldr
; SLOW: ldr
; SLOW-NOT: ldp
; FAST: ldp
define <2 x i64> @test_nopair_ld(ptr %p) {
  %tmp1 = load <2 x i64>, < 2 x i64>* %p, align 8
  %add.ptr2 = getelementptr inbounds i64, ptr %p, i64 2
  %tmp2 = load <2 x i64>, ptr %add.ptr2, align 8
  %add = add nsw <2 x i64> %tmp1, %tmp2
  ret <2 x i64> %add
}
