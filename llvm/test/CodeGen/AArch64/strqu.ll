; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-linux-gnu    | FileCheck --check-prefixes=CHECK,NOSPLIT %s
; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64_be-linux-gnu | FileCheck --check-prefixes=CHECK,NOSPLIT %s
; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-linux-gnu    -mcpu=exynos-m1 | FileCheck --check-prefixes=CHECK,NOSPLIT %s
; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64_be-linux-gnu -mcpu=exynos-m1 | FileCheck --check-prefixes=CHECK,SPLIT %s

; CHECK-LABEL: test_split_f:
; NOSPLIT: str q{{[0-9]+}}, [x{{[0-9]+}}]
; SPLIT: st1 { v{{[0-9]+}}.2s }, [x{{[0-9]+}}]
; SPLIT: st1 { v{{[0-9]+}}.2s }, [x{{[0-9]+}}]
define void @test_split_f(<4 x float> %val, <4 x float>* %addr) {
  store <4 x float> %val, <4 x float>* %addr, align 8
  ret void
}

; CHECK-LABEL: test_split_d:
; NOSPLIT: str q{{[0-9]+}}, [x{{[0-9]+}}]
; SPLIT: st1 { v{{[0-9]+}}.2d }, [x{{[0-9]+}}]
define void @test_split_d(<2 x double> %val, <2 x double>* %addr) {
  store <2 x double> %val, <2 x double>* %addr, align 8
  ret void
}

; CHECK-LABEL: test_split_128:
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}]
define void @test_split_128(fp128 %val, fp128* %addr) {
  store fp128 %val, fp128* %addr, align 8
  ret void
}
