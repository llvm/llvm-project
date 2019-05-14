; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=generic < %s | FileCheck --check-prefixes=ALIGN2,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a35 < %s | FileCheck --check-prefixes=ALIGN2,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cyclone < %s | FileCheck --check-prefixes=ALIGN2,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=falkor < %s | FileCheck --check-prefixes=ALIGN2,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=kryo < %s | FileCheck --check-prefixes=ALIGN2,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a53 < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderx < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt81 < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt83 < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt88 < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderx2t99 < %s | FileCheck --check-prefixes=ALIGN3,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a57 < %s | FileCheck --check-prefixes=ALIGN4,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a72 < %s | FileCheck --check-prefixes=ALIGN4,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a73 < %s | FileCheck --check-prefixes=ALIGN4,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m1 < %s | FileCheck --check-prefixes=ALIGN4,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m2 < %s | FileCheck --check-prefixes=ALIGN4,CHECK %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m3 < %s | FileCheck --check-prefixes=ALIGN5,CHECK %s

define void @test() {
  ret void
}

; CHECK-LABEL: test
; ALIGN2: .p2align 2
; ALIGN3: .p2align 3
; ALIGN4: .p2align 4
; ALIGN5: .p2align 5

define void @test_optsize() optsize {
  ret void
}

; CHECK-LABEL: test_optsize
; CHECK-NEXT: .p2align 2
