; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=generic < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a35 < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a53 < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a73 < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cyclone < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=falkor < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=kryo < %s | FileCheck --check-prefix=ALIGN2 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderx < %s | FileCheck --check-prefix=ALIGN3 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt81 < %s | FileCheck --check-prefix=ALIGN3 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt83 < %s | FileCheck --check-prefix=ALIGN3 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderxt88 < %s | FileCheck --check-prefix=ALIGN3 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=thunderx2t99 < %s | FileCheck --check-prefix=ALIGN3 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a57 < %s | FileCheck --check-prefix=ALIGN4 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=cortex-a72 < %s | FileCheck --check-prefix=ALIGN4 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m1 < %s | FileCheck --check-prefix=ALIGN4 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m2 < %s | FileCheck --check-prefix=ALIGN4 %s
; RUN: llc -mtriple=aarch64-unknown-linux -mcpu=exynos-m3 < %s | FileCheck --check-prefix=ALIGN4 %s

define void @test() {
  ret void
}

; CHECK-LABEL: test
; ALIGN2: .p2align 2
; ALIGN3: .p2align 3
; ALIGN4: .p2align 4
