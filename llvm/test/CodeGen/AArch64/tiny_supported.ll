; RUN: llc -verify-machineinstrs -o - -mtriple=aarch64-none-linux-gnu -code-model=tiny < %s 2>&1 | FileCheck %s
; RUN: llc -verify-machineinstrs -o - -mtriple=aarch64-none-eabi -code-model=tiny < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -verify-machineinstrs -o - -mtriple=arm64-apple-darwin -code-model=tiny < %s 2>&1 | FileCheck %s --check-prefix=NOTINY
; RUN: not --crash llc -verify-machineinstrs -o - -mtriple=arm64-apple-ios -code-model=tiny < %s 2>&1 | FileCheck %s --check-prefix=NOTINY
; RUN: not --crash llc -verify-machineinstrs -o - -mtriple=aarch64-unknown-windows-msvc -code-model=tiny < %s 2>&1 | FileCheck %s --check-prefix=NOTINY

; CHECK-NOT:   tiny code model is only supported on ELF
; CHECK-LABEL:   foo
; NOTINY:   tiny code model is only supported on ELF

define void @foo() {
  ret void
}
