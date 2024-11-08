; Test emitting version_min directives.

; Let's not split this into separate ARM/AArch64 parts.
; REQUIRES: aarch64-registered-target

; RUN: llc %s -filetype=asm -o - --mtriple arm64-apple-tvos9.0.0 | FileCheck %s --check-prefix=TVOS
; RUN: llc %s -filetype=asm -o - --mtriple thumbv7s-apple-ios7.0.0 | FileCheck %s --check-prefix=IOS
; RUN: llc %s -filetype=asm -o - --mtriple thumbv7k-apple-watchos2.0.0 | FileCheck %s --check-prefix=WATCHOS

; TVOS: .tvos_version_min 9, 0
; IOS: .ios_version_min 7, 0
; WATCHOS: .watchos_version_min 2, 0
