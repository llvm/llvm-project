; Test emitting version_min directives.

; RUN: llc %s -filetype=asm -o - --mtriple x86_64-apple-tvos9.0.0-simulator | FileCheck %s --check-prefix=TVOS
; RUN: llc %s -filetype=asm -o - --mtriple x86_64-apple-tvos9.0.0 | FileCheck %s --check-prefix=TVOS
; RUN: llc %s -filetype=asm -o - --mtriple x86_64-apple-driverkit19.0.0 | FileCheck %s --check-prefix=DRIVERKIT
; RUN: llc %s -filetype=asm -o - --mtriple i386-apple-ios7.0.0-simulator | FileCheck %s --check-prefix=IOS
; RUN: llc %s -filetype=asm -o - --mtriple i386-apple-watchos2.0.0-simulator | FileCheck %s --check-prefix=WATCHOS

; TVOS: .tvos_version_min 9, 0
; DRIVERKIT: .build_version driverkit, 19, 0
; IOS: .ios_version_min 7, 0
; WATCHOS: .watchos_version_min 2, 0
