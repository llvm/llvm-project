; RUN: llc -mtriple=arm64-apple-ios7.0 -o - %s | FileCheck %s --check-prefix=CHECK-MACHO
; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK-ELF

; CHECK-MACHO: .subsections_via_symbols
; CHECK-ELF-NOT: .subsections_via_symbols
