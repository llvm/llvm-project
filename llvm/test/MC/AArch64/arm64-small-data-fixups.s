; RUN: llvm-mc -triple arm64-apple-darwin -filetype=obj -o - %s | llvm-readobj -r - | FileCheck %s

foo:
  .long 0
  .p2align 2
bar:
  .long 1

baz:
  .byte foo - bar
  .short foo - bar

; CHECK: File: <stdin>
; CHECK: Format: Mach-O arm64
; CHECK: Arch: aarch64
; CHECK: AddressSize: 64bit
; CHECK: Relocations [
; CHECK:  Section __text {
; CHECK:    0x9 0 1 1 ARM64_RELOC_SUBTRACTOR 0 bar
; CHECK:    0x9 0 1 1 ARM64_RELOC_UNSIGNED 0 foo
; CHECK:    0x8 0 0 1 ARM64_RELOC_SUBTRACTOR 0 bar
; CHECK:    0x8 0 0 1 ARM64_RELOC_UNSIGNED 0 foo
; CHECK:  }
; CHECK: ]
