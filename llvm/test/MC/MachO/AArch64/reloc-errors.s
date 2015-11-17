; RUN: not llvm-mc -triple aarch64-none-macho %s -filetype=obj -o - 2>&1 | FileCheck %s

; CHECK: error: conditional branch requires assembler-local label. 'external' is external.
  b.eq external

; CHECK: error: Invalid relocation on conditional branch
  tbz w0, #4, external

; CHECK: error: unknown AArch64 fixup kind!
  adr x0, external
