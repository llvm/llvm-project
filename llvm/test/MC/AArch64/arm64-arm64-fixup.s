; RUN: llvm-mc < %s -triple arm64-apple-darwin --show-encoding | FileCheck %s

foo:
  adr x3, Lbar
; CHECK: adr x3, Lbar            ; encoding: [0x03'A',A,A,0x10'A']
; CHECK: fixup A - offset: 0, value: Lbar, kind: fixup_aarch64_pcrel_adr_imm21
Lbar:
  adrp x3, _printf@page
; CHECK: adrp x3, _printf@PAGE      ; encoding: [0x03'A',A,A,0x90'A']
; CHECK: fixup A - offset: 0, value: _printf@PAGE, kind: fixup_aarch64_pcrel_adrp_imm21
