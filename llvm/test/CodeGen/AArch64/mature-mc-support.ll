; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -mtriple=aarch64-pc-linux < %s > /dev/null 2> %t3
; RUN: FileCheck %s < %t3

; RUN: not llc -mtriple=aarch64-pc-linux -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
