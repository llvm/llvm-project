; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -march=x86 < %s > /dev/null 2> %t1
; RUN: FileCheck %s < %t1

; RUN: not llc -march=x86 -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; RUN: not llc -march=x86-64 < %s > /dev/null 2> %t3
; RUN: FileCheck %s < %t3

; RUN: not llc -march=x86-64 -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
