; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).
; FIXME: SPARC doesn't use the integrated assembler by default in all cases
; so we only test that -filetype=obj tries to parse the assembly.

; SKIP: not llc -march=sparc < %s > /dev/null 2> %t1
; SKIP: FileCheck %s < %t1

; RUN: not llc -march=sparc -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; SKIP: not llc -march=sparcv9 < %s > /dev/null 2> %t3
; SKIP: FileCheck %s < %t3

; RUN: not llc -march=sparcv9 -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
