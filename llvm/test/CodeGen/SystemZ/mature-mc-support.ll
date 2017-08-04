; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).
; FIXME: SystemZ doesn't use the integrated assembler by default so we only test
; that -filetype=obj tries to parse the assembly.

; SKIP: not llc -march=systemz < %s > /dev/null 2> %t1
; SKIP: FileCheck %s < %t1

; RUN: not llc -march=systemz -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2


module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
