; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).
; FIXME: Mips doesn't use the integrated assembler by default so we only test
; that -filetype=obj tries to parse the assembly.

; SKIP: not llc -march=mips < %s > /dev/null 2> %t1
; SKIP: FileCheck %s < %t1

; RUN: not llc -march=mips -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; SKIP: not llc -march=mipsel < %s > /dev/null 2> %t3
; SKIP: FileCheck %s < %t3

; RUN: not llc -march=mipsel -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

; SKIP: not llc -march=mips64 < %s > /dev/null 2> %t5
; SKIP: FileCheck %s < %t5

; RUN: not llc -march=mips64 -filetype=obj < %s > /dev/null 2> %t6
; RUN: FileCheck %s < %t6

; SKIP: not llc -march=mips64el < %s > /dev/null 2> %t7
; SKIP: FileCheck %s < %t7

; RUN: not llc -march=mips64el -filetype=obj < %s > /dev/null 2> %t8
; RUN: FileCheck %s < %t8

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
