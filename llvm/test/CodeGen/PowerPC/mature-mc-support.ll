; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).
; FIXME: PowerPC doesn't use the integrated assembler by default in all cases
; so we only test that -filetype=obj tries to parse the assembly.
; FIXME: PowerPC doesn't appear to support -filetype=obj for ppc64le

; SKIP: not llc -march=ppc32 < %s > /dev/null 2> %t1
; SKIP: FileCheck %s < %t1

; RUN: not llc -march=ppc32 -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; Test that we don't try to produce COFF for ppc.
; RUN: not llc -mtriple=powerpc-mingw32 -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; SKIP: not llc -march=ppc64 < %s > /dev/null 2> %t3
; SKIP: FileCheck %s < %t3

; RUN: not llc -march=ppc64 -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

; SKIP: not llc -march=ppc64le < %s > /dev/null 2> %t5
; SKIP: FileCheck %s < %t5

; SKIP: not llc -march=ppc64le -filetype=obj < %s > /dev/null 2> %t6
; SKIP: FileCheck %s < %t6

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
