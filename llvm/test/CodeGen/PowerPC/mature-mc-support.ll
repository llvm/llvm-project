; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -mtriple=ppc32-- < %s > /dev/null 2> %t1
; RUN: FileCheck %s < %t1

; RUN: not llc -mtriple=ppc32-- -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; Test that we don't try to produce COFF for ppc.
; RUN: not llc -mtriple=powerpc-mingw32 -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

; RUN: not llc -mtriple=ppc64-- < %s > /dev/null 2> %t3
; RUN: FileCheck %s < %t3

; RUN: not llc -mtriple=ppc64-- -filetype=obj < %s > /dev/null 2> %t4
; RUN: FileCheck %s < %t4

; RUN: not llc -mtriple=ppc64--le < %s > /dev/null 2> %t5
; RUN: FileCheck %s < %t5

; RUN: not llc -mtriple=ppc64--le -filetype=obj < %s > /dev/null 2> %t6
; RUN: FileCheck %s < %t6

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
