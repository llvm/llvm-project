; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -mtriple=s390x-linux-gnu < %s > /dev/null 2> %t1
; RUN: FileCheck %s < %t1

; RUN: not llc -mtriple=s390x-linux-gnu -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2


module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: error: unknown directive
