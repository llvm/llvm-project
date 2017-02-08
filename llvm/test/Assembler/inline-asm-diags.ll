; RUN: not llc -march=x86 -filetype=obj < %s 2>&1 -o /dev/null | FileCheck %s

module asm ".word 0x10"
module asm ".word -bar"

; CHECK: <inline asm>:2:7: error: expected relocatable expression

module asm ".word -foo"
; CHECK: <inline asm>:3:7: error: expected relocatable expression
