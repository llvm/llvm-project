; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Tests bug: https://llvm.org/bugs/show_bug.cgi?id=24646
; CHECK: error: failed to parse constraints

define void @foo() nounwind {
call void asm sideeffect "mov x0, #42","=~{x0},~{x19},mov |0,{x19},mov x0, #4~x{21}"()ounwi #4~x{21}"()ounwindret
