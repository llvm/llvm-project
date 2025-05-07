; RUN: llc -mtriple=mips -fast-isel=false -O0 < %s 2>&1 | FileCheck %s -check-prefix=O0
; RUN: llc -mtriple=mips -fast-isel=false -O2 < %s 2>&1 | FileCheck %s -check-prefix=O2

; At -O0, DAGCombine won't try to merge these consecutive loads but it will at
; -O2.

define void @foo() nounwind {
entry:
  %0 = alloca [2 x i8], align 32
  store i8 1, ptr %0
  %1 = getelementptr inbounds [2 x i8], ptr %0, i32 0, i32 1
  store i8 1, ptr %1
  ret void
}

; O0: addiu $[[REG:[0-9]+]], $zero, 1
; O0-DAG: sb $[[REG]], 0($sp)
; O0-DAG: sb $[[REG]], 1($sp)

; O2: addiu $[[REG:[0-9]+]], $zero, 257
; O2: sh $[[REG]], 0($sp)
