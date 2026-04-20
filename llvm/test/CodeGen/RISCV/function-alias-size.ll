; RUN: llc -mtriple=riscv32 --relocation-model=pic -target-abi ilp32 %s -o - < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 --relocation-model=pic -target-abi ilp32 %s -o - -filetype=obj < %s | llvm-objdump --syms -r - | FileCheck %s --check-prefix=OBJDUMP32

; RUN: llc -mtriple=riscv64 --relocation-model=pic -target-abi lp64 %s -o - < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 --relocation-model=pic -target-abi lp64 %s -o - -filetype=obj < %s | llvm-objdump --syms -r - | FileCheck %s --check-prefix=OBJDUMP64

; Check that we emit size information for function aliases:

@a = constant ptr bitcast (ptr @_ZN3fooD1Ev to ptr)
@_ZN3fooD1Ev = alias void (), void ()* @_ZN3fooD2Ev
define void @_ZN3fooD2Ev() nounwind {
; ASM-LABEL: _ZN3fooD2Ev:
; ASM:       # %bb.0:
; ASM-NEXT:    ret
  ret void
}

@two_ints = private global {i32, i32} {i32 1, i32 2}
@elem0 = alias i32, getelementptr({i32, i32}, {i32, i32}*  @two_ints, i32 0, i32 0)
@elem1 = alias i32, getelementptr({i32, i32}, {i32, i32}*  @two_ints, i32 0, i32 1)

; ASM-LABEL: .Ltwo_ints:
; ASM-NEXT: .word 1
; ASM-NEXT: .word 2
; ASM-NEXT: .size .Ltwo_ints, 8

; The function alias symbol should have the same size expression:
; ASM-LABEL: .globl _ZN3fooD1Ev
; ASM-NEXT: .type _ZN3fooD1Ev,@function
; ASM-NEXT: _ZN3fooD1Ev = _ZN3fooD2Ev
; ASM-NEXT: .size _ZN3fooD1Ev, .Lfunc_end0-_ZN3fooD2Ev

; But for the aliases using a GEP, we have to subtract the offset:
; ASM-LABEL: .globl elem0
; ASM-NEXT:  elem0 = .Ltwo_ints
; ASM-NEXT:  .size elem0, 4
; ASM-LABEL: .globl elem1
; ASM-NEXT:  elem1 = .Ltwo_ints+4
; ASM-NEXT:  .size elem1, 4

; Check that the ELF st_size value was set correctly:
; OBJDUMP32-LABEL: SYMBOL TABLE:
; OBJDUMP32-NEXT: {{0+}}0 l    df *ABS* {{0+}} function-alias-size.ll
; OBJDUMP32-DAG: {{0+}}0 g     F .text [[SIZE:[0-9a-f]+]] _ZN3fooD2Ev
; OBJDUMP32-DAG: {{0+}}0 g     O .data.rel.ro {{0+}}4 a
; OBJDUMP32-DAG: {{0+}}0 g     F .text [[SIZE]] _ZN3fooD1Ev
; elem1 should have a size of 4 and not 8:
; OBJDUMP32-DAG: {{0+}}0 g     O .{{s?}}data {{0+}}4 elem0
; OBJDUMP32-DAG: {{0+}}4 g     O .{{s?}}data {{0+}}4 elem1

; OBJDUMP64-LABEL: SYMBOL TABLE:
; OBJDUMP64-NEXT: {{0+}}0 l    df *ABS* {{0+}} function-alias-size.ll
; OBJDUMP64-DAG: {{0+}}0 g     F .text [[SIZE:[0-9a-f]+]] _ZN3fooD2Ev
; OBJDUMP64-DAG: {{0+}}0 g     O .data.rel.ro {{0+}}8 a
; OBJDUMP64-DAG: {{0+}}0 g     F .text [[SIZE]] _ZN3fooD1Ev
; elem1 should have a size of 4 and not 8:
; OBJDUMP64-DAG: {{0+}}0 g     O .{{s?}}data {{0+}}4 elem0
; OBJDUMP64-DAG: {{0+}}4 g     O .{{s?}}data {{0+}}4 elem1
