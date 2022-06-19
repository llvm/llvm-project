; RUN: llc < %s -mtriple=mips -mcpu=mips32 -mips-ssection-threshold=8 -verify-machineinstrs \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt \
; RUN:   | FileCheck %s --check-prefixes=BASIC,COMMON,ADDR32
; RUN: llc < %s -mtriple=mips -mcpu=mips32 -mips-ssection-threshold=8 -verify-machineinstrs \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt -membedded-data \
; RUN:   | FileCheck %s --check-prefixes=EMBDATA,COMMON,ADDR32

; RUN: llc < %s -mtriple=mips64 -mcpu=mips4 -mips-ssection-threshold=8 -verify-machineinstrs \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt -target-abi n64 \
; RUN:   | FileCheck %s --check-prefixes=BASIC,COMMON,N64
; RUN: llc < %s -mtriple=mips64 -mcpu=mips4 -mips-ssection-threshold=8 -verify-machineinstrs \
; RUN:     -relocation-model=static -mattr=+noabicalls,+sym32 -mgpopt -target-abi n64 \
; RUN:   | FileCheck %s --check-prefixes=BASIC,COMMON,N64
; RUN: llc < %s -mtriple=mips64 -mcpu=mips4 -mips-ssection-threshold=8 -verify-machineinstrs \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt -target-abi n32 \
; RUN:   | FileCheck %s --check-prefixes=BASIC,COMMON,ADDR32

; Test the layout of objects when compiling for static, noabicalls environment.

%struct.anon = type { i32, i32 }

; Check that when synthesizing a pointer to the second element of foo, that
; we use the correct addition operation. O32 and N32 have 32-bit address
; spaces, so they use addiu. N64 has a 64bit address space, but has a submode
; where symbol sizes are 32 bits. In those cases we use daddiu.

; CHECK-LABEL: A1:
; N64:       daddiu ${{[0-9]+}}, $gp, %gp_rel(foo)
; ADDR32:    addiu ${{[0-9]+}}, $gp, %gp_rel(foo)

; BASIC: .type  s0,@object
; BASIC-NEXT: .section .sdata,"aw",@progbits

; EMDATA: .type  s0,@object
; EMDATA-NEXT: .section .rodata,"a",@progbits

@s0 = constant [8 x i8] c"AAAAAAA\00", align 4

; BASIC: .type  foo,@object
; BASIC-NOT:  .section

; EMBDATA: .type  foo,@object
; EMBDATA-NEXT:  .section .sdata,"aw",@progbits
@foo = global %struct.anon { i32 2, i32 3 }

; COMMON:  .type bar,@object
; COMMON-NEXT:  .section  .sbss,"aw",@nobits
@bar = global %struct.anon zeroinitializer

define i8* @A0() nounwind {
entry:
	ret i8* getelementptr ([8 x i8], [8 x i8]* @s0, i32 0, i32 0)
}

define i32 @A1() nounwind {
entry:
  load i32, i32* getelementptr (%struct.anon, %struct.anon* @foo, i32 0, i32 0), align 8 
  load i32, i32* getelementptr (%struct.anon, %struct.anon* @foo, i32 0, i32 1), align 4 
  add i32 %1, %0
  ret i32 %2
}

