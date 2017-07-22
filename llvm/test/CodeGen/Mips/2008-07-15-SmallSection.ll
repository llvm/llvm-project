; RUN: llc < %s -march=mips -mcpu=mips32 -mips-ssection-threshold=8 \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt \
; RUN:   | FileCheck %s --check-prefixes=BASIC,COMMON
; RUN: llc < %s -march=mips -mcpu=mips32 -mips-ssection-threshold=8 \
; RUN:     -relocation-model=static -mattr=+noabicalls -mgpopt -membedded-data \
; RUN:   | FileCheck %s --check-prefixes=EMBDATA,COMMON

; Test the layout of objects when compiling for static, noabicalls environment.

%struct.anon = type { i32, i32 }

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

