; RUN: llc < %s -march=mips -mcpu=mips32 -mips-ssection-threshold=8 -relocation-model=static -mattr=+noabicalls -mgpopt | FileCheck %s

%struct.anon = type { i32, i32 }

@s0 = constant [8 x i8] c"AAAAAAA\00", align 4

; CHECK: .type  foo,@object
; CHECK-NEXT: .section .sdata,"aw",@progbits
@foo = global %struct.anon { i32 2, i32 3 }

; CHECK:  .type bar,@object
; CHECK-NEXT:  .section  .sbss,"aw",@nobits
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

