; DISABLED: llc < %s -mips-ssection-threshold=8 -march=mips -o %t0
; DISABLED: llc < %s -mips-ssection-threshold=0 -march=mips -o %t1
; DISABLED: grep {sdata} %t0 | count 1
; DISABLED: grep {sbss} %t0 | count 1
; DISABLED: grep {gp_rel} %t0 | count 2
; DISABLED: not grep {sdata} %t1 
; DISABLED: not grep {sbss} %t1 
; DISABLED: not grep {gp_rel} %t1
; DISABLED: grep {\%hi} %t1 | count 2
; DISABLED: grep {\%lo} %t1 | count 3
; RUN: false
; XFAIL: *


target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-unknown-psp-elf"

  %struct.anon = type { i32, i32 }
@s0 = global [8 x i8] c"AAAAAAA\00", align 4
@foo = global %struct.anon { i32 2, i32 3 }
@bar = global %struct.anon zeroinitializer 

define i8* @A0() nounwind {
entry:
	ret i8* getelementptr ([8 x i8]* @s0, i32 0, i32 0)
}

define i32 @A1() nounwind {
entry:
  load i32* getelementptr (%struct.anon* @foo, i32 0, i32 0), align 8 
  load i32* getelementptr (%struct.anon* @foo, i32 0, i32 1), align 4 
  add i32 %1, %0
  ret i32 %2
}

