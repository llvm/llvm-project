; Test parameter passing and return values
;RUN: llc --march=cellspu %s -o - | FileCheck %s

; this fits into registers r3-r74
%paramstruct = type { i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
                      i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
                      i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
                      i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
                      i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
                      i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32}
define ccc i32 @test_regs( %paramstruct %prm )
{
;CHECK:	lr	$3, $74
;CHECK:	bi	$lr
  %1 = extractvalue %paramstruct %prm, 71
  ret i32 %1
}

define ccc i32 @test_regs_and_stack( %paramstruct %prm, i32 %stackprm )
{
;CHECK-NOT:	a	$3, $74, $75
  %1 = extractvalue %paramstruct %prm, 71
  %2 = add i32 %1, %stackprm
  ret i32 %2
}

define ccc %paramstruct @test_return( i32 %param,  %paramstruct %prm )
{
;CHECK:  lqd	{{\$[0-9]+}}, 80($sp)
;CHECK-NOT:	ori	{{\$[0-9]+, \$[0-9]+, 0}}
;CHECK:  lr    $3, $4
  ret %paramstruct %prm
}

