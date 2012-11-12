; RUN: llc < %s -march=cellspu | FileCheck %s

%0 = type {i32, i32} 
@buffer = global [ 72 x %0 ] zeroinitializer

define void@test( ) {
; Check that there is no illegal "a rt, ra, imm" instruction 
; CHECK-NOT:	a	 {{\$., \$., 5..}}
; CHECK:	a	{{\$., \$., \$.}}
	store %0 {i32 1, i32 2} , 
                %0* getelementptr ([72 x %0]* @buffer, i32 0, i32 71)
	ret void
}
