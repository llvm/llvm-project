; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@X = global i19 4, align 16

define i19 *@test() align 32 {
	%X = alloca i19, align 4
	%Y = alloca i51, i32 42, align 16
	%Z = alloca i32, align 1
	ret i19 *%X
}

define i19 *@test2() {
	%X = malloc i19, align 4
	%Y = malloc i51, i32 42, align 16
	%Z = malloc i32, align 1
	ret i19 *%X
}


