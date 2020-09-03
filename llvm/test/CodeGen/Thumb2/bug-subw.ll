; pr23772 - [ARM] r226200 can emit illegal thumb2 instruction: "sub sp, r12, #80"
; RUN: llc -march=thumb -mcpu=cortex-m3 -O3 -filetype=asm -o - %s | FileCheck %s
; CHECK-NOT: sub{{.*}} sp, r{{.*}}, #
; CHECK:     .fnend
; TODO: Missed optimization. The three instructions generated to subtract SP can be converged to a single one
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"
target triple = "thumbv7m-unknown-unknown"
%B = type {%B*}
%R = type {i32}
%U = type {%U*, i8, i8}
%E = type {%B*, %U*}
%X = type {i32, i8, i8}
declare external [0 x i8]* @memalloc(i32, i32, i32)
declare external void @memfree([0 x i8]*, i32, i32)
define void @foo(%B* %pb$, %R* %pr$) nounwind {
L.0:
	%pb = alloca %B*
	%pr = alloca %R*
	store %B* %pb$, %B** %pb
	store %R* %pr$, %R** %pr
	%pe = alloca %E*
	%0 = load %B*, %B** %pb
	%1 = bitcast %B* %0 to %E*
	store %E* %1, %E** %pe
	%2 = load %R*, %R** %pr
	%3 = getelementptr %R, %R* %2, i32 0, i32 0
	%4 = load i32, i32* %3
	switch i32 %4, label %L.1 [
		i32 1, label %L.3
	]
L.3:
	%px = alloca %X*
	%5 = load %R*, %R** %pr
	%6 = bitcast %R* %5 to %X*
	store %X* %6, %X** %px
	%7 = load %X*, %X** %px
	%8 = getelementptr %X, %X* %7, i32 0, i32 0
	%9 = load i32, i32* %8
	%10 = icmp ne i32 %9, 0
	br i1 %10, label %L.5, label %L.4
L.5:
	%pu = alloca %U*
	%11 = call [0 x i8]* @memalloc(i32 8, i32 4, i32 0)
	%12 = bitcast [0 x i8]* %11 to %U*
	store %U* %12, %U** %pu
	%13 = load %X*, %X** %px
	%14 = getelementptr %X, %X* %13, i32 0, i32 1
	%15 = load i8, i8* %14
	%16 = load %U*, %U** %pu
	%17 = getelementptr %U, %U* %16, i32 0, i32 1
	store i8 %15, i8* %17
	%18 = load %E*, %E** %pe
	%19 = getelementptr %E, %E* %18, i32 0, i32 1
	%20 = load %U*, %U** %19
	%21 = load %U*, %U** %pu
	%22 = getelementptr %U, %U* %21, i32 0, i32 0
	store %U* %20, %U** %22
	%23 = load %U*, %U** %pu
	%24 = load %E*, %E** %pe
	%25 = getelementptr %E, %E* %24, i32 0, i32 1
	store %U* %23, %U** %25
	br label %L.4
L.4:
	%26 = load %X*, %X** %px
	%27 = bitcast %X* %26 to [0 x i8]*
	call void @memfree([0 x i8]* %27, i32 8, i32 0)
	br label %L.2
L.1:
	br label %L.2
L.2:
	br label %return
return:
	ret void
}
