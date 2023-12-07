; pr23772 - [ARM] r226200 can emit illegal thumb2 instruction: "sub sp, r12, #80"
; RUN: llc -march=thumb -mcpu=cortex-m3 -O3 -filetype=asm -o - %s | FileCheck %s
; CHECK-NOT: sub{{.*}} sp, r{{.*}}, #
; CHECK:     .fnend
; TODO: Missed optimization. The three instructions generated to subtract SP can be converged to a single one
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"
target triple = "thumbv7m-unknown-unknown"
%B = type {ptr}
%R = type {i32}
%U = type {ptr, i8, i8}
%E = type {ptr, ptr}
%X = type {i32, i8, i8}
declare external ptr @memalloc(i32, i32, i32)
declare external void @memfree(ptr, i32, i32)
define void @foo(ptr %pb$, ptr %pr$) nounwind {
L.0:
	%pb = alloca ptr
	%pr = alloca ptr
	store ptr %pb$, ptr %pb
	store ptr %pr$, ptr %pr
	%pe = alloca ptr
	%0 = load ptr, ptr %pb
	store ptr %0, ptr %pe
	%1 = load ptr, ptr %pr
	%2 = load i32, ptr %1
	switch i32 %2, label %L.1 [
		i32 1, label %L.3
	]
L.3:
	%px = alloca ptr
	%3 = load ptr, ptr %pr
	store ptr %3, ptr %px
	%4 = load ptr, ptr %px
	%5 = load i32, ptr %4
	%6 = icmp ne i32 %5, 0
	br i1 %6, label %L.5, label %L.4
L.5:
	%pu = alloca ptr
	%7 = call ptr @memalloc(i32 8, i32 4, i32 0)
	store ptr %7, ptr %pu
	%8 = load ptr, ptr %px
	%9 = getelementptr %X, ptr %8, i32 0, i32 1
	%10 = load i8, ptr %9
	%11 = load ptr, ptr %pu
	%12 = getelementptr %U, ptr %11, i32 0, i32 1
	store i8 %10, ptr %12
	%13 = load ptr, ptr %pe
	%14 = getelementptr %E, ptr %13, i32 0, i32 1
	%15 = load ptr, ptr %14
	%16 = load ptr, ptr %pu
	store ptr %15, ptr %16
	%17 = load ptr, ptr %pu
	%18 = load ptr, ptr %pe
	%19 = getelementptr %E, ptr %18, i32 0, i32 1
	store ptr %17, ptr %19
	br label %L.4
L.4:
	%20 = load ptr, ptr %px
	call void @memfree(ptr %20, i32 8, i32 0)
	br label %L.2
L.1:
	br label %L.2
L.2:
	br label %return
return:
	ret void
}
