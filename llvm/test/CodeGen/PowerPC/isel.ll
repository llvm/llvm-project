target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s

define i64 @test1(i64 %a, i64 %b, i64 %c, i64 %d) {
entry:
	%p = icmp uge i64 %a, %b
	%x = select i1 %p, i64 %c, i64 %d
	ret i64 %x
; CHECK: @test1
; CHECK: isel
}

define i32 @test2(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
	%p = icmp uge i32 %a, %b
	%x = select i1 %p, i32 %c, i32 %d
	ret i32 %x
; CHECK: @test2
; CHECK: isel
}

