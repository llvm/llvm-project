; RUN: opt -basicaa -memcpyopt -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = global i32 0

declare void @otherf(i32*)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

; memcpyopt should not touch atomic ops
define void @test1() nounwind uwtable ssp {
; CHECK: test1
; CHECK: store atomic
  %x = alloca [101 x i32], align 16
  %bc = bitcast [101 x i32]* %x to i8*
  call void @llvm.memset.p0i8.i64(i8* %bc, i8 0, i64 400, i1 false)
  %gep1 = getelementptr inbounds [101 x i32], [101 x i32]* %x, i32 0, i32 100
  store atomic i32 0, i32* %gep1 unordered, align 4
  %gep2 = getelementptr inbounds [101 x i32], [101 x i32]* %x, i32 0, i32 0
  call void @otherf(i32* %gep2)
  ret void
}

; memcpyopt across unordered store
define void @test2() nounwind uwtable ssp {
; CHECK: test2
; CHECK: call
; CHECK-NEXT: store atomic
; CHECK-NEXT: call
  %old = alloca i32
  %new = alloca i32
  call void @otherf(i32* nocapture %old)
  store atomic i32 0, i32* @x unordered, align 4
  %v = load i32, i32* %old
  store i32 %v, i32* %new
  call void @otherf(i32* nocapture %new)  
  ret void
}

