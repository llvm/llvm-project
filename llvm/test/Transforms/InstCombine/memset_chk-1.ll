; Test lib call simplification of __memset_chk calls with various values
; for dstlen and len.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; rdar://7719085

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.T = type { [100 x i32], [100 x i32], [1024 x i8] }
@t = common global %struct.T zeroinitializer

; Check cases where dstlen >= len.

define void @test_simplify1() {
; CHECK: @test_simplify1
  %dst = bitcast %struct.T* @t to i8*

; CHECK-NEXT: call void @llvm.memset.p0i8.i64
  call i8* @__memset_chk(i8* %dst, i32 0, i64 1824, i64 1824)
  ret void
}

define void @test_simplify2() {
; CHECK: @test_simplify2
  %dst = bitcast %struct.T* @t to i8*

; CHECK-NEXT: call void @llvm.memset.p0i8.i64
  call i8* @__memset_chk(i8* %dst, i32 0, i64 1824, i64 3648)
  ret void
}

define void @test_simplify3() {
; CHECK: @test_simplify3
  %dst = bitcast %struct.T* @t to i8*

; CHECK-NEXT: call void @llvm.memset.p0i8.i64
  call i8* @__memset_chk(i8* %dst, i32 0, i64 1824, i64 -1)
  ret void
}

; Check cases where dstlen < len.

define void @test_no_simplify1() {
; CHECK: @test_no_simplify1
  %dst = bitcast %struct.T* @t to i8*

; CHECK-NEXT: call i8* @__memset_chk
  call i8* @__memset_chk(i8* %dst, i32 0, i64 1824, i64 400)
  ret void
}

define void @test_no_simplify2() {
; CHECK: @test_no_simplify2
  %dst = bitcast %struct.T* @t to i8*

; CHECK-NEXT: call i8* @__memset_chk
  call i8* @__memset_chk(i8* %dst, i32 0, i64 1824, i64 0)
  ret void
}

declare i8* @__memset_chk(i8*, i32, i64, i64)
