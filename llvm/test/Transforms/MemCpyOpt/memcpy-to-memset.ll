; RUN: opt -memcpyopt -S < %s | FileCheck %s

@cst = internal constant [3 x i32] [i32 -1, i32 -1, i32 -1], align 4

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
declare void @foo(i32*) nounwind

define void @test1() nounwind {
  %arr = alloca [3 x i32], align 4
  %arr_i8 = bitcast [3 x i32]* %arr to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arr_i8, i8* bitcast ([3 x i32]* @cst to i8*), i64 12, i32 4, i1 false)
  %arraydecay = getelementptr inbounds [3 x i32]* %arr, i64 0, i64 0
  call void @foo(i32* %arraydecay) nounwind
  ret void
; CHECK: @test1
; CHECK: call void @llvm.memset
; CHECK-NOT: call void @llvm.memcpy
; CHECK: ret void
}
