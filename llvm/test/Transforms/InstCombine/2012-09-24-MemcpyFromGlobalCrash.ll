; RUN: opt < %s -instcombine -S | FileCheck %s

; Check we don't crash due to lack of target data.

@G = constant [100 x i8] zeroinitializer

declare void @bar(i8*)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

define void @test() {
; CHECK: @test
; CHECK: llvm.memcpy
; CHECK: ret void
  %A = alloca [100 x i8]
  %a = getelementptr inbounds [100 x i8]* %A, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* getelementptr inbounds ([100 x i8]* @G, i64 0, i32 0), i64 100, i32 4, i1 false)
  call void @bar(i8* %a) readonly
  ret void
}
