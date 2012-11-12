; RUN: opt -scalarrepl -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.lifetime.start(i64, i8*)
declare void @llvm.lifetime.end(i64, i8*)

%t1 = type {i32, i32, i32}

define void @test1() {
; CHECK: @test1
  %A = alloca %t1
  %A1 = getelementptr %t1* %A, i32 0, i32 0
  %A2 = getelementptr %t1* %A, i32 0, i32 1
  %A3 = getelementptr %t1* %A, i32 0, i32 2
  %B = bitcast i32* %A1 to i8*
  store i32 0, i32* %A1
  call void @llvm.lifetime.start(i64 -1, i8* %B)
  ret void
; CHECK-NEXT: ret void
}

define void @test2() {
; CHECK: @test2
  %A = alloca %t1
  %A1 = getelementptr %t1* %A, i32 0, i32 0
  %A2 = getelementptr %t1* %A, i32 0, i32 1
  %A3 = getelementptr %t1* %A, i32 0, i32 2
  %B = bitcast i32* %A2 to i8*
  store i32 0, i32* %A2
  call void @llvm.lifetime.start(i64 -1, i8* %B)
  %C = load i32* %A2
  ret void
; CHECK: ret void
}

define void @test3() {
; CHECK: @test3
  %A = alloca %t1
  %A1 = getelementptr %t1* %A, i32 0, i32 0
  %A2 = getelementptr %t1* %A, i32 0, i32 1
  %A3 = getelementptr %t1* %A, i32 0, i32 2
  %B = bitcast i32* %A2 to i8*
  store i32 0, i32* %A2
  call void @llvm.lifetime.start(i64 6, i8* %B)
  %C = load i32* %A2
  ret void
; CHECK-NEXT: ret void
}

define void @test4() {
; CHECK: @test4
  %A = alloca %t1
  %A1 = getelementptr %t1* %A, i32 0, i32 0
  %A2 = getelementptr %t1* %A, i32 0, i32 1
  %A3 = getelementptr %t1* %A, i32 0, i32 2
  %B = bitcast i32* %A2 to i8*
  store i32 0, i32* %A2
  call void @llvm.lifetime.start(i64 1, i8* %B)
  %C = load i32* %A2
  ret void
; CHECK-NEXT: ret void
}

%t2 = type {i32, [4 x i8], i32}

define void @test5() {
; CHECK: @test5
  %A = alloca %t2
; CHECK: alloca{{.*}}i8
; CHECK: alloca{{.*}}i8
; CHECK: alloca{{.*}}i8

  %A21 = getelementptr %t2* %A, i32 0, i32 1, i32 0
  %A22 = getelementptr %t2* %A, i32 0, i32 1, i32 1
  %A23 = getelementptr %t2* %A, i32 0, i32 1, i32 2
  %A24 = getelementptr %t2* %A, i32 0, i32 1, i32 3
; CHECK-NOT: store i8 1
  store i8 1, i8* %A21
  store i8 2, i8* %A22
  store i8 3, i8* %A23
  store i8 4, i8* %A24

  %A1 = getelementptr %t2* %A, i32 0, i32 0
  %A2 = getelementptr %t2* %A, i32 0, i32 1, i32 1
  %A3 = getelementptr %t2* %A, i32 0, i32 2
  store i8 0, i8* %A2
  call void @llvm.lifetime.start(i64 5, i8* %A2)
; CHECK: llvm.lifetime{{.*}}i64 1
; CHECK: llvm.lifetime{{.*}}i64 1
; CHECK: llvm.lifetime{{.*}}i64 1
  %C = load i8* %A2
  ret void
}

%t3 = type {[4 x i16], [4 x i8]}

define void @test6() {
; CHECK: @test6
  %A = alloca %t3
; CHECK: alloca i8
; CHECK: alloca i8
; CHECK: alloca i8

  %A11 = getelementptr %t3* %A, i32 0, i32 0, i32 0
  %A12 = getelementptr %t3* %A, i32 0, i32 0, i32 1
  %A13 = getelementptr %t3* %A, i32 0, i32 0, i32 2
  %A14 = getelementptr %t3* %A, i32 0, i32 0, i32 3
  store i16 11, i16* %A11
  store i16 12, i16* %A12
  store i16 13, i16* %A13
  store i16 14, i16* %A14
; CHECK-NOT: store i16 11
; CHECK-NOT: store i16 12
; CHECK-NOT: store i16 13
; CHECK-NOT: store i16 14

  %A21 = getelementptr %t3* %A, i32 0, i32 1, i32 0
  %A22 = getelementptr %t3* %A, i32 0, i32 1, i32 1
  %A23 = getelementptr %t3* %A, i32 0, i32 1, i32 2
  %A24 = getelementptr %t3* %A, i32 0, i32 1, i32 3
  store i8 21, i8* %A21
  store i8 22, i8* %A22
  store i8 23, i8* %A23
  store i8 24, i8* %A24
; CHECK: store i8 21
; CHECK: store i8 22
; CHECK: store i8 23
; CHECK-NOT: store i8 24

  %B = bitcast i16* %A13 to i8*
  call void @llvm.lifetime.start(i64 7, i8* %B)
; CHECK: lifetime.start{{.*}}i64 1
; CHECK: lifetime.start{{.*}}i64 1
; CHECK: lifetime.start{{.*}}i64 1

  ret void
}
