; RUN: opt -S -instcombine < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@f.a = private unnamed_addr constant [1 x i32] [i32 12], align 4
@f.b = private unnamed_addr constant [1 x i32] [i32 55], align 4
@f.c = linkonce unnamed_addr alias [1 x i32], [1 x i32]* @f.b

define signext i32 @test1(i32 signext %x) #0 {
entry:
  %idxprom = sext i32 %x to i64
  %arrayidx = getelementptr inbounds [1 x i32], [1 x i32]* @f.a, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @test1
; CHECK: ret i32 12
}

declare void @foo(i64* %p)
define void @test2(i32 signext %x, i64 %v) #0 {
entry:
  %p = alloca i64
  %idxprom = sext i32 %x to i64
  %arrayidx = getelementptr inbounds i64, i64* %p, i64 %idxprom
  store i64 %v, i64* %arrayidx
  call void @foo(i64* %p)
  ret void

; CHECK-LABEL: @test2
; CHECK: %p = alloca i64
; CHECK: store i64 %v, i64* %p
; CHECK: ret void
}

define signext i32 @test3(i32 signext %x, i1 %y) #0 {
entry:
  %idxprom = sext i32 %x to i64
  %p = select i1 %y, [1 x i32]* @f.a, [1 x i32]* @f.b
  %arrayidx = getelementptr inbounds [1 x i32], [1 x i32]* %p, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @test3
; CHECK: getelementptr inbounds [1 x i32], [1 x i32]* %p, i64 0, i64 0
}

define signext i32 @test4(i32 signext %x, i1 %y) #0 {
entry:
  %idxprom = sext i32 %x to i64
  %arrayidx = getelementptr inbounds [1 x i32], [1 x i32]* @f.c, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @test4
; CHECK: getelementptr inbounds [1 x i32], [1 x i32]* @f.c, i64 0, i64 %idxprom
}

attributes #0 = { nounwind readnone }

