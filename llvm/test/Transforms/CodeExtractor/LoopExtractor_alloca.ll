; RUN: opt -passes=debugify,loop-simplify,loop-extract -S < %s | FileCheck %s

; This tests 2 cases:
; 1. loop1 should be extracted into a function, without extracting %v1 alloca.
; 2. loop2 should be extracted into a function, with the %v2 alloca.
;
; This used to produce an invalid IR, where `memcpy` will have a reference to
; the, now, external value (local to the extracted loop function).

; CHECK-LABEL: define void @test()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v1 = alloca i32
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i32* %v1
; CHECK-NEXT:   %p1 = bitcast i32* %v1 to i8*
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i8* %p1,
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 undef, i8* %p1, i64 4, i1 true)

; CHECK-LABEL: define internal void @test.loop2()
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   %v2 = alloca i32
; CHECK-NEXT:   %p2 = bitcast i32* %v2 to i8*

; CHECK-LABEL: define internal void @test.loop1(i8* %p1)
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   br

define void @test() {
entry:
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  %p1 = bitcast i32* %v1 to i8*
  %p2 = bitcast i32* %v2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 undef, i8* %p1, i64 4, i1 true)
  br label %loop1

loop1:
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %p1)
  %r1 = call i32 @foo(i8* %p1)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %p1)
  %cmp1 = icmp ne i32 %r1, 0
  br i1 %cmp1, label %loop1, label %loop2

loop2:
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %p2)
  %r2 = call i32 @foo(i8* %p2)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %p2)
  %cmp2 = icmp ne i32 %r2, 0
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

declare i32 @foo(i8*)

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
