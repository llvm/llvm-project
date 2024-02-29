; RUN: opt -passes=debugify,loop-simplify,loop-extract -S < %s | FileCheck %s
; RUN: opt -passes=debugify,loop-simplify,loop-extract -S < %s --try-experimental-debuginfo-iterators | FileCheck %s

; This tests 2 cases:
; 1. loop1 should be extracted into a function, without extracting %v1 alloca.
; 2. loop2 should be extracted into a function, with the %v2 alloca.
;
; This used to produce an invalid IR, where `memcpy` will have a reference to
; the, now, external value (local to the extracted loop function).

; CHECK-LABEL: define void @test()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v1 = alloca i32
; CHECK-NEXT:   call void @llvm.dbg.value(metadata ptr %v1
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 undef, ptr %v1, i64 4, i1 true)

; CHECK-LABEL: define internal void @test.loop2()
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   %v2 = alloca i32

; CHECK-LABEL: define internal void @test.loop1(ptr %v1)
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   br

define void @test() {
entry:
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 undef, ptr %v1, i64 4, i1 true)
  br label %loop1

loop1:
  call void @llvm.lifetime.start.p0(i64 4, ptr %v1)
  %r1 = call i32 @foo(ptr %v1)
  call void @llvm.lifetime.end.p0(i64 4, ptr %v1)
  %cmp1 = icmp ne i32 %r1, 0
  br i1 %cmp1, label %loop1, label %loop2

loop2:
  call void @llvm.lifetime.start.p0(i64 4, ptr %v2)
  %r2 = call i32 @foo(ptr %v2)
  call void @llvm.lifetime.end.p0(i64 4, ptr %v2)
  %cmp2 = icmp ne i32 %r2, 0
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

declare i32 @foo(ptr)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
