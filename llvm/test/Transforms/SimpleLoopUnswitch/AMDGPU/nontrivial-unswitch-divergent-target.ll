; RUN: opt -mtriple=amdgcn-- -passes='loop(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-- -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-- -passes='simple-loop-unswitch<nontrivial>' -verify-memoryssa -S < %s | FileCheck %s

declare i32 @a()
declare i32 @b()
declare i32 @c()

; Non-trivial loop unswitching where there are two distinct trivial
; conditions to unswitch within the loop. The conditions are divergent
; and should not unswitch.
define void @test1(ptr %ptr, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test1(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b
; CHECK: loop_begin:
; CHECK-NEXT: br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  %unused.a = call i32 @a()
  br label %latch
; CHECK: loop_a:
; CHECK-NEXT: %unused.a = call i32 @a()
; CHECK-NEXT: br label %latch

loop_b:
  br i1 %cond2, label %loop_b_a, label %loop_b_b
; CHECK: loop_b:
; CHECK-NEXT: br i1 %cond2, label %loop_b_a, label %loop_b_b

loop_b_a:
  %unused.b = call i32 @b()
  br label %latch
; CHECK: loop_b_a:
; CHECK-NEXT: %unused.b = call i32 @b()
; CHECK-NEXT: br label %latch

loop_b_b:
  %unused.c = call i32 @c()
  br label %latch
; CHECK: loop_b_b:
; CHECK-NEXT: %unused.c = call i32 @c()
; CHECK-NEXT: br label %latch

latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit
; CHECK: latch:
; CHECK-NEXT: %v = load i1, ptr %ptr
; CHECK-NEXT: br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK: loop_exit:
; CHECK-NEXT: ret void
}

; Non-trivial loop unswitching where there are two distinct trivial
; conditions to unswitch within the loop. The conditions are known to
; be uniform, so it should be unswitchable. However, unswitch
; currently does not make use of UniformityAnalysis.
define amdgpu_kernel void @test1_uniform(ptr %ptr, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test1_uniform(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b
; CHECK: loop_begin:
; CHECK-NEXT: br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  %unused.a = call i32 @a()
  br label %latch
; CHECK: loop_a:
; CHECK-NEXT: %unused.a = call i32 @a()
; CHECK-NEXT: br label %latch

loop_b:
  br i1 %cond2, label %loop_b_a, label %loop_b_b
; CHECK: loop_b:
; CHECK-NEXT: br i1 %cond2, label %loop_b_a, label %loop_b_b

loop_b_a:
  %unused.b = call i32 @b()
  br label %latch
; CHECK: loop_b_a:
; CHECK-NEXT: %unused.b = call i32 @b()
; CHECK-NEXT: br label %latch

loop_b_b:
  %unused.c = call i32 @c()
  br label %latch
; CHECK: loop_b_b:
; CHECK-NEXT: %unused.c = call i32 @c()
; CHECK-NEXT: br label %latch

latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit
; CHECK: latch:
; CHECK-NEXT: %v = load i1, ptr %ptr
; CHECK-NEXT: br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK: loop_exit:
; CHECK-NEXT: ret void
}

; Non-trivial loop unswitching where there are two distinct trivial
; conditions to unswitch within the loop. There is no divergence
; because it's assumed it can only execute with a workgroup of size 1.
define void @test1_single_lane_execution(ptr %ptr, i1 %cond1, i1 %cond2) #0 {
; CHECK-LABEL: @test1_single_lane_execution(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  br i1 %cond2, label %loop_b_a, label %loop_b_b
; The second unswitched condition.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br i1 %cond2, label %entry.split.split.us, label %entry.split.split

loop_b_a:
  call i32 @b()
  br label %latch
; The 'loop_b_a' unswitched loop.
;
; CHECK:       entry.split.split.us:
; CHECK-NEXT:    br label %loop_begin.us1
;
; CHECK:       loop_begin.us1:
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    br label %loop_b_a.us
;
; CHECK:       loop_b_a.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us2
;
; CHECK:       latch.us2:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us1, label %loop_exit.split.split.us
;
; CHECK:       loop_exit.split.split.us:
; CHECK-NEXT:    br label %loop_exit.split

loop_b_b:
  call i32 @c()
  br label %latch
; The 'loop_b_b' unswitched loop.
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    br label %loop_b_b
;
; CHECK:       loop_b_b:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %latch
;
; CHECK:       latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split.split
;
; CHECK:       loop_exit.split.split:
; CHECK-NEXT:    br label %loop_exit.split

latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" }
