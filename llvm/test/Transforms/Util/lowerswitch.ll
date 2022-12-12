; RUN: opt -passes=lowerswitch -S < %s | FileCheck %s

; Test that we don't crash and have a different basic block for each incoming edge.
define void @test0(i32 %mode) {
; CHECK-LABEL: @test0
;
; CHECK: icmp eq i32 %mode, 4
; CHECK-NEXT: label %BB3, label %BB2
;
; CHECK: icmp eq i32 %mode, 2
; CHECK-NEXT: label %BB3, label %BB2
;
; CHECK: icmp eq i32 %mode, 0
; CHECK-NEXT: label %BB3, label %BB2
;
; CHECK: %merge = phi i64 [ 1, %BB3 ], [ 0, %LeafBlock ], [ 0, %LeafBlock1 ], [ 0, %LeafBlock3 ]
BB1:
  switch i32 %mode, label %BB2 [
    i32 3, label %BB2
    i32 5, label %BB2
    i32 0, label %BB3
    i32 2, label %BB3
    i32 4, label %BB3
  ]

BB2:
  %merge = phi i64 [ 1, %BB3 ], [ 0, %BB1 ], [ 0, %BB1 ], [ 0, %BB1 ]
  ret void

BB3:
  br label %BB2
}

; Test switch cases that are merged into a single case during lowerswitch
; (take 84 and 85 below) - check that the number of incoming phi values match
; the number of branches.
define void @test1(i32 %mode) {
; CHECK-LABEL: @test1
entry:
  br label %bb1

bb1:
  switch i32 %mode, label %bb1 [
    i32 84, label %bb3
    i32 85, label %bb3
    i32 86, label %bb2
    i32 78, label %exit
    i32 99, label %bb3
  ]

bb2:
  br label %bb3

bb3:
; CHECK-LABEL: bb3
; CHECK: %tmp = phi i32 [ 1, %NodeBlock ], [ 0, %bb2 ], [ 1, %LeafBlock3 ]
  %tmp = phi i32 [ 1, %bb1 ], [ 0, %bb2 ], [ 1, %bb1 ], [ 1, %bb1 ]
; CHECK-NEXT: %tmp2 = phi i32 [ 2, %NodeBlock ], [ 5, %bb2 ], [ 2, %LeafBlock3 ]
  %tmp2 = phi i32 [ 2, %bb1 ], [ 2, %bb1 ], [ 5, %bb2 ], [ 2, %bb1 ]
  br label %exit

exit:
  ret void
}

; Test that we don't crash.
define void @test2(i32 %mode) {
; CHECK-LABEL: @test2
  br i1 undef, label %1, label %._crit_edge

; <label>:1                                       ; preds = %0
  switch i32 %mode, label %33 [
    i32 2, label %2
    i32 3, label %3
    i32 4, label %4
    i32 5, label %5
    i32 6, label %6
    i32 7, label %7
    i32 8, label %8
    i32 9, label %9
    i32 10, label %10
    i32 11, label %14
    i32 12, label %18
    i32 13, label %22
    i32 14, label %26
    i32 15, label %27
    i32 16, label %34
    i32 17, label %34
    i32 18, label %34
    i32 19, label %34
    i32 22, label %34
    i32 20, label %31
    i32 21, label %32
  ]

; <label>:2                                       ; preds = %1
  br label %34

; <label>:3                                       ; preds = %1
  br label %34

; <label>:4                                       ; preds = %1
  br label %34

; <label>:5                                       ; preds = %1
  br label %34

; <label>:6                                       ; preds = %1
  br label %34

; <label>:7                                       ; preds = %1
  br label %34

; <label>:8                                       ; preds = %1
  br label %34

; <label>:9                                       ; preds = %1
  br label %34

; <label>:10                                      ; preds = %1
  br i1 undef, label %11, label %12

; <label>:11                                      ; preds = %10
  br label %13

; <label>:12                                      ; preds = %10
  br label %13

; <label>:13                                      ; preds = %12, %11
  br label %34

; <label>:14                                      ; preds = %1
  br i1 undef, label %15, label %16

; <label>:15                                      ; preds = %14
  br label %17

; <label>:16                                      ; preds = %14
  br label %17

; <label>:17                                      ; preds = %16, %15
  br label %34

; <label>:18                                      ; preds = %1
  br i1 undef, label %19, label %20

; <label>:19                                      ; preds = %18
  br label %21

; <label>:20                                      ; preds = %18
  br label %21

; <label>:21                                      ; preds = %20, %19
  br label %34

; <label>:22                                      ; preds = %1
  br i1 undef, label %23, label %24

; <label>:23                                      ; preds = %22
  br label %25

; <label>:24                                      ; preds = %22
  br label %25

; <label>:25                                      ; preds = %24, %23
  br label %34

; <label>:26                                      ; preds = %1
  br label %34

; <label>:27                                      ; preds = %1
  br i1 undef, label %28, label %29

; <label>:28                                      ; preds = %27
  br label %30

; <label>:29                                      ; preds = %27
  br label %30

; <label>:30                                      ; preds = %29, %28
  br label %34

; <label>:31                                      ; preds = %1
  br label %34

; <label>:32                                      ; preds = %1
  br label %34

; <label>:33                                      ; preds = %1
  br label %34

; <label>:34                                      ; preds = %33, %32, %31, %30, %26, %25, %21, %17, %13, %9, %8, %7, %6, %5, %4, %3, %2, %1, %1, %1, %1, %1
  %o.0 = phi float [ undef, %33 ], [ undef, %32 ], [ undef, %31 ], [ undef, %30 ], [ undef, %26 ], [ undef, %25 ], [ undef, %21 ], [ undef, %17 ], [ undef, %13 ], [ undef, %9 ], [ undef, %8 ], [ undef, %7 ], [ undef, %6 ], [ undef, %5 ], [ undef, %4 ], [ undef, %3 ], [ undef, %2 ], [ undef, %1 ], [ undef, %1 ], [ undef, %1 ], [ undef, %1 ], [ undef, %1 ]
  br label %._crit_edge

._crit_edge:                                      ; preds = %34, %0
  ret void
}

; Test that the PHI node in for.cond should have one entry for each predecessor
; of its parent basic block after lowerswitch merged several cases into a new
; default block.
define void @test3(i32 %mode) {
; CHECK-LABEL: @test3
entry:
  br label %lbl1

lbl1:                                             ; preds = %cleanup, %entry
  br label %lbl2

lbl2:                                             ; preds = %cleanup, %lbl1
  br label %for.cond

for.cond:                                         ; preds = %cleanup, %cleanup, %lbl2
; CHECK: for.cond:
; CHECK: phi i16 [ undef, %lbl2 ], [ %b.3, %LeafBlock ], [ %b.3, %LeafBlock1 ]{{$}}
; CHECK: for.cond1:
  %b.2 = phi i16 [ undef, %lbl2 ], [ %b.3, %cleanup ], [ %b.3, %cleanup ]
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.cond
  %b.3 = phi i16 [ %b.2, %for.cond ], [ undef, %for.inc ]
  %tobool = icmp ne i16 %b.3, 0
  br i1 %tobool, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond1
  br i1 undef, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  br label %cleanup

for.inc:                                          ; preds = %for.body
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br i1 undef, label %if.then4, label %for.body7

if.then4:                                         ; preds = %for.end
  br label %cleanup

for.body7:                                        ; preds = %for.end
  br label %cleanup

cleanup:                                          ; preds = %for.body7, %if.then4, %if.then
  switch i32 %mode, label %unreachable [
    i32 0, label %for.cond
    i32 2, label %lbl1
    i32 5, label %for.cond
    i32 3, label %lbl2
  ]

unreachable:                                      ; preds = %cleanup
  unreachable
}

; Test that the PHI node in cleanup17 is removed as the switch default block is
; not reachable.
define void @test4(i32 %mode) {
; CHECK-LABEL: @test4
entry:
  switch i32 %mode, label %cleanup17 [
    i32 0, label %return
    i32 9, label %return
  ]

cleanup17:
; CHECK: cleanup17:
; CHECK-NOT: phi i16 [ undef, %entry ]
; CHECK: return:

  %retval.4 = phi i16 [ undef, %entry ]
  unreachable

return:
  ret void
}

; Test that the PHI node in for.inc is updated correctly as the switch is
; replaced with a single branch to for.inc
define void @test5(i32 %mode) {
; CHECK-LABEL: @test5
entry:
  br i1 undef, label %cleanup10, label %cleanup10.thread

cleanup10.thread:
  br label %for.inc

cleanup10:
  switch i32 %mode, label %unreachable [
    i32 0, label %for.inc
    i32 4, label %for.inc
  ]

for.inc:
; CHECK: for.inc:
; CHECK-NEXT: phi i16 [ 0, %cleanup10.thread ], [ undef, %cleanup10 ]
%0 = phi i16 [ undef, %cleanup10 ], [ 0, %cleanup10.thread ], [ undef, %cleanup10 ]
  unreachable

unreachable:
  unreachable
}
