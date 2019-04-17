; RUN: opt -S -jump-threading < %s | FileCheck %s

; Check that the heuristic for avoiding accidental introduction of irreducible
; loops doesn't also prevent us from threading simple constructs where this
; isn't a problem.

declare void @opaque_body()

define void @jump_threading_loopheader() {
; CHECK-LABEL: @jump_threading_loopheader
top:
    br label %entry

entry:
    %ind = phi i32 [0, %top], [%nextind, %latch]
    %nextind = add i32 %ind, 1
    %cmp = icmp ule i32 %ind, 10
; CHECK: br i1 %cmp, label %latch, label %exit
    br i1 %cmp, label %body, label %latch

body:
    call void @opaque_body()
; CHECK: br label %entry
    br label %latch

latch:
    %cond = phi i2 [1, %entry], [2, %body]
    switch i2 %cond, label %unreach [
        i2 2, label %entry
        i2 1, label %exit
    ]

unreach:
    unreachable

exit:
    ret void
}

; We also need to check the opposite order of the branches, in the switch
; instruction because jump-threading relies on that to decide which edge to
; try to thread first.
define void @jump_threading_loopheader2() {
; CHECK-LABEL: @jump_threading_loopheader2
top:
    br label %entry

entry:
    %ind = phi i32 [0, %top], [%nextind, %latch]
    %nextind = add i32 %ind, 1
    %cmp = icmp ule i32 %ind, 10
; CHECK: br i1 %cmp, label %exit, label %latch
    br i1 %cmp, label %body, label %latch

body:
    call void @opaque_body()
; CHECK: br label %entry
    br label %latch

latch:
    %cond = phi i2 [1, %entry], [2, %body]
    switch i2 %cond, label %unreach [
        i2 1, label %entry
        i2 2, label %exit
    ]

unreach:
    unreachable

exit:
    ret void
}

; Check if we can handle undef branch condition.
define void @jump_threading_loopheader3() {
; CHECK-LABEL: @jump_threading_loopheader3
top:
    br label %entry

entry:
    %ind = phi i32 [0, %top], [%nextind, %latch]
    %nextind = add i32 %ind, 1
    %cmp = icmp ule i32 %ind, 10
; CHECK: br i1 %cmp, label %latch, label %exit
    br i1 %cmp, label %body, label %latch

body:
    call void @opaque_body()
; CHECK: br label %entry
    br label %latch

latch:
   %phi = phi i32 [undef, %entry], [0, %body]
   %cmp1 = icmp eq i32 %phi, 0
   br i1 %cmp1, label %entry, label %exit

exit:
    ret void
}
