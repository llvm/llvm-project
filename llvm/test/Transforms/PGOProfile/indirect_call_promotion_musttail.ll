; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8

define ptr @func1() {
  ret ptr null
}

define ptr @func2() {
  ret ptr null
}

define ptr @func3() {
  ret ptr null
}

define ptr @func4() {
  ret ptr null
}

define ptr @bar() {
entry:
  %tmp = load ptr, ptr @foo, align 8
; ICALL-PROM:   [[CMP1:%[0-9]+]] = icmp eq ptr %tmp, @func4
; ICALL-PROM:   br i1 [[CMP1]], label %if.true.direct_targ, label %[[L1:[0-9]+]], !prof [[BRANCH_WEIGHT1:![0-9]+]]
; ICALL-PROM: if.true.direct_targ:
; ICALL-PROM:   [[DIRCALL_RET1:%[0-9]+]] = musttail call ptr @func4()
; ICALL-PROM:   ret ptr [[DIRCALL_RET1]]
; ICALL-PROM: [[L1]]:
; ICALL-PROM:   [[CMP2:%[0-9]+]] = icmp eq ptr %tmp, @func2
; ICALL-PROM:   br i1 [[CMP2]], label %if.true.direct_targ1, label %[[L2:[0-9]+]], !prof [[BRANCH_WEIGHT2:![0-9]+]]
; ICALL-PROM: if.true.direct_targ1:
; ICALL-PROM:   [[DIRCALL_RET2:%[0-9]+]] = musttail call ptr @func2()
; ICALL-PROM:   ret ptr [[DIRCALL_RET2]]
; ICALL-PROM: [[L2]]:
; ICALL-PROM:   [[CMP3:%[0-9]+]] = icmp eq ptr %tmp, @func3
; ICALL-PROM:   br i1 [[CMP3]], label %if.true.direct_targ2, label %[[L3:[0-9]+]], !prof [[BRANCH_WEIGHT3:![0-9]+]]
; ICALL-PROM: if.true.direct_targ2:
; ICALL-PROM:   [[DIRCALL_RET3:%[0-9]+]] = musttail call ptr @func3()
; ICALL-PROM:   ret ptr [[DIRCALL_RET3]]
; ICALL-PROM: [[L3]]:
; ICALL-PROM:   %call = musttail call ptr %tmp()
; ICALL-PROM:   ret ptr %call
  %call = musttail call ptr %tmp(), !prof !1
  ret ptr %call
}

define ptr @bar2() {
entry:
  %tmp = load ptr, ptr @foo, align 8
; ICALL-PROM:   [[CMP1:%[0-9]+]] = icmp eq ptr %tmp, @func4
; ICALL-PROM:   br i1 [[CMP1]], label %if.true.direct_targ, label %[[L4:[0-9]+]], !prof [[BRANCH_WEIGHT4:![0-9]+]]
; ICALL-PROM: if.true.direct_targ:
; ICALL-PROM:   [[DIRCALL_RET1:%[0-9]+]] = musttail call ptr @func4()
; ICALL-PROM:   ret ptr [[DIRCALL_RET1]]
; ICALL-PROM: [[L4]]:
; ICALL-PROM:   %call = musttail call ptr %tmp()
; ICALL-PROM:   ret ptr %call
  %call = musttail call ptr %tmp(), !prof !2
  ret ptr %call
}

!1 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410, i64 -6929281286627296573, i64 150, i64 -2545542355363006406, i64 10}
!2 = !{!"VP", i32 0, i64 100, i64 7651369219802541373, i64 100}

; ICALL-PROM: [[BRANCH_WEIGHT1]] = !{!"branch_weights", i32 1030, i32 570}
; ICALL-PROM: [[BRANCH_WEIGHT2]] = !{!"branch_weights", i32 410, i32 160}
; ICALL-PROM: [[BRANCH_WEIGHT3]] = !{!"branch_weights", i32 150, i32 10}
; ICALL-PROM: [[BRANCH_WEIGHT4]] = !{!"branch_weights", i32 100, i32 0}
