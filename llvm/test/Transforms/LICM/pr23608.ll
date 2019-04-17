; RUN: opt -S -licm %s | FileCheck %s
; ModuleID = '../pr23608.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.PyFrameObject = type { i32 }

@a = common global %struct.PyFrameObject* null, align 8
@__msan_origin_tls = external thread_local(initialexec) global i32

define void @fn1() {
entry:
  br label %indirectgoto

while.cond:                                       ; preds = %indirectgoto, %bb15
  %tmp = load %struct.PyFrameObject*, %struct.PyFrameObject** @a, align 8
  %_msld = load i64, i64* inttoptr (i64 and (i64 ptrtoint (%struct.PyFrameObject** @a to i64), i64 -70368744177665) to i64*), align 8
  %tmp1 = load i32, i32* inttoptr (i64 add (i64 and (i64 ptrtoint (%struct.PyFrameObject** @a to i64), i64 -70368744177665), i64 35184372088832) to i32*), align 8
  %f_iblock = getelementptr inbounds %struct.PyFrameObject, %struct.PyFrameObject* %tmp, i64 0, i32 0
  br label %bb2

bb:                                               ; preds = %while.cond
  call void @__msan_warning_noreturn()
  unreachable

bb2:                                              ; preds = %while.cond
  %tmp3 = load i32, i32* %f_iblock, align 4
  %tmp4 = ptrtoint i32* %f_iblock to i64
  %tmp8 = inttoptr i64 %tmp4 to i32*
  %tobool = icmp eq i64 %tmp4, 0
  br i1 %tobool, label %bb13, label %bb15

bb13:                                             ; preds = %bb2
; CHECK-LABEL: bb13:
; CHECK: %tmp8.le = inttoptr
  %.lcssa7 = phi i32* [ %tmp8, %bb2 ]
  call void @__msan_warning_noreturn()
  unreachable

bb15:                                             ; preds = %bb2
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %bb15
  ret void

indirectgoto:                                     ; preds = %indirectgoto, %entry
  indirectbr i8* null, [label %indirectgoto, label %while.cond]
}

declare void @__msan_warning_noreturn()
