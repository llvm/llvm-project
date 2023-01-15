; RUN: opt -S -passes=consthoist < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; PR18626
define ptr @test1(i1 %cmp, ptr %tmp) {
entry:
  call void @foo(ptr inttoptr (i64 68719476735 to ptr))
  br i1 %cmp, label %if.end, label %return

if.end:                                           ; preds = %bb1
  call void @foo(ptr inttoptr (i64 68719476736 to ptr))
  br label %return

return:
  %retval.0 = phi ptr [ null, %entry ], [ inttoptr (i64 68719476736 to ptr), %if.end ]
  store i64 1172321806, ptr %tmp
  ret ptr %retval.0

; CHECK-LABEL: @test1
; CHECK: if.end:
; CHECK: %2 = inttoptr i64 %const to ptr
; CHECK-NEXT: br
; CHECK: return:
; CHECK-NEXT: %retval.0 = phi ptr [ null, %entry ], [ %2, %if.end ]
}

define void @test2(i1 %cmp, ptr %tmp) {
entry:
  call void @foo(ptr inttoptr (i64 68719476736 to ptr))
  br i1 %cmp, label %if.end, label %return

if.end:                                           ; preds = %bb1
  call void @foo(ptr inttoptr (i64 68719476736 to ptr))
  br label %return

return:
  store ptr inttoptr (i64 68719476735 to ptr), ptr %tmp
  ret void

; CHECK-LABEL: @test2
; CHECK: return:
; CHECK-NEXT: %const_mat = add i64 %const, -1
; CHECK-NEXT: inttoptr i64 %const_mat to ptr
}

declare void @foo(ptr)

; PR18768
define i32 @test3(i1 %c) {
entry:
  br i1 %c, label %if.then, label %if.end3

if.then:                                          ; preds = %entry
  br label %if.end3

if.end3:                                          ; preds = %if.then, %entry
  %d.0 = phi ptr [ inttoptr (i64 985162435264511 to ptr), %entry ], [ null, %if.then ]
  %cmp4 = icmp eq ptr %d.0, inttoptr (i64 985162435264511 to ptr)
  %cmp6 = icmp eq ptr %d.0, inttoptr (i64 985162418487296 to ptr)
  %or = or i1 %cmp4, %cmp6
  br i1 %or, label %if.then8, label %if.end9

if.then8:                                         ; preds = %if.end3
  ret i32 1

if.end9:                                          ; preds = %if.then8, %if.end3
  ret i32 undef
}

; <rdar://problem/16394449>
define i64 @switch_test1(i64 %a) {
; CHECK-LABEL: @switch_test1
; CHECK: %0 = phi i64 [ %const, %case2 ], [ %const_mat, %Entry ], [ %const_mat, %Entry ]
Entry:
  %sel = add i64 %a, 4519019440
  switch i64 %sel, label %fail [
    i64 462, label %continuation
    i64 449, label %case2
    i64 443, label %continuation
  ]

case2:
  br label %continuation

continuation:
  %0 = phi i64 [ 4519019440, %case2 ], [ 4519019460, %Entry ], [ 4519019460, %Entry ]
  ret i64 0;

fail:
  ret i64 -1;
}

define i64 @switch_test2(i64 %a) {
; CHECK-LABEL: @switch_test2
; CHECK: %2 = phi ptr [ %1, %case2 ], [ %0, %Entry ], [ %0, %Entry ]
Entry:
  %sel = add i64 %a, 4519019440
  switch i64 %sel, label %fail [
    i64 462, label %continuation
    i64 449, label %case2
    i64 443, label %continuation
  ]

case2:
  br label %continuation

continuation:
  %0 = phi ptr [ inttoptr(i64 4519019440 to ptr), %case2 ], [ inttoptr(i64 4519019460 to ptr), %Entry ], [ inttoptr(i64 4519019460 to ptr), %Entry ]
  ret i64 0;

fail:
  ret i64 -1;
}

