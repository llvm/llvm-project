; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

; This load is safe to speculate, as it's from a safe offset
; within an alloca.

; CHECK-LABEL: @yes(
; CHECK-NOT: br

define void @yes(i1 %c) nounwind {
entry:
  %a = alloca [4 x ptr], align 8
  %__a.addr = getelementptr [4 x ptr], ptr %a, i64 0, i64 3
  call void @frob(ptr %__a.addr)
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %tmp5 = load ptr, ptr %__a.addr, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi ptr [ undef, %if.then ], [ %tmp5, %if.end ]
  ret void
}

; CHECK-LABEL: @no0(
; CHECK: br i1 %c

define void @no0(i1 %c) nounwind {
entry:
  %a = alloca [4 x ptr], align 8
  %__a.addr = getelementptr [4 x ptr], ptr %a, i64 0, i64 4
  call void @frob(ptr %__a.addr)
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %tmp5 = load ptr, ptr %__a.addr, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi ptr [ undef, %if.then ], [ %tmp5, %if.end ]
  ret void
}

; CHECK-LABEL: @no1(
; CHECK: br i1 %c

define void @no1(i1 %c, i64 %n) nounwind {
entry:
  %a = alloca [4 x ptr], align 8
  %__a.addr = getelementptr [4 x ptr], ptr %a, i64 0, i64 %n
  call void @frob(ptr %__a.addr)
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %tmp5 = load ptr, ptr %__a.addr, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi ptr [ undef, %if.then ], [ %tmp5, %if.end ]
  ret void
}

; CHECK-LABEL: @no2(
; CHECK: br i1 %c

define void @no2(i1 %c, i64 %n) nounwind {
entry:
  %a = alloca [4 x ptr], align 8
  %__a.addr = getelementptr [4 x ptr], ptr %a, i64 1, i64 0
  call void @frob(ptr %__a.addr)
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %tmp5 = load ptr, ptr %__a.addr, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi ptr [ undef, %if.then ], [ %tmp5, %if.end ]
  ret void
}

declare void @frob(ptr nocapture %p)
