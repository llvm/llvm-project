; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix=AIX64
; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix=AIX32

define signext i32 @test1(i32 signext %i, ptr nocapture %Func, ptr nocapture %Func2) {
entry:
; CHECK-LABEL: test1:
; CHECK:    std 2, 24(1)
; CHECK-NOT:    std 2, 24(1)

; AIX64-LABEL: test1:
; AIX64: std 2, 40(1)
; AIX64-NOT: std 2, 40(1)

; AIX32-LABEL: test1:
; AIX32: stw 2, 20(1)
; AIX32-NOT: std 2, 20(1)
  %call = tail call signext i32 %Func(i32 signext %i)
  %call1 = tail call signext i32 %Func2(i32 signext %i)
  %add2 = add nsw i32 %call1, %call
  ret i32 %add2
}

define signext i32 @test2(i32 signext %i, i32 signext %j, ptr nocapture %Func, ptr nocapture %Func2) {
entry:
; CHECK-LABEL: test2:
; CHECK:    std 2, 24(1)
; CHECK-NOT:    std 2, 24(1)

; AIX64-LABEL: test2:
; AIX64: std 2, 40(1)
; AIX64-NOT: std 2, 40(1)

; AIX32-LABEL: test2:
; AIX32: stw 2, 20(1)
; AIX32-NOT: std 2, 20(1)
  %call = tail call signext i32 %Func(i32 signext %i)
  %tobool = icmp eq i32 %j, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call signext i32 %Func(i32 signext %i)
  %add2 = add nsw i32 %call1, %call
  %call3 = tail call signext i32 %Func2(i32 signext %i)
  %add4 = add nsw i32 %add2, %call3
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %Sum.0 = phi i32 [ %add4, %if.then ], [ %call, %entry ]
  %call5 = tail call signext i32 %Func(i32 signext %i)
  %add6 = add nsw i32 %call5, %Sum.0
  ret i32 %add6
}

; Check for multiple TOC saves with if then else where neither dominates the other.
define signext i32 @test3(i32 signext %i, ptr nocapture %Func, ptr nocapture %Func2) {
; CHECK-LABEL: test3:
; CHECK:    std 2, 24(1)
; CHECK-NOT:    std 2, 24(1)

; AIX64-LABEL: test3:
; AIX64-COUNT-3: std 2, 40(1)

; AIX32-LABEL: test3:
; AIX32-COUNT-3: stw 2, 20(1)
entry:
  %tobool = icmp eq i32 %i, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call signext i32 %Func(i32 signext %i)
  br label %if.end

if.else:                                          ; preds = %entry
  %call1 = tail call signext i32 %Func2(i32 signext 0)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %Sum.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  %call3 = tail call signext i32 %Func(i32 signext %i)
  %add4 = add nsw i32 %call3, %Sum.0
  ret i32 %add4
}

define signext i32 @test4(i32 signext %i, ptr nocapture %Func, ptr nocapture %Func2) {
; CHECK-LABEL: test4:
; CHECK:    std 2, 24(1)
; CHECK-NOT:    std 2, 24(1)

; AIX64-LABEL: test4:
; AIX64: std 2, 40(1)
; AIX64-NOT: std 2, 40(1)

; AIX32-LABEL: test4:
; AIX32: stw 2, 20(1)
; AIX32-NOT: std 2, 20(1)
entry:
  %call = tail call signext i32 %Func(i32 signext %i)
  %tobool = icmp eq i32 %i, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call signext i32 %Func(i32 signext %i)
  br label %if.end

if.else:                                          ; preds = %entry
  %call3 = tail call signext i32 %Func2(i32 signext 0)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %call1.pn = phi i32 [ %call1, %if.then ], [ %call3, %if.else ]
  %Sum.0 = add nsw i32 %call1.pn, %call
  ret i32 %Sum.0
}

; Check for multiple TOC saves with if then where neither is redundant.
define signext i32 @test5(i32 signext %i, ptr nocapture %Func, ptr nocapture readnone %Func2) {
entry:
; CHECK-LABEL: test5:
; CHECK:    std 2, 24(1)

; AIX64-LABEL: test5:
; AIX64: std 2, 40(1)

; AIX32-LABEL: test5:
; AIX32: stw 2, 20(1)
  %tobool = icmp eq i32 %i, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call signext i32 %Func(i32 signext %i)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %Sum.0 = phi i32 [ %call, %if.then ], [ 0, %entry ]
  %call1 = tail call signext i32 %Func(i32 signext %i)
  %add2 = add nsw i32 %call1, %Sum.0
  ret i32 %add2
}

; Check for multiple TOC saves if there are dynamic allocations on the stack.
define signext i32 @test6(i32 signext %i, ptr nocapture %Func, ptr nocapture %Func2) {
entry:
; CHECK-LABEL: test6:
; CHECK:    std 2, 24(1)
; CHECK:    std 2, 24(1)

; AIX64-LABEL: test6:
; AIX64: std 2, 40(1)
; AIX64: std 2, 40(1)

; AIX32-LABEL: test6:
; AIX32: stw 2, 20(1)
; AIX32: stw 2, 20(1)
  %conv = sext i32 %i to i64
  %0 = alloca i8, i64 %conv, align 16
  %call = tail call signext i32 %Func(i32 signext %i)
  call void @useAlloca(ptr nonnull %0, i32 signext %call)
  %call1 = call signext i32 %Func2(i32 signext %i)
  %add2 = add nsw i32 %call1, %call
  ret i32 %add2
}

declare void @useAlloca(ptr, i32 signext)
