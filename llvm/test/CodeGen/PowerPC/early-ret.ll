; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo(ptr %P) #0 {
entry:
  %tobool = icmp eq ptr %P, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 0, ptr %P, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void

; CHECK: @foo
; CHECK: beqlr
; CHECK: blr
}

define void @bar(ptr %P, ptr %Q) #0 {
entry:
  %tobool = icmp eq ptr %P, null
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  store i32 0, ptr %P, align 4
  %tobool1 = icmp eq ptr %Q, null
  br i1 %tobool1, label %if.end3, label %if.then2

if.then2:                                         ; preds = %if.then
  store i32 1, ptr %Q, align 4
  br label %if.end3

if.else:                                          ; preds = %entry
  store i32 0, ptr %Q, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.then, %if.then2, %if.else
  ret void

; CHECK: @bar
; CHECK: beqlr
; CHECK: blr
}


@.str0 = private unnamed_addr constant [2 x i8] c"a\00"
@.str1 = private unnamed_addr constant [2 x i8] c"b\00"
@.str2 = private unnamed_addr constant [2 x i8] c"c\00"
@.str3 = private unnamed_addr constant [2 x i8] c"d\00"
@.str4 = private unnamed_addr constant [2 x i8] c"e\00"
define ptr @dont_assert(i32 %x) {
; LLVM would assert due to moving an early return into the jump table block and
; removing one of its predecessors despite that block ending with an indirect
; branch.
entry:
  switch i32 %x, label %sw.epilog [
    i32 1, label %return
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
    i32 255, label %sw.bb4
  ]
sw.bb1: br label %return
sw.bb2: br label %return
sw.bb3: br label %return
sw.bb4: br label %return
sw.epilog: br label %return
return:
  %retval.0 = phi ptr [ null, %sw.epilog ],
                      [ @.str4, %sw.bb4 ],
                      [ @.str3, %sw.bb3 ],
                      [ @.str2, %sw.bb2 ],
                      [ @.str1, %sw.bb1 ],
                      [ @.str0, %entry ]
  ret ptr %retval.0
}

attributes #0 = { nounwind }
