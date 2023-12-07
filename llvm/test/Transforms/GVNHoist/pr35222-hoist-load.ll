; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s
; CHECK-LABEL: build_tree
; CHECK: load
; CHECK: load
; Check that the load is not hoisted because the call can potentially
; modify the global

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@heap = external global i32, align 4

define i32 @build_tree() unnamed_addr {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %tmp9 = load i32, ptr @heap, align 4
  %cmp = call i1 @pqdownheap(i32 %tmp9)
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %tmp20 = load i32, ptr @heap, align 4
  ret i32 %tmp20
}

declare i1 @pqdownheap(i32)

@i = external hidden unnamed_addr global i32, align 4
@j = external hidden unnamed_addr global [573 x i32], align 4
@v = external global i1

; CHECK-LABEL: test
; CHECK-LABEL: do.end
; CHECK: load
; Check that the load is not hoisted because the call can potentially
; modify the global

define i32 @test() {
entry:
  br label %for.cond

for.cond:
  %a3 = load volatile i1, ptr @v
  br i1 %a3, label %for.body, label %while.end

for.body:
  br label %if.then

if.then:
  %tmp4 = load i32, ptr @i, align 4
  br label %for.cond

while.end:
  br label %do.body

do.body:
  %tmp9 = load i32, ptr getelementptr inbounds ([573 x i32], ptr @j,
i32 0, i32 1), align 4
  %tmp10 = load i32, ptr @i, align 4
  call void @fn()
  %a1 = load volatile i1, ptr @v
  br i1 %a1, label %do.body, label %do.end

do.end:
  %tmp20 = load i32, ptr getelementptr inbounds ([573 x i32], ptr @j,
i32 0, i32 1), align 4
  ret i32 %tmp20
}

declare void @fn()
