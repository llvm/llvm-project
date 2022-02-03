; RUN: opt -S -passes=newgvn %s | FileCheck %s

; CHECK: define void @tinkywinky() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %d = alloca i32, align 4
; CHECK-NEXT:   store i32 0, i32* null, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK: for.cond:                                         ; preds = %if.end, %entry
; CHECK-NEXT:   %0 = load i32, i32* null, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %0, 1
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %while.cond
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %1 = load i32, i32* @a, align 4
; CHECK-NEXT:   store i32 %1, i32* %d, align 4
; CHECK-NEXT:   br label %L
; CHECK: L:                                                ; preds = %if.then, %for.body
; CHECK-NEXT:   %tobool = icmp ne i32 %1, 0
; CHECK-NEXT:   br i1 %tobool, label %if.then, label %if.end
; CHECK: if.then:                                          ; preds = %L
; CHECK-NEXT:   call void (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @patatino, i32 0, i32 0))
; CHECK-NEXT:   br label %L
; CHECK: if.end:                                           ; preds = %L
; CHECK-NEXT:   br label %for.cond
; CHECK: while.cond:                                       ; preds = %while.body, %for.cond
; CHECK-NEXT:   br i1 undef, label %while.body, label %while.end
; CHECK: while.body:                                       ; preds = %while.cond
; CHECK-NEXT:   call void (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @patatino, i32 0, i32 0))
; CHECK-NEXT:   br label %while.cond
; CHECK: while.end:
; CHECK-NEXT:   %2 = load i32, i32* @a, align 4
; CHECK-NEXT:   store i32 %2, i32* undef, align 4
; CHECK-NEXT:   ret void

@a = external global i32, align 4
@patatino = external unnamed_addr constant [2 x i8], align 1
define void @tinkywinky() {
entry:
  %d = alloca i32, align 4
  store i32 0, i32* null, align 4
  br label %for.cond
for.cond:
  %0 = load i32, i32* null, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.body, label %while.cond
for.body:
  %1 = load i32, i32* @a, align 4
  store i32 %1, i32* %d, align 4
  br label %L
L:
  %2 = load i32, i32* %d, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %if.then, label %if.end
if.then:
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @patatino, i32 0, i32 0))
  br label %L
if.end:
  br label %for.cond
while.cond:
  br i1 undef, label %while.body, label %while.end
while.body:
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @patatino, i32 0, i32 0))
  br label %while.cond
while.end:
  %3 = load i32, i32* @a, align 4
  store i32 %3, i32* undef, align 4
  ret void
}
declare void @printf(i8*, ...) #1
