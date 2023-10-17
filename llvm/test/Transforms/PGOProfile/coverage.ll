; RUN: opt < %s -passes=pgo-instr-gen -pgo-function-entry-coverage -S | FileCheck %s --implicit-check-not="instrprof.cover" --check-prefixes=CHECK,ENTRY
; RUN: opt < %s -passes=pgo-instr-gen -pgo-block-coverage -S | FileCheck %s --implicit-check-not="instrprof.cover" --check-prefixes=CHECK,BLOCK
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %c = call i1 @choice()
  br i1 %c, label %if.then, label %if.else

; CHECK-LABEL: if.then:
if.then:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.else:
if.else:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.end:
if.end:
  ret void
}

define void @bar() {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %c = call i1 @choice()
  br i1 %c, label %if.then, label %if.end

; CHECK-LABEL: if.then:
if.then:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.end:
if.end:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  ret void
}

define void @goo() {
; CHECK-LABEL: entry:
entry:
  ; CHECK: call void @llvm.instrprof.cover({{.*}})
  ret void
}

define void @loop() {
; CHECK-LABEL: entry:
entry:
  ; CHECK: call void @llvm.instrprof.cover({{.*}})
  br label %while
while:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %while
}

; Function Attrs: noinline nounwind ssp uwtable
define void @hoo(i32 %a) #0 {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %rem = srem i32 %0, 2
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

; CHECK-LABEL: if.then:
if.then:                                          ; preds = %entry
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.else:
if.else:                                          ; preds = %entry
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.end:
if.end:                                           ; preds = %if.else, %if.then
  store i32 1, i32* %i, align 4
  br label %for.cond

; CHECK-LABEL: for.cond:
for.cond:                                         ; preds = %for.inc, %if.end
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %a.addr, align 4
  %cmp1 = icmp slt i32 %1, %2
  br i1 %cmp1, label %for.body, label %for.end

; CHECK-LABEL: for.body:
for.body:                                         ; preds = %for.cond
  %3 = load i32, i32* %a.addr, align 4
  %rem2 = srem i32 %3, 3
  %cmp3 = icmp eq i32 %rem2, 0
  br i1 %cmp3, label %if.then4, label %if.else5

; CHECK-LABEL: if.then4:
if.then4:                                         ; preds = %for.body
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end10

; CHECK-LABEL: if.else5:
if.else5:                                         ; preds = %for.body
  %4 = load i32, i32* %a.addr, align 4
  %rem6 = srem i32 %4, 1001
  %cmp7 = icmp eq i32 %rem6, 0
  br i1 %cmp7, label %if.then8, label %if.end9

; CHECK-LABEL: if.then8:
if.then8:                                         ; preds = %if.else5
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %return

; CHECK-LABEL: if.end9:
if.end9:                                          ; preds = %if.else5
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end10

; CHECK-LABEL: if.end10:
if.end10:                                         ; preds = %if.end9, %if.then4
  br label %for.inc

; CHECK-LABEL: for.inc:
for.inc:                                          ; preds = %if.end10
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

; CHECK-LABEL: for.end:
for.end:                                          ; preds = %for.cond
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %return

; CHECK-LABEL: return:
return:                                           ; preds = %for.end, %if.then8
  ret void
}

declare i1 @choice()

; CHECK: declare void @llvm.instrprof.cover({{.*}})
