; RUN: opt < %s -passes=pgo-instr-gen -pgo-function-entry-coverage -S | FileCheck %s --implicit-check-not="instrprof.cover" --check-prefixes=CHECK,GEN,ENTRY
; RUN: opt < %s -passes=pgo-instr-gen -pgo-block-coverage -S | FileCheck %s --implicit-check-not="instrprof.cover" --check-prefixes=CHECK,GEN,BLOCK

; RUN: llvm-profdata merge %S/Inputs/coverage.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefixes=CHECK,USE
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @foo()
; USE-SAME: !prof ![[HOT:[0-9]+]]
define void @foo() {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %c = call i1 @choice()
  br i1 %c, label %if.then, label %if.else
  ; USE: br i1 %c, label %if.then, label %if.else, !prof ![[WEIGHTS0:[0-9]+]]

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

; CHECK-LABEL: @bar()
; USE-SAME: !prof ![[HOT:[0-9]+]]
define void @bar() {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %c = call i1 @choice()
  br i1 %c, label %if.then, label %if.end
  ; USE: br i1 %c, label %if.then, label %if.end, !prof ![[WEIGHTS1:[0-9]+]]

; CHECK-LABEL: if.then:
if.then:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end

; CHECK-LABEL: if.end:
if.end:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  ret void
}

; CHECK-LABEL: @goo()
; USE-SAME: !prof ![[HOT:[0-9]+]]
define void @goo() {
; CHECK-LABEL: entry:
entry:
  ; GEN: call void @llvm.instrprof.cover({{.*}})
  ret void
}

; CHECK-LABEL: @loop()
; USE-SAME: !prof ![[HOT:[0-9]+]]
define void @loop() {
; CHECK-LABEL: entry:
entry:
  ; GEN: call void @llvm.instrprof.cover({{.*}})
  br label %while
while:
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %while
}

; CHECK-LABEL: @hoo(
; USE-SAME: !prof ![[HOT:[0-9]+]]
define void @hoo(i32 %a) #0 {
; CHECK-LABEL: entry:
entry:
  ; ENTRY: call void @llvm.instrprof.cover({{.*}})
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %rem = srem i32 %0, 2
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.else
  ; USE: br i1 %cmp, label %if.then, label %if.else, !prof ![[WEIGHTS1]]

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
  store i32 1, ptr %i, align 4
  br label %for.cond

; CHECK-LABEL: for.cond:
for.cond:                                         ; preds = %for.inc, %if.end
  %1 = load i32, ptr %i, align 4
  %2 = load i32, ptr %a.addr, align 4
  %cmp1 = icmp slt i32 %1, %2
  br i1 %cmp1, label %for.body, label %for.end
  ; USE: br i1 %cmp1, label %for.body, label %for.end, !prof ![[WEIGHTS1]]

; CHECK-LABEL: for.body:
for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %a.addr, align 4
  %rem2 = srem i32 %3, 3
  %cmp3 = icmp eq i32 %rem2, 0
  br i1 %cmp3, label %if.then4, label %if.else5
  ; USE: br i1 %cmp3, label %if.then4, label %if.else5, !prof ![[WEIGHTS0]]

; CHECK-LABEL: if.then4:
if.then4:                                         ; preds = %for.body
  ; BLOCK: call void @llvm.instrprof.cover({{.*}})
  br label %if.end10

; CHECK-LABEL: if.else5:
if.else5:                                         ; preds = %for.body
  %4 = load i32, ptr %a.addr, align 4
  %rem6 = srem i32 %4, 1001
  %cmp7 = icmp eq i32 %rem6, 0
  br i1 %cmp7, label %if.then8, label %if.end9
  ; USE: br i1 %cmp7, label %if.then8, label %if.end9, !prof ![[WEIGHTS1]]

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
  %5 = load i32, ptr %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, ptr %i, align 4
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

; GEN: declare void @llvm.instrprof.cover({{.*}})

; USE-DAG: ![[HOT]] = !{!"function_entry_count", i64 10000}
; USE-DAG: ![[WEIGHTS0]] = !{!"branch_weights", i32 1, i32 1}
; USE-DAG: ![[WEIGHTS1]] = !{!"branch_weights", i32 1, i32 0}
