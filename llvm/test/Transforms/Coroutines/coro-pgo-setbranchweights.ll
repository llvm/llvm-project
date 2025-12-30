; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata
; RUN: opt < %t/a.ll --passes=pgo-instr-use -pgo-test-profile-file=%t/a.profdata

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

define void @_bar() presplitcoroutine personality ptr null {
  %1 = call token @llvm.coro.save(ptr null)
  %2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %2, label %5 [
    i8 0, label %3
    i8 1, label %4
  ]

3:                                                ; preds = %0
  ret void

4:                                                ; preds = %0
  ret void

5:                                                ; preds = %0
  ret void
}

declare token @llvm.coro.save(ptr)

declare i8 @llvm.coro.suspend(token, i1)

;--- a.proftext
# IR level Instrumentation Flag
:ir

_bar
# Func Hash:
1063705160175073211
# Num Counters:
2
1
0
