; Regression test for pgo-instr-gen not invalidating GlobalsAA.
; RUN: opt < %s -passes='require<globals-aa>,pgo-instr-gen,function-attrs' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @callee(i32 %x) mustprogress willreturn {
entry:
  ret i32 42
}

; CHECK: define{{.*}} @caller(i32 %y) #[[ATTR:[0-9]+]]
; CHECK: call void @llvm.instrprof.increment
; CHECK: attributes #[[ATTR]] = { mustprogress nounwind willreturn }
define i32 @caller(i32 %y) mustprogress willreturn {
entry:
  %v = call i32 @callee(i32 %y)
  ret i32 0
}
