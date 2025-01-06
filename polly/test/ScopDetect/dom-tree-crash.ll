; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s

; CHECK: Detected Scops in Function foo

; This unit test case is to check if the following IR does not crash in isHoistableLoad function during Scop Detection.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnueabi"

define void @foo(ptr %block) {
entry:
  br label %for.body

for.cond1.preheader:                              ; preds = %for.body
  %0 = load ptr, ptr null, align 8
  %1 = load i16, ptr %block, align 2
  %2 = load i16, ptr %0, align 2
  br label %foo.exit

for.body:                                         ; preds = %for.body, %entry
  br i1 false, label %for.cond1.preheader, label %for.body

foo.exit:                                     ; preds = %for.cond1.preheader
  ret void
}

define void @init_foo() {
entry:
  store ptr null, ptr null, align 8
  ret void
}
