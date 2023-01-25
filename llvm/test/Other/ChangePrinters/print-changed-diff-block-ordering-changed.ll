; RUN: opt -passes=jump-threading %s -disable-output --print-changed=diff 2>&1 | FileCheck %s

; CHECK: IR Dump After JumpThreadingPass

define void @f(i1 %0) {
  br i1 %0, label %5, label %2

2:                                                ; preds = %1
  br i1 false, label %b, label %3

3:                                                ; preds = %2
  %4 = call i64 null()
  br label %b

b:                                                ; preds = %3, %2
  br label %5

5:                                                ; preds = %b, %1
  ret void
}
