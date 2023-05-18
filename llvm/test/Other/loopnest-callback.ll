;RUN: opt -disable-output -passes=loop-interchange -print-after-all < %s 2>&1 | FileCheck %s

; CHECK: IR Dump After LoopInterchangePass
define void @foo() {
entry:
  br label %for.cond
for.cond:
  br label %for.cond
}
