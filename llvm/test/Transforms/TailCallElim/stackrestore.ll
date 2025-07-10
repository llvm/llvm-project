; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

define void @foo() {
; CHECK-LABEL: define void @foo()
; CHECK-NOT:   tail call void @llvm.stackrestore.p0
;
entry:
  %0 = call ptr @llvm.stacksave.p0()
  call void @llvm.stackrestore.p0(ptr %0)
  ret void
}

declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
