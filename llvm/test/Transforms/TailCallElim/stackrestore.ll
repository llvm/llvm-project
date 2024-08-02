; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

target datalayout = "E-m:a-Fi32-i64:64-p:32:32-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

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
