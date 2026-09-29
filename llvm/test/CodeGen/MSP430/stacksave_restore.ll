; RUN: llc < %s -mtriple=msp430

target triple = "msp430"

define void @foo() {
entry:
  %0 = tail call ptr @llvm.stacksave()
  tail call void @llvm.stackrestore(ptr %0)
  ret void
}

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)
