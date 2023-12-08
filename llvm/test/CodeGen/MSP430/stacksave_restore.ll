; RUN: llc < %s -march=msp430

target triple = "msp430"

define void @foo() {
entry:
  %0 = tail call i8* @llvm.stacksave()
  tail call void @llvm.stackrestore(i8* %0)
  ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
