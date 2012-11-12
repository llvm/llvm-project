; RUN: llvm-as < %s | llvm-dis


define void @foo() {
  bitcast i32* null to i32*
  ret void
}
