; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s


define void @foo() {
  bitcast ptr null to ptr
  ret void
}
