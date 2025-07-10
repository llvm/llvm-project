; RUN: not llc -mtriple=aarch64-unknown-linux-coff -filetype=null %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: cannot initialize MC for non-Windows COFF object files
define void @foo() {
  ret void
}
