; RUN: not llc -mtriple=aarch64-pc-unknown-xcoff -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64-pc-unknown-goff -filetype=null %s 2>&1 | FileCheck %s

; Make sure there is no crash or assert with unexpected object
; formats.

; CHECK: LLVM ERROR: unsupported object format
define void @foo() {
  ret void
}
