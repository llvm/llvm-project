; RUN: not llc -mtriple=aarch64-pc-unknown-xcoff -filetype=null %s 2>&1 | FileCheck -check-prefix=OBJFORMAT %s
; RUN: not llc -mtriple=aarch64-pc-unknown-goff -filetype=null %s 2>&1 | FileCheck -check-prefix=OBJFORMAT %s

; RUN: not llc -mtriple=aarch64-unknown-linux-coff -filetype=null %s 2>&1 | FileCheck -check-prefix=MCINIT %s
; CHECK: LLVM ERROR: cannot initialize MC for non-Windows COFF object files

; Make sure there is no crash or assert with unexpected object
; formats.

; OBJFORMAT: LLVM ERROR: unsupported object format
; MCINIT: LLVM ERROR: cannot initialize MC for non-Windows COFF object files
define void @foo() {
  ret void
}
